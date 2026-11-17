"""Run official demos, or generate and score a full raw-backbone GenEval set."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json,bind_run,commit_image,completed,load_prompts,paths,seed_for,sha,validate_outputs


def identity(stage):
    if stage not in [256,512]: raise ValueError('Unknown stage')
    number=1 if stage==256 else 2
    return f'llamagen_fresh_stage{number}_{stage}_backbone',f't2i_XL_stage{number}_{stage}.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['demo','generate','score'],required=True)
    p.add_argument('--fresh-root',type=Path,required=True)
    p.add_argument('--assets-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--stage',type=int,choices=[256,512],default=512)
    p.add_argument('--shards',type=int,default=32)
    p.add_argument('--index',type=int,default=0)
    a=p.parse_args(); f=a.fresh_root.resolve(); source=f/'source'; models=f/'models'
    setup=json.loads((f/'setup_complete.json').read_text())
    method,checkpoint=identity(a.stage); run=f/'runs'/method
    if a.phase=='demo':
        env=os.environ.copy(); env['PYTHONPATH']=str(source)
        for stage in [256,512]:
            name,weight=identity(stage); out=f/'native_images'/name
            out.mkdir(parents=True,exist_ok=True)
            # Execute the original script and its original four prompts unmodified.
            subprocess.run([sys.executable,str(source/'autoregressive/sample/sample_t2i.py'),
                '--gpt-ckpt',str(models/weight),'--vq-ckpt',str(models/'vq_ds16_t2i.pt'),
                '--t5-path',str(models/'t5-ckpt'),'--gpt-model','GPT-XL','--image-size',str(stage),
                '--precision','bf16','--seed','42'],cwd=out,env=env,check=True)
            if not (out/'sample_t2i.png').is_file(): raise ValueError('Official demo did not save its grid')
            print('LLAMAGEN_FRESH_NATIVE_IMAGE',out/'sample_t2i.png',flush=True)
        atomic_json(f/'native_images/completed.json',{'status':'official_demos_completed',
            'stages':[256,512],'prompts_per_stage':4,'quality_verified':False,
            'notice':'Inspect both grids; completion alone does not prove prompt alignment'})
        return
    if a.phase=='score':
        from merge_full import merge
        merge(a.assets_root,run,method,a.shards)
        common=['--root',str(a.assets_root),'--scratch',str(a.data_root),'--out',str(run/'merged'),'--scope','full']
        subprocess.run([sys.executable,str(ROOT/'evaluate_benchmark.py'),*common,'--stage','preflight'],check=True)
        import evaluate_benchmark
        evaluate_benchmark.METHODS=[method]
        sys.argv=[sys.argv[0],*common,'--stage','evaluate']; evaluate_benchmark.main()
        print('LLAMAGEN_FRESH_FULL_SCORE_OK',method,flush=True); return
    if not 0<=a.index<a.shards<=553: raise ValueError('Invalid shard')
    rows=load_prompts(a.assets_root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    ids=list(range(a.index,553,a.shards)); out=run/'shards'/f'{method}_{a.index}'
    spec={'scope':'full_shard','method':method,'shards':a.shards,'shard_index':a.index,'prompt_ids':ids,
        'setup':setup,'environment_freeze_sha256':sha(f/'environment.freeze.txt'),
        'runner_sha256':sha(Path(__file__)),'seed_rule':'42 + 4*original_prompt_id + sample_id',
        'samples_per_prompt':4,'resolution':a.stage,'cfg':7.5,'top_k':1000,'top_p':1.0,
        'temperature':1.0,'dtype':'bf16','tf32':True,'inchworm':False}
    bind_run(out,spec)
    sys.path.insert(0,str(source))
    import torch
    import numpy as np
    from PIL import Image
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
    from tokenizer.tokenizer_image.vq_model import VQ_models
    if not torch.cuda.is_available(): raise RuntimeError('Use an allocated GPU')
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    latent=a.stage//16
    with torch.inference_mode():
        vq=VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).cuda().eval()
        vq.load_state_dict(torch.load(models/'vq_ds16_t2i.pt',map_location='cpu')['model'],strict=True)
        gpt=GPT_models['GPT-XL'](block_size=latent**2,cls_token_num=120,model_type='t2i').cuda().bfloat16().eval()
        state=torch.load(models/checkpoint,map_location='cpu')
        key=next(k for k in ['model','module','state_dict'] if k in state)
        gpt.load_state_dict(state[key],strict=True); del state
        t5=T5Embedder(device='cuda',local_cache=True,cache_dir=str(models/'t5-ckpt'),
            dir_or_name='flan-t5-xl',torch_dtype=torch.bfloat16,model_max_length=120)
        for i in ids:
            features,masks=t5.get_text_embeddings([rows[i]['prompt']])
            valid=int(masks[0].sum().item())
            features=torch.cat([features[:,valid:],features[:,:valid]],dim=1)
            masks=torch.flip(masks,[-1]); features=features*masks[:,:,None]
            for j in range(4):
                image_path,receipt=paths(out,method,i,j)
                if completed(image_path,receipt,spec,rows[i]):
                    print('SKIP_COMPLETE',i,j,flush=True); continue
                seed=seed_for(i,j); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                torch.cuda.synchronize(); start=time.perf_counter()
                codes=generate(gpt,features,latent**2,masks,cfg_scale=7.5,temperature=1.,top_k=1000,top_p=1.,sample_logits=True)
                decoded=vq.decode_code(codes,[1,8,latent,latent])
                pixels=decoded[0].float().cpu().numpy().transpose(1,2,0)
                image=Image.fromarray(np.clip((pixels+1)*127.5+0.5,0,255).astype(np.uint8))
                commit_image(out,method,i,j,rows[i],spec,image,{'seed':seed,'seconds':time.perf_counter()-start,
                    'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'backtracks':0})
                del decoded,codes,image
                print('FRESH_LLAMA_IMAGE_OK',i,j,flush=True)
        count=validate_outputs(out,method,rows,ids,4)
    atomic_json(out/'completed.json',{'images':count,'status':'completed_shard'})
    print('LLAMAGEN_FRESH_SHARD_OK',a.index,count,flush=True)


if __name__=='__main__': main()

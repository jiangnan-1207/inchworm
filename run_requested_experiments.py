"""Batch-one profiling and thirty qualitative pairs per model; no GenEval score."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

from experiment_plan import pair_prompts_for_model, select_100, summarize
from inchworm_core import Config, generate
from protocol import atomic_json, bind_run, load_prompts, sha


def load_model(name,root,data):
    import torch
    import numpy as np
    from PIL import Image
    if name.startswith('janus'):
        from transformers import AutoModelForCausalLM
        from janus.models import VLChatProcessor, MultiModalityCausalLM
        from janus_backend import JanusBackend, State
        directory=data/'models'/('Janus-Pro-1B' if name=='janus1b' else 'Janus-Pro-7B')
        processor=VLChatProcessor.from_pretrained(str(directory),local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(str(directory),trust_remote_code=True,
            local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval()

        def initial(text):
            prompt=processor.apply_sft_template_for_multi_turn_prompts(conversations=[
                {'role':'<|User|>','content':text},{'role':'<|Assistant|>','content':''}],
                sft_format=processor.sft_format,system_prompt='')+processor.image_start_tag
            ids=processor.tokenizer.encode(prompt)
            tokens=torch.tensor([ids,ids],device='cuda'); tokens[1,1:-1]=processor.pad_id
            return model.language_model.get_input_embeddings()(tokens)

        def run(text,method):
            inputs=initial(text)
            torch.cuda.synchronize()
            start=time.perf_counter()
            if method=='inchworm':
                result=generate(JanusBackend(model),State(prompt_inputs=inputs),576,Config())
                tokens=result['tokens']; backtracks=result['backtracks']
            else:
                cache=None; tokens=[]
                for _ in range(576):
                    output=model.language_model.model(inputs_embeds=inputs,past_key_values=cache,use_cache=True)
                    cache=output.past_key_values
                    logits=model.gen_head(output.last_hidden_state[:,-1,:])
                    p=torch.softmax(logits[1:2]+5*(logits[0:1]-logits[1:2]),dim=-1)
                    z=torch.multinomial(p,1); tokens.append(int(z.item()))
                    inputs=model.prepare_gen_img_embeds(z[:,0].repeat_interleave(2)).unsqueeze(1)
                backtracks=0
                del cache,output,logits,p,z
            torch.cuda.synchronize(); token_seconds=time.perf_counter()-start
            codes=torch.tensor([tokens],device='cuda',dtype=torch.int)
            decoded=model.gen_vision_model.decode_code(codes,shape=[1,8,24,24])
            pixels=decoded.float().cpu().numpy().transpose(0,2,3,1)[0]
            image=Image.fromarray(np.clip((pixels+1)*127.5,0,255).astype(np.uint8))
            return image,token_seconds,backtracks
        # Check both 1B and 7B against their own native loop before profiling.
        with torch.inference_mode():
            torch.manual_seed(123)
            inputs=initial('a photo of a red apple'); cache=None; reference=[]
            for _ in range(32):
                output=model.language_model.model(inputs_embeds=inputs,past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                logits=model.gen_head(output.last_hidden_state[:,-1,:])
                p=torch.softmax(logits[1:2]+5*(logits[0:1]-logits[1:2]),dim=-1)
                z=torch.multinomial(p,1); reference.append(int(z.item()))
                inputs=model.prepare_gen_img_embeds(z[:,0].repeat_interleave(2)).unsqueeze(1)
            torch.manual_seed(123)
            actual=generate(JanusBackend(model),State(prompt_inputs=initial('a photo of a red apple')),
                32,Config(max_backtracks=0))['tokens']
            if actual!=reference: raise AssertionError('Janus disabled-controller native parity failed')
            del cache,output,inputs,logits,p,z
        print('JANUS_NATIVE_PARITY_OK',name,flush=True)
        return run

    sys.path.insert(0,str(root/'benchmarks/LlamaGen'))
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate as native_generate
    from language.t5 import T5Embedder
    from llamagen_backend import LlamaGenBackend, State, check_native_prefixes
    directory=data/'models/LlamaGen'
    vq=VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).cuda().eval()
    checkpoint=torch.load(directory/'vq_ds16_t2i.pt',map_location='cpu',weights_only=False)
    vq.load_state_dict(checkpoint['model'],strict=True); del checkpoint
    model=GPT_models['GPT-XL'](block_size=1024,cls_token_num=120,model_type='t2i').to(device='cuda',dtype=torch.bfloat16).eval()
    checkpoint=torch.load(directory/'t2i_XL_stage2_512.pt',map_location='cpu',weights_only=False)
    key=next(k for k in ['model','module','state_dict'] if k in checkpoint)
    model.load_state_dict(checkpoint[key],strict=True); del checkpoint
    encoder=T5Embedder(device='cuda',local_cache=True,cache_dir=str(directory/'t5-ckpt'),
        dir_or_name='flan-t5-xl',torch_dtype=torch.bfloat16,model_max_length=120)

    def condition(text):
        embedding,masks=encoder.get_text_embeddings([text]); valid=int(masks[0].sum().item())
        embedding=torch.cat([embedding[:,valid:,:],embedding[:,:valid,:]],dim=1)
        masks=torch.flip(masks,[-1]); return embedding*masks[:,:,None],masks

    # Hard gate before any reported samples. It also checks a restored prefix.
    with torch.inference_mode():
        cond,masks=condition('a photo of a red apple')
        check_native_prefixes(model,cond,masks)
        del cond,masks

    def run(text,method):
        cond,masks=condition(text)
        torch.cuda.synchronize(); start=time.perf_counter()
        if method=='inchworm':
            backend=LlamaGenBackend(model,cond,masks)
            try:
                result=generate(backend,State(),1024,Config())
                tokens=torch.tensor([result['tokens']],device='cuda',dtype=torch.int)
                backtracks=result['backtracks']
            finally:
                backend.close()
        else:
            tokens=native_generate(model,cond,1024,masks,cfg_scale=7.5,
                temperature=1.0,top_k=1000,top_p=1.0,sample_logits=True)
            backtracks=0
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        decoded=vq.decode_code(tokens,[1,8,32,32])
        pixels=decoded.float().cpu().numpy().transpose(0,2,3,1)[0]
        image=Image.fromarray(np.clip((pixels+1)*127.5+0.5,0,255).astype(np.uint8))
        for layer in model.layers: layer.attention.kv_cache=None
        model.causal_mask=None
        return image,seconds,backtracks
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen','janus7b'],required=True)
    p.add_argument('--scope',choices=['profile','pairs','both'],default='both')
    args=p.parse_args()
    import torch
    import numpy as np
    assert torch.cuda.is_available(),'Use an allocated GPU compute node'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=load_prompts(args.root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    ids=select_100(rows)
    group={'janus1b':0,'llamagen':1,'janus7b':2}[args.model]
    pairs=pair_prompts_for_model(args.model)
    out=args.out/args.model
    profiling=args.scope in ['profile','both'] and args.model!='janus7b'
    qualitative=args.scope in ['pairs','both']
    if not profiling and not qualitative: raise ValueError('This original task profiles only Janus1B and LlamaGen')
    specification={'model':args.model,'scope':args.scope,'batch_size':1,'dtype':'bf16','config':asdict(Config()),
        'profiling_prompt_ids':ids if profiling else [],'pairs':pairs if qualitative else [],
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'source_sha256':{f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'asset_revisions':json.loads((args.root/'benchmarks/setup_records/model_revisions_v1.json').read_text()),
        'timing':'batch1; synchronized; text preparation + image tokens + decode + CPU pixel transfer; excludes model load and PNG I/O',
        'warmup_images_per_method':1,'llama_cache':'functional immutable live prefixes; native-capacity padded attention; engineering implementation'}
    # JSON round-trip makes tuple pairs stable during resume comparison.
    specification=json.loads(json.dumps(specification))
    bind_run(out,specification)
    with torch.inference_mode():
        run=load_model(args.model,args.root,args.data_root)
        def measure(prompt,method,seed):
            gc.collect(); torch.cuda.empty_cache()
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            image,seconds,backtracks=run(prompt,method)
            torch.cuda.synchronize(); latency=time.perf_counter()-start
            return image,{'token_seconds':seconds,'latency_seconds':latency,'backtracks':backtracks,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        for method in ['backbone','inchworm']:
            image,_=measure(pairs[0][1],method,0); del image
        records=[]
        cases=[]
        if profiling:
            cases.extend(('profile',i,rows[i]['prompt'],42+4*i) for i in ids)
        if qualitative: cases.extend(('pairs',i,text,10042+i) for i,text in pairs)
        for phase,i,text,seed in cases:
            # Alternate execution order to reduce a systematic first/second bias.
            methods=['backbone','inchworm'] if i%2==0 else ['inchworm','backbone']
            for method in methods:
                path=out/method/phase/f'{i:05d}.json'; png=path.with_suffix('.png')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['specification']!=specification or record['image_sha256']!=sha(png):
                        raise ValueError('Resume record or image differs')
                else:
                    image,diagnostics=measure(text,method,seed)
                    png.parent.mkdir(parents=True,exist_ok=True)
                    temp=png.with_suffix('.png.part'); image.save(temp,format='PNG'); temp.replace(png)
                    record={'phase':phase,'sample_id':i,'prompt':text,'seed':seed,'method':method,
                        'specification':specification,'image_sha256':sha(png),**diagnostics}
                    atomic_json(path,record); del image
                if phase=='profile': records.append(record)
                print('SAMPLE_OK',args.model,phase,i,method,record['latency_seconds'],record['backtracks'],flush=True)
    if profiling:
        summary=summarize(records,[(m,i) for m in ['backbone','inchworm'] for i in ids])
        atomic_json(out/'profile_summary.json',{'specification':specification,'results':summary})
    atomic_json(out/'completed.json',{'profiling_images':len(records),'qualitative_pairs':len(pairs) if qualitative else 0,
        'qualitative_images':2*len(pairs) if qualitative else 0,'status':'completed'})
    print('REQUESTED_EXPERIMENTS_OK',args.model,flush=True)


if __name__=='__main__': main()

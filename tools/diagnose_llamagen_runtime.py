"""Controlled native LlamaGen precision/conditioning probe; no benchmark results."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json,sha

PROMPTS=['a photo of a bench','a photo of a red apple','a photo of a dog']
SOURCE_REVISION='ce98ec41803a74a90ce68c40ababa9eaeffeb4ec'


def prepare_condition(features,masks):
    """Exactly the upstream sample_t2i.py left-padding and zero masking."""
    import torch
    shifted=[]
    for feature,mask in zip(features,masks):
        valid=int(mask.sum().item())
        shifted.append(torch.cat([feature[valid:],feature[:valid]]))
    masks=torch.flip(masks,[-1])
    return torch.stack(shifted)*masks[:,:,None],masks


def feature_stats(features,masks):
    import torch
    values=features.float()[masks.bool()]
    pooled=(features.float()*masks[:,:,None]).sum(1)/masks.sum(1,keepdim=True)
    normalized=torch.nn.functional.normalize(pooled,dim=-1)
    return {'finite':bool(torch.isfinite(features).all().item()),
        'valid_mean':values.mean().item(),'valid_std':values.std().item(),
        'valid_abs_max':values.abs().max().item(),
        'pooled_prompt_cosines':(normalized@normalized.T).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','data-root','asset-report','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.asset_report.read_text(encoding='utf-8'))
    if report['status']!='assets_match_pinned_sources': raise ValueError('Pass the asset audit first')
    source=args.source_root.resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_REVISION:
        raise ValueError('Unexpected official source revision')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True)
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Use a new empty runtime probe directory')
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    import torch
    import torchvision
    import transformers
    from transformers import AutoTokenizer,T5EncoderModel
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate
    from tokenizer.tokenizer_image.vq_model import VQ_models
    if not torch.cuda.is_available(): raise RuntimeError('Run in an allocated GPU job')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    directory=args.data_root/'models/LlamaGen'
    t5_path=directory/'t5-ckpt/flan-t5-xl'
    result={'status':'started','scope':'diagnostic only; not a benchmark or performance run',
        'prompts':PROMPTS,'seed_per_case':42,'batch_size':3,'resolution':512,'source_revision':SOURCE_REVISION,
        'asset_audit':str(args.asset_report),'asset_report_sha256':sha(args.asset_report),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'packages':{n:importlib.metadata.version(n) for n in ['accelerate','tokenizers','sentencepiece']},
        'module_paths':{'torch':torch.__file__,'transformers':transformers.__file__},
        'cfg':7.5,'top_k':1000,'top_p':1.0,'temperature':1.0,
        'tf32':False,'cases':[],'loading':{},
        'notice':'Per-case RNG reset differs from the old demo initialization; no pixel identity is asserted. '
                 'A completed probe does not establish which images follow their prompts.'}
    target=args.out/'runtime_report.json'
    atomic_json(target,result)
    original_load=T5EncoderModel.from_pretrained
    encoded={}
    with torch.inference_mode():
        for name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
            def audited_load(*positional,**kwargs):
                model,info=original_load(*positional,**{**kwargs,'output_loading_info':True})
                result['loading']['t5_'+name]={
                    'missing_keys':info.get('missing_keys',[]),'mismatched_keys':info.get('mismatched_keys',[]),
                    'error_msgs':info.get('error_msgs',[]),
                    'unexpected_keys_count':len(info.get('unexpected_keys',[])),
                    'unexpected_keys_sample':info.get('unexpected_keys',[])[:12]}
                return model
            print('ENCODE_START',name,flush=True)
            # Keep the official T5Embedder's preprocessing and loading/device map.
            # Reload from original disk weights for FP32; upcasting rounded BF16
            # weights would not be a valid FP32 control.
            with patch.object(T5EncoderModel,'from_pretrained',side_effect=audited_load):
                encoder=T5Embedder(device='cuda',local_cache=True,cache_dir=str(directory/'t5-ckpt'),
                    dir_or_name='flan-t5-xl',torch_dtype=dtype,model_max_length=120)
            features,masks=encoder.get_text_embeddings(PROMPTS)
            result['loading']['t5_'+name]['feature_stats']=feature_stats(features,masks)
            if not torch.isfinite(features).all(): raise ValueError('Nonfinite T5 features')
            encoded[name]=(features.cpu(),masks.cpu())
            if name=='bf16':
                texts=[encoder.text_preprocessing(t) for t in PROMPTS]
                current=encoder.tokenizer(texts,max_length=120,padding='max_length',truncation=True,
                    return_attention_mask=True,add_special_tokens=True,return_tensors='pt')
                slow=AutoTokenizer.from_pretrained(str(t5_path),use_fast=False,local_files_only=True)
                slow_tokens=slow(texts,max_length=120,padding='max_length',truncation=True,
                    return_attention_mask=True,add_special_tokens=True,return_tensors='pt')
                equal=all(torch.equal(current[k],slow_tokens[k]) for k in ['input_ids','attention_mask'])
                result['tokenizer']={'current_class':type(encoder.tokenizer).__name__,
                    'slow_class':type(slow).__name__,'same_ids_and_masks_for_probe_prompts':equal,
                    'cleaned_prompts':texts,'current_valid_ids':[
                        ids[mask.bool()].tolist() for ids,mask in zip(current['input_ids'],current['attention_mask'])],
                    'slow_valid_ids':[
                        ids[mask.bool()].tolist() for ids,mask in zip(slow_tokens['input_ids'],slow_tokens['attention_mask'])]}
                print('TOKENIZER_CURRENT_EQUALS_SLOW',equal,flush=True)
                if not equal:
                    encoder.tokenizer=slow
                    slow_features,slow_masks=encoder.get_text_embeddings(PROMPTS)
                    encoded['slow_bf16']=(slow_features.cpu(),slow_masks.cpu())
                    del slow_features,slow_masks
                del current,slow,slow_tokens
            print('T5_LOAD_INFO',name,json.dumps(result['loading']['t5_'+name]),flush=True)
            atomic_json(target,result)
            del encoder,features,masks
            gc.collect(); torch.cuda.empty_cache()
        a,b=encoded['bf16'][0].float(),encoded['fp32'][0].float()
        mask=encoded['bf16'][1].bool()
        result['bf16_vs_fp32_feature_relative_l2']=((a[mask]-b[mask]).norm()/b[mask].norm().clamp_min(1e-12)).item()
        del a,b,mask
        vq=VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).cuda().eval()
        checkpoint=torch.load(directory/'vq_ds16_t2i.pt',map_location='cpu',weights_only=True)
        vq.load_state_dict(checkpoint['model'],strict=True); del checkpoint
        model=GPT_models['GPT-XL'](block_size=1024,cls_token_num=120,model_type='t2i').cuda().eval()
        checkpoint=torch.load(directory/'t2i_XL_stage2_512.pt',map_location='cpu',weights_only=True)
        key=next(k for k in ['model','module','state_dict'] if k in checkpoint)
        weights=checkpoint[key]
        cases=[('default_bf16','bf16',torch.bfloat16),
            ('t5_fp32_gpt_bf16','fp32',torch.bfloat16),('full_fp32','fp32',torch.float32)]
        if 'slow_bf16' in encoded: cases.insert(1,('slow_tokenizer_bf16','slow_bf16',torch.bfloat16))
        for name,feature_key,dtype in cases:
            # Reload for each dtype from the original FP32 checkpoint.
            model.to(dtype=dtype)
            model.load_state_dict(weights,strict=True)
            features,masks=encoded[feature_key]
            cond,condition_masks=prepare_condition(features.to('cuda',dtype=dtype),masks.to('cuda'))
            torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            print('GENERATE_START',name,flush=True)
            tokens=generate(model,cond,1024,condition_masks,cfg_scale=7.5,
                temperature=1.0,top_k=1000,top_p=1.0,sample_logits=True)
            folder=args.out/name; folder.mkdir()
            torch.save(tokens.cpu(),folder/'image_tokens.pt')
            # Decode images individually to bound VQ activation memory.
            images=[]
            for i in range(3):
                decoded=vq.decode_code(tokens[i:i+1],[1,8,32,32]).cpu()
                if not torch.isfinite(decoded).all(): raise ValueError('Nonfinite VQ output')
                images.append(decoded)
                torchvision.utils.save_image(decoded,folder/f'{i:02d}.png',normalize=True,value_range=(-1,1))
                del decoded
            torchvision.utils.save_image(torch.cat(images),folder/'grid.png',nrow=3,normalize=True,value_range=(-1,1))
            result['cases'].append({'name':name,'t5_dtype':'float32' if feature_key=='fp32' else 'bfloat16',
                'gpt_dtype':str(dtype),'vq_dtype':'float32','decode_batch_size':1,
                'images':{p.name:sha(p) for p in sorted(folder.glob('*.png'))}})
            atomic_json(target,result)
            print('RUNTIME_CASE_OK',name,folder/'grid.png',flush=True)
            for layer in model.layers: layer.attention.kv_cache=None
            model.causal_mask=None
            del tokens,cond,condition_masks,images
            gc.collect(); torch.cuda.empty_cache()
    result['status']='completed'; atomic_json(target,result)
    print('LLAMAGEN_RUNTIME_PROBE_OK',args.out,flush=True)


if __name__=='__main__': main()

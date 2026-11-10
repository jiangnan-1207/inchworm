"""VQ reconstruction, teacher-forced text response, and native-cache diagnosis."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json,sha
from tools.diagnose_llamagen_runtime import prepare_condition,PROMPTS,SOURCE_REVISION


def image_prediction_logits(logits,text_length,image_length):
    """Text's last position predicts image token 0; no token may predict itself."""
    selected=logits[:,text_length-1:text_length-1+image_length,:]
    if selected.shape[1]!=image_length: raise ValueError('Wrong teacher-forced alignment')
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','data-root','asset-report','reference-image','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if not args.reference_image.is_file(): raise ValueError('Missing reference image')
    if json.loads(args.asset_report.read_text())['status']!='assets_match_pinned_sources':
        raise ValueError('Pass the pinned asset audit first')
    source=args.source_root.resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_REVISION:
        raise ValueError('Unexpected upstream revision')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True)
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Use a new empty output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    import torch
    import torch.nn.functional as F
    import torchvision
    import numpy as np
    from PIL import Image
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from tokenizer.tokenizer_image.vq_model import VQ_models
    from llamagen_backend import check_native_prefixes
    if not torch.cuda.is_available(): raise RuntimeError('Use an allocated CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    directory=args.data_root/'models/LlamaGen'
    target=args.out/'structure_report.json'
    report={'status':'started','scope':'structural diagnostic, not benchmark',
        'source_revision':SOURCE_REVISION,'prompts':PROMPTS,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'reference_image':str(args.reference_image),
        'reference_sha256':sha(args.reference_image),
        'notice':'A single reference image and token prefix are diagnostic only. '
                 'Caption NLL ranking is not a pass/fail quality metric.'}
    atomic_json(target,report)
    with torch.inference_mode():
        encoder=T5Embedder(device='cuda',local_cache=True,cache_dir=str(directory/'t5-ckpt'),
            dir_or_name='flan-t5-xl',torch_dtype=torch.bfloat16,model_max_length=120)
        features,masks=encoder.get_text_embeddings(PROMPTS)
        features=features.cpu(); masks=masks.cpu()
        del encoder
        gc.collect(); torch.cuda.empty_cache()
        vq=VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).cuda().eval()
        checkpoint=torch.load(directory/'vq_ds16_t2i.pt',map_location='cpu',weights_only=True)
        vq.load_state_dict(checkpoint['model'],strict=True); del checkpoint
        image=Image.open(args.reference_image).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
        pixels=torch.from_numpy(np.array(image).copy()).permute(2,0,1).unsqueeze(0).float().cuda()/127.5-1
        quant,_,info=vq.encode(pixels)
        codes=info[-1].reshape(1,-1)
        if codes.shape[1]!=1024: raise ValueError('Expected 32x32 reference tokens')
        decoded=vq.decode_code(codes,[1,8,32,32])
        direct=vq.decode(quant)
        mse=F.mse_loss((decoded+1)/2,(pixels+1)/2).item()
        report['vq']={'reconstruction_mse_0_1':mse,
            'encode_decode_vs_decode_code_max_abs':(decoded-direct).abs().max().item(),
            'reference_code_min':codes.min().item(),'reference_code_max':codes.max().item()}
        torchvision.utils.save_image(torch.cat([pixels,decoded]),args.out/'vq_roundtrip.png',
            nrow=2,normalize=True,value_range=(-1,1))
        codes=codes.cpu()
        print('VQ_ROUNDTRIP',json.dumps(report['vq']),flush=True)
        atomic_json(target,report)
        del vq,quant,info,decoded,direct,pixels
        gc.collect(); torch.cuda.empty_cache()
        # FP32 structural comparison removes BF16 attention-rounding ambiguity.
        model=GPT_models['GPT-XL'](block_size=1024,cls_token_num=120,model_type='t2i').cuda().eval()
        checkpoint=torch.load(directory/'t2i_XL_stage2_512.pt',map_location='cpu',weights_only=True)
        key=next(k for k in ['model','module','state_dict'] if k in checkpoint)
        model.load_state_dict(checkpoint[key],strict=True); del checkpoint
        cond,masks=prepare_condition(features.float().cuda(),masks.cuda())
        image_tokens=codes.cuda().long().expand(3,-1)
        text=cond.shape[1]; length=image_tokens.shape[1]; size=text+length-1
        mask=torch.tril(torch.ones(size,size,dtype=torch.bool,device='cuda'))[None].repeat(3,1,1)
        mask[:,:,:text]&=masks.bool()[:,None,:]
        diagonal=torch.arange(size,device='cuda'); mask[:,diagonal,diagonal]=True
        positions=torch.arange(size,device='cuda')
        logits,_=model(image_tokens[:,:-1],cond,positions,mask=mask[:,None])
        predictions=image_prediction_logits(logits,text,length)
        losses=F.cross_entropy(predictions.transpose(1,2),image_tokens,reduction='none').mean(1)
        steps=16
        uncached=predictions[:,:steps].softmax(-1).cpu()
        report['teacher_forced_reference_nll']={prompt:loss.item() for prompt,loss in zip(PROMPTS,losses)}
        report['text_response_mean_tv_on_same_image_prefix']={
            str((i,j)):(uncached[i]-uncached[j]).abs().sum(-1).mean().item()/2
            for i,j in [(0,1),(0,2),(1,2)]}
        report['text_rope_prefix_max_abs']=model.freqs_cis[:text].abs().max().item()
        print('TEXT_RESPONSE',json.dumps(report['text_response_mean_tv_on_same_image_prefix']),flush=True)
        print('TEACHER_FORCED_NLL',json.dumps(report['teacher_forced_reference_nll']),flush=True)
        del logits,predictions,losses,mask
        gc.collect(); torch.cuda.empty_cache()
        with torch.device('cuda'):
            model.setup_caches(3,text+1024,model.tok_embeddings.weight.dtype)
        model.causal_mask[:,:,:text]&=masks.bool()[:,None,:]
        diagonal=torch.arange(model.max_seq_length,device='cuda')
        model.causal_mask[:,diagonal,diagonal]=True
        tv=[]
        for t in range(steps):
            if t==0:
                logits,_=model(None,cond,torch.arange(text,device='cuda'))
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
                    logits,_=model(image_tokens[:,t-1:t],None,torch.tensor([text+t-1],device='cuda'))
            probabilities=logits[:,-1].softmax(-1).cpu()
            tv.append(((probabilities-uncached[:,t]).abs().sum(-1)/2).tolist())
        report['native_cache_vs_full_prefix']={'positions':steps,'total_variation_by_position_and_prompt':tv,
            'max_total_variation':max(max(v) for v in tv),'dtype':'float32','cfg_scale':1,
            'comparison':'identical forced image prefixes, full vocabulary, no sampling or top-k'}
        for layer in model.layers: layer.attention.kv_cache=None
        model.causal_mask=None
        print('NATIVE_CACHE_VS_FULL',report['native_cache_vs_full_prefix']['max_total_variation'],flush=True)
        # Record the corrected adapter gate separately; failure must not be
        # hidden behind an overall script-completed marker.
        try:
            check_native_prefixes(model,cond[:1],masks[:1])
            report['adapter_gate_fp32']={'status':'passed','threshold_total_variation':0.01}
        except AssertionError as error:
            report['adapter_gate_fp32']={'status':'failed','error':str(error),'threshold_total_variation':0.01}
        finally:
            for layer in model.layers: layer.attention.kv_cache=None
            model.causal_mask=None
        print('ADAPTER_GATE_FP32',json.dumps(report['adapter_gate_fp32']),flush=True)
        model.bfloat16()
        try:
            check_native_prefixes(model,cond[:1].bfloat16(),masks[:1])
            report['adapter_gate_bf16']={'status':'passed','threshold_total_variation':0.01}
        except AssertionError as error:
            report['adapter_gate_bf16']={'status':'failed','error':str(error),'threshold_total_variation':0.01}
        finally:
            for layer in model.layers: layer.attention.kv_cache=None
            model.causal_mask=None
        print('ADAPTER_GATE_BF16',json.dumps(report['adapter_gate_bf16']),flush=True)
    report['status']='diagnostic_completed'; atomic_json(target,report)
    print('LLAMAGEN_STRUCTURE_DIAGNOSTIC_COMPLETED',args.out,flush=True)


if __name__=='__main__': main()

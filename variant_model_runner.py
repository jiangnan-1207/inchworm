"""Separate three-method loader; original experiment loader is untouched.

Model loading/sampling adapted from Janus and LlamaGen (see licenses/).
Derived by tools/build_variant_loader.py; review changes before rebuilding.
"""
import time
import sys
from inchworm_core import Config, generate
from inchworm_variant import VariantConfig, generate as generate_variant
from inchworm_variant_backend import VariantBackend


def run_controller(backend,state,length,method,config):
    if method=='inchworm_variant':
        return generate_variant(VariantBackend(backend,config.temperature),state,length,config)
    if method=='inchworm':
        return generate(backend,state,length,Config())
    raise ValueError('Unknown controller')


def diagnostics(method,backtracks,result):
    if method=='backbone':
        return {'backtracks':0,'checks':0,'accepted_edits':0,'kept_original':0,'events':[]}
    return {'backtracks':backtracks,'checks':result.get('checks',backtracks),
        'accepted_edits':result.get('accepted_edits',backtracks),
        'kept_original':result.get('kept_original',0),'events':result['events']}


def load_model(name,root,data,variant_config=VariantConfig()):
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
            if method in ['inchworm','inchworm_variant']:
                result=run_controller(JanusBackend(model),State(prompt_inputs=inputs),576,method,variant_config)
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
            return image,token_seconds,diagnostics(method,backtracks,locals().get("result"))
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
        if method in ['inchworm','inchworm_variant']:
            backend=LlamaGenBackend(model,cond,masks)
            try:
                result=run_controller(backend,State(),1024,method,variant_config)
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
        return image,seconds,diagnostics(method,backtracks,locals().get("result"))
    return run

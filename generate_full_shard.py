"""One load per model, full GenEval support; pilot is a separate labelled run."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from protocol import (PROMPT_SHA256, METHODS, atomic_json, bind_run, commit_image,
                      completed, load_prompts, paths, seed_for, select_ids, sha, validate_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--scratch','--data-root',dest='scratch',required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scope', choices=['pilot', 'full'], required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--shard-index',type=int,required=True)
    parser.add_argument('--shards',type=int,default=8)
    args = parser.parse_args()
    root, scratch = Path(args.root), Path(args.scratch)
    rows = load_prompts(root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    if args.scope != 'full':
        raise ValueError('This production worker only accepts the full benchmark')
    if not 0 <= args.shard_index < args.shards <= 553:
        raise ValueError('Invalid shard specification')
    ids = select_ids(rows, args.scope)[args.shard_index::args.shards]
    args.model = 'llamagen' if args.method == 'llamagen_baseline' else 'janus'
    samples = 4 if args.scope == 'full' else 1
    sources = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    from inchworm_core import Config
    full_cfg = Config(max_backtracks=None)
    full_cfg.validate()
    specification = {
        'scope': 'full_shard', 'formal_complete_benchmark':False,
        'shard_index':args.shard_index, 'shards':args.shards, 'method':args.method,
        'prompt_sha256': PROMPT_SHA256, 'prompt_ids': ids, 'samples_per_prompt': samples,
        'methods': METHODS, 'source_sha256': sources,
        'seed_rule': '42 + 4*original_prompt_id + sample_id',
        'janus': {'resolution':384,'tokens':576,'cfg':5.0,'temperature':1.0,'dtype':'bf16',
                  'baseline_tensor_batch':4,'inchworm_tensor_batch':1},
        'inchworm': asdict(full_cfg),
        'inchworm_reproduction_status': 'engineering_configuration_v1; missing original paper settings',
        'llamagen': {'checkpoint':'t2i_XL_stage2_512.pt','resolution':512,
                     'tokens':1024,'cfg':7.5,'temperature':1.0,'top_k':1000,'top_p':1.0,'dtype':'bf16'},
        'asset_revisions': json.loads((root/'benchmarks/setup_records/model_revisions_v1.json').read_text()),
    }
    bind_run(args.out, specification)
    import torch
    import numpy as np
    from PIL import Image
    assert torch.cuda.is_available(), 'Generation requires a CUDA GPU'
    print('RUN_SCOPE:', args.scope, 'PROMPTS:',len(ids),'SAMPLES_PER_PROMPT:',samples, flush=True)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    print('INCHWORM: uncapped backtracking, provisional engineering configuration', flush=True)

    def seed(value):
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        np.random.seed(value)

    def save(method, i, j, image, diag):
        commit_image(args.out, method, i, j, rows[i], specification, image, diag)
        print(f'IMAGE_OK {method} prompt={i} sample={j} seconds={diag["generation_seconds"]:.2f}'
              f' backtracks={diag.get("backtracks",0)}',flush=True)

    def already(method, i, j):
        image, receipt = paths(args.out,method,i,j)
        yes = completed(image, receipt, specification, rows[i])
        if yes:
            print(f'SKIP_COMPLETE {method} {i}/{j}',flush=True)
        return yes

    load_start = time.perf_counter()
    if args.model == 'janus':
        from transformers import AutoModelForCausalLM
        from janus.models import VLChatProcessor, MultiModalityCausalLM
        from janus_backend import JanusBackend, State
        from inchworm_core import generate
        model_dir = str(scratch/'models/Janus-Pro-7B')
        processor = VLChatProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-load_start

        def initial(text):
            prompt = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=[{'role':'<|User|>','content':text},{'role':'<|Assistant|>','content':''}],
                sft_format=processor.sft_format,system_prompt='') + processor.image_start_tag
            encoded = processor.tokenizer.encode(prompt)
            tokens = torch.tensor([encoded,encoded],dtype=torch.long,device='cuda')
            tokens[1,1:-1] = processor.pad_id
            return model.language_model.get_input_embeddings()(tokens)

        # Same-model, same-seed short native-loop comparison before any outputs.
        # This checks the uncapped controller with backtracking disabled, and
        # does not claim cross-hardware identity of the earlier saved apple PNG.
        with torch.inference_mode():
            seed(123)
            inputs = initial(rows[ids[0]]['prompt'])
            reference = []
            cache = None
            for _ in range(32):
                output = model.language_model.model(inputs_embeds=inputs,past_key_values=cache,use_cache=True)
                cache = output.past_key_values
                logits = model.gen_head(output.last_hidden_state[:,-1,:])
                probs = torch.softmax(logits[1:2]+5.0*(logits[0:1]-logits[1:2]),dim=-1)
                z = torch.multinomial(probs,num_samples=1)
                reference.append(int(z.item()))
                inputs = model.prepare_gen_img_embeds(z[:,0].repeat_interleave(2)).unsqueeze(1)
            seed(123)
            actual = generate(JanusBackend(model),State(prompt_inputs=initial(rows[ids[0]]['prompt'])),
                              32,Config(max_backtracks=0))['tokens']
            assert reference == actual, 'Disabled-backtracking sampling differs from native loop'
            del cache, output, inputs, logits, probs
        print('JANUS_NATIVE_LOOP_PARITY_32_TOKENS_OK',flush=True)
        if args.method == 'janus_baseline':
            from native_janus_batch import run_baseline
            run_baseline(model, initial, args, rows, ids, samples, specification, load_seconds)
            print('MODEL_GENERATION_DONE',args.method,flush=True)
            return
        for method in [args.method]:
            cfg = Config(max_backtracks=0) if method == 'janus_baseline' else full_cfg
            for i in ids:
                for j in range(samples):
                    if already(method,i,j):
                        continue
                    seed(seed_for(i,j))
                    with torch.inference_mode():
                        backend = JanusBackend(model)
                        state = State(prompt_inputs=initial(rows[i]['prompt']))
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = generate(backend,state,576,cfg)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter()-start
                        decode_start = time.perf_counter()
                        code = torch.tensor([result['tokens']],dtype=torch.int,device='cuda')
                        decoded = model.gen_vision_model.decode_code(code,shape=[1,8,24,24])
                        pixels = decoded.float().cpu().numpy().transpose(0,2,3,1)
                        image = Image.fromarray(np.clip((pixels[0]+1)*127.5,0,255).astype(np.uint8))
                        torch.cuda.synchronize()
                        decode_seconds = time.perf_counter()-decode_start
                    save(method,i,j,image,{'seed':seed_for(i,j),'generation_seconds':seconds,
                        'decode_seconds':decode_seconds,'load_seconds_shared':load_seconds,
                        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                        'backtracks':result['backtracks'],'counters':backend.counters(),
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'tokens':result['tokens'],'events':result['events']})
                    del backend,state,result,code,decoded,pixels,image
            validate_outputs(args.out,method,rows,ids,samples)
            print('GENERATION_METHOD_OK',method,flush=True)
    else:
        sys.path.insert(0,str(root/'benchmarks/LlamaGen'))
        from tokenizer.tokenizer_image.vq_model import VQ_models
        from autoregressive.models.gpt import GPT_models
        from autoregressive.models.generate import generate as llama_generate
        from language.t5 import T5Embedder
        # Match official sample_t2i.py precision and padding behavior.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        directory = scratch/'models/LlamaGen'
        vq = VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).cuda().eval()
        checkpoint = torch.load(directory/'vq_ds16_t2i.pt',map_location='cpu',weights_only=False)
        vq.load_state_dict(checkpoint['model'],strict=True)
        del checkpoint
        gpt = GPT_models['GPT-XL'](block_size=1024,cls_token_num=120,model_type='t2i').to(device='cuda',dtype=torch.bfloat16).eval()
        checkpoint = torch.load(directory/'t2i_XL_stage2_512.pt',map_location='cpu',weights_only=False)
        key = next((key for key in ['model','module','state_dict'] if key in checkpoint),None)
        if key is None:
            raise ValueError('Unknown LlamaGen checkpoint schema')
        gpt.load_state_dict(checkpoint[key],strict=True)
        del checkpoint
        encoder = T5Embedder(device='cuda',local_cache=True,cache_dir=str(directory/'t5-ckpt'),
                            dir_or_name='flan-t5-xl',torch_dtype=torch.bfloat16,model_max_length=120)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-load_start
        method = 'llamagen_baseline'
        for i in ids:
            with torch.inference_mode():
                text_start = time.perf_counter()
                emb, masks = encoder.get_text_embeddings([rows[i]['prompt']])
                valid = int(masks[0].sum().item())
                emb = torch.cat([emb[:,valid:,:],emb[:,:valid,:]],dim=1)
                masks = torch.flip(masks,dims=[-1])
                cond = emb*masks[:,:,None]
                torch.cuda.synchronize()
                text_seconds = time.perf_counter()-text_start
            for j in range(samples):
                if already(method,i,j):
                    continue
                seed(seed_for(i,j))
                with torch.inference_mode():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    tokens = llama_generate(gpt,cond,1024,masks,cfg_scale=7.5,
                                           temperature=1.0,top_k=1000,top_p=1.0,sample_logits=True)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-start
                    decode_start = time.perf_counter()
                    decoded = vq.decode_code(tokens,[1,8,32,32])
                    pixels = decoded.float().cpu().numpy().transpose(0,2,3,1)[0]
                    image = Image.fromarray(np.clip((pixels+1)*127.5+0.5,0,255).astype(np.uint8))
                    torch.cuda.synchronize()
                    decode_seconds = time.perf_counter()-decode_start
                save(method,i,j,image,{'seed':seed_for(i,j),'generation_seconds':seconds,
                    'decode_seconds':decode_seconds,'text_encoding_seconds_per_prompt':text_seconds,
                    'load_seconds_shared':load_seconds,'tokens':tokens[0].tolist(),
                    'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                del tokens,decoded,pixels,image
        validate_outputs(args.out,method,rows,ids,samples)
        print('GENERATION_METHOD_OK',method,flush=True)
    print('MODEL_GENERATION_DONE',args.model,flush=True)


if __name__ == '__main__':
    main()

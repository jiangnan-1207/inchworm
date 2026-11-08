"""Native Janus sampling, four images per prompt in a single tensor batch."""
import time
from protocol import commit_image, completed, paths, seed_for, validate_outputs


def run_baseline(model, initial, args, rows, ids, samples, specification, load_seconds):
    import torch
    import numpy as np
    from PIL import Image
    assert samples==4
    method='janus_baseline'
    for i in ids:
        done=[completed(*paths(args.out,method,i,j),specification,rows[i]) for j in range(samples)]
        if all(done):
            print('SKIP_COMPLETE_PROMPT',i,flush=True)
            continue
        # Always sample a full group of four, including during partial resume,
        # so kernel batch shape and per-image random generators stay fixed.
        generators=[torch.Generator(device='cuda').manual_seed(seed_for(i,j)) for j in range(samples)]
        with torch.inference_mode():
            inputs=initial(rows[i]['prompt']).repeat(samples,1,1)
            codes=torch.empty((samples,576),dtype=torch.int,device='cuda')
            cache=None
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            for t in range(576):
                output=model.language_model.model(inputs_embeds=inputs,past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                logits=model.gen_head(output.last_hidden_state[:,-1,:])
                probabilities=torch.softmax(logits[1::2]+5.0*(logits[0::2]-logits[1::2]),dim=-1)
                tokens=torch.cat([torch.multinomial(probabilities[j:j+1],1,generator=generators[j])
                                  for j in range(samples)],dim=0)
                codes[:,t]=tokens[:,0]
                inputs=model.prepare_gen_img_embeds(tokens[:,0].repeat_interleave(2)).unsqueeze(1)
            torch.cuda.synchronize()
            batch_seconds=time.perf_counter()-start
            decode_start=time.perf_counter()
            decoded=model.gen_vision_model.decode_code(codes,shape=[samples,8,24,24])
            pixels=decoded.float().cpu().numpy().transpose(0,2,3,1)
            torch.cuda.synchronize()
            decode_seconds=time.perf_counter()-decode_start
        for j in range(samples):
            if done[j]:
                continue
            image=Image.fromarray(np.clip((pixels[j]+1)*127.5,0,255).astype(np.uint8))
            commit_image(args.out,method,i,j,rows[i],specification,image,{
                'seed':seed_for(i,j),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                'generation_seconds':batch_seconds/samples,
                'generation_seconds_definition':'amortized tensor-batch time per image',
                'batch_generation_seconds':batch_seconds,'tensor_batch_size':samples,
                'decode_seconds':decode_seconds/samples,'load_seconds_shared':load_seconds,
                'backtracks':0,'tokens':codes[j].tolist(),
                'forward_calls_per_batch':576,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        print(f'FULL_BASELINE_PROMPT_OK id={i} images=4 batch_seconds={batch_seconds:.2f}',flush=True)
        del cache,output,inputs,codes,decoded,pixels,probabilities,tokens
    validate_outputs(args.out,method,rows,ids,samples)
    print('GENERATION_METHOD_OK',method,flush=True)

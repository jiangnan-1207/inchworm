"""Complete 553x4 GenEval generation for an explicit original/variant version."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from four_task_protocol import full_shard_ids, variant_method
from ablation_plan import BENCHMARK_VERSIONS, ablation_flags, benchmark_config, controller_method
from protocol import (PROMPT_SHA256,atomic_json,bind_run,commit_image,completed,
    load_prompts,paths,seed_for,sha,validate_outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen'],required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--variant',choices=BENCHMARK_VERSIONS,required=True)
    a=p.parse_args()
    method=variant_method(a.model,a.variant)
    ids=full_shard_ids(a.shard_index,a.shards)
    rows=load_prompts(a.root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    config=benchmark_config(a.variant)
    generation_method=controller_method(a.variant)
    spec={'scope':'full_shard','formal_complete_benchmark':False,'method':method,'model':a.model,'variant':a.variant,
        'shards':a.shards,'shard_index':a.shard_index,'prompt_ids':ids,'prompt_sha256':PROMPT_SHA256,
        'samples_per_prompt':4,'seed_rule':'42 + 4*original_prompt_id + sample_id',
        'variant_config':asdict(config),'controller_method':generation_method,
        'ablation_flags':ablation_flags(a.variant),'batch_size':1,'dtype':'bf16',
        'resolution':384 if a.model=='janus1b' else 512,
        'cfg':5.0 if a.model=='janus1b' else 7.5,'base_temperature':1.0,
        'image_tokens':576 if a.model=='janus1b' else 1024,
        'top_k':None if a.model=='janus1b' else 1000,
        'source_sha256':{f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'asset_revisions':json.loads((a.root/'benchmarks/setup_records/model_revisions_v1.json').read_text()),
        'reproduction_status':'engineering_configuration; not exact paper reproduction'}
    out=a.run/'shards'/f'{method}_{a.shard_index}'
    bind_run(out,spec)
    import torch
    import numpy as np
    from variant_model_runner import load_model
    if not torch.cuda.is_available(): raise RuntimeError('Run on an allocated CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    with torch.inference_mode():
        run=load_model(a.model,a.root,a.data_root) if a.variant in ['original','backbone'] else load_model(a.model,a.root,a.data_root,config)
        image,_,_=run(rows[ids[0]]['prompt'],generation_method); del image
        for i in ids:
            for j in range(4):
                image_path,receipt=paths(out,method,i,j)
                if completed(image_path,receipt,spec,rows[i]):
                    print('SKIP_COMPLETE',method,i,j,flush=True); continue
                seed=seed_for(i,j)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                image,token_seconds,diag=run(rows[i]['prompt'],generation_method)
                if a.variant=='backbone' and any(diag[k] for k in ['backtracks','checks','accepted_edits','kept_original']):
                    raise ValueError('Pure backbone run unexpectedly used a backtracking controller')
                torch.cuda.synchronize(); seconds=time.perf_counter()-start
                commit_image(out,method,i,j,rows[i],spec,image,{
                    'seed':seed,'generation_seconds':token_seconds,'latency_seconds':seconds,
                    'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**diag})
                del image
                print('VARIANT_FULL_IMAGE_OK',method,i,j,flush=True)
        count=validate_outputs(out,method,rows,ids,4)
    atomic_json(out/'completed.json',{'status':'completed_shard','images':count,'prompts':len(ids)})
    print('VARIANT_FULL_SHARD_OK',method,a.shard_index,count,flush=True)


if __name__=='__main__': main()

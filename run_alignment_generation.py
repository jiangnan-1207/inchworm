"""Both backbones and ORIGINAL Inchworm on complete DPG and TIIF testmini."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from alignment_protocol import build_manifest,check_sample,sample_paths
from inchworm_core import Config
from protocol import atomic_json,bind_run,sha


def generation_spec(root,model):
    manifest=json.loads((root/'benchmarks/setup_records/alignment_manifest.json').read_text())
    actual=build_manifest(root)
    if actual['prompt_sha256']!=manifest['prompt_sha256'] or actual['source_sha256']!=manifest['source_sha256']:
        raise ValueError('Dataset changed after preparation')
    return manifest,dict(model=model,manifest_sha256=sha(root/'benchmarks/setup_records/alignment_manifest.json'),
        config=asdict(Config()),methods=['backbone','inchworm'],variant=False,
        seed='20261109 + 4*global_manifest_row_id + sample_id; identical across methods',
        resolution=384 if model=='janus1b' else 512,cfg=5 if model=='janus1b' else 7.5,
        temperature=1,top_k=None if model=='janus1b' else 1000,dtype='bfloat16',
        source_sha256={p.name:sha(p) for p in sorted(root.glob('*.py'))},
        model_revisions=json.loads((root/'configs/model_revisions.json').read_text()),
        reproduction_status='engineering original Inchworm implementation; not exact paper reproduction')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','data-root','run']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen'],required=True)
    p.add_argument('--shards',type=int,default=16); p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    if not 0<=a.index<a.shards: raise ValueError('Invalid shard')
    manifest,spec=generation_spec(a.root,a.model)
    shard=a.run/'workers'/a.model/str(a.index)
    bind_run(shard,dict(spec=spec,index=a.index,shards=a.shards))
    import numpy as np
    import torch
    from variant_model_runner import load_model
    if not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA GPU required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    count=0
    with torch.inference_mode():
        run=load_model(a.model,a.root,a.data_root)
        for i,row in enumerate(manifest['rows']):
            if i%a.shards!=a.index: continue
            for j in range(row['samples']):
                for method in ['backbone','inchworm']:
                    if check_sample(a.run,a.model,method,row,j,spec): continue
                    seed=20261109+4*i+j
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                    image,token_time,diag=run(row['prompt'],method)
                    torch.cuda.synchronize()
                    path,receipt=sample_paths(a.run,a.model,method,row,j); path.parent.mkdir(parents=True,exist_ok=True)
                    tmp=path.with_suffix('.png.part'); image.save(tmp,format='PNG'); tmp.replace(path)
                    atomic_json(receipt,dict(spec=spec,row=row,image_sha256=sha(path),seed=seed,
                        latency_seconds=time.perf_counter()-start,generation_seconds=token_time,
                        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),**diag))
                    del image; count+=1
                    print('ALIGNMENT_IMAGE_OK',a.model,method,row['benchmark'],row['key'],j,flush=True)
    atomic_json(shard/'completed.json',dict(status='completed',new_images=count))
    print('ALIGNMENT_SHARD_OK',flush=True)


if __name__=='__main__': main()

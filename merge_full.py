"""Verify disjoint full coverage before linking worker outputs for evaluation."""
import argparse
import json
from pathlib import Path
from protocol import METHODS, atomic_json, load_prompts, validate_outputs


def merge(root,run,method,shards,destination=None):
    rows=load_prompts(Path(root)/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    run=Path(run)
    seen=set()
    shared=None
    sources=[]
    for s in range(shards):
        directory=run/'shards'/f'{method}_{s}'
        spec=json.loads((directory/'run_spec.json').read_text())
        ids=list(range(s,553,shards))
        if spec['method']!=method or spec['shards']!=shards or spec['shard_index']!=s or spec['prompt_ids']!=ids:
            raise ValueError('Wrong shard specification')
        common={k:v for k,v in spec.items() if k not in ['shard_index','prompt_ids']}
        if shared is None:
            shared=common
        if shared!=common:
            raise ValueError('Worker sources, assets or settings differ')
        validate_outputs(directory,method,rows,ids,4)
        if seen.intersection(ids):
            raise ValueError('Overlapping prompt IDs')
        seen.update(ids)
        sources.extend((i,directory/method/f'{i:05d}') for i in ids)
    if seen!=set(range(553)):
        raise ValueError('All 553 prompts are required')
    merged_root=Path(destination) if destination is not None else run/'merged'
    merged=merged_root/method
    merged.mkdir(parents=True,exist_ok=True)
    for i,source in sources:
        target=merged/f'{i:05d}'
        if target.is_symlink():
            if target.resolve()!=source.resolve():
                raise ValueError('Unexpected existing symlink')
        elif target.exists():
            raise ValueError('Refusing to replace an existing directory')
        else:
            target.symlink_to(source.resolve(),target_is_directory=True)
    count=validate_outputs(merged_root,method,rows,list(range(553)),4)
    atomic_json(merged_root/f'{method}.coverage.json',{'prompts':553,'images':count,'specification':shared})
    print('FULL_COVERAGE_OK',method,553,count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--shards',type=int,required=True)
    p.add_argument('--destination',type=Path)
    a=p.parse_args()
    merge(a.root,a.run,a.method,a.shards,a.destination)

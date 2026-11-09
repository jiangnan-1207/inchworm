"""Read-only validation of all thirty pairs, independent of Slurm status."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import sha


def validate(run):
    run=Path(run); seen=set(); total_backtracks=0
    records=sorted((run/'workers').glob('*/run_spec.json'))
    if not records: raise ValueError('No worker manifests')
    shared=None
    for manifest in records:
        spec=json.loads(manifest.read_text())
        ids=spec['prompt_ids']; index=spec['shard_index']; shards=spec['shards']
        if not 0<=index<shards<=30 or ids!=list(range(index,30,shards)) or seen.intersection(ids):
            raise ValueError('Wrong or overlapping prompt partition')
        common={k:v for k,v in spec.items() if k not in ['prompt_ids','shard_index']}
        if shared is not None and common!=shared: raise ValueError('Worker sources/configurations differ')
        shared=common
        parity=json.loads((manifest.parent/'parity.json').read_text())
        if (not parity['includes_newline_boundary'] or parity['maximum_total_variation']>parity['threshold']
            or parity['restored_total_variation']>parity['threshold']): raise ValueError('Failed parity gate')
        if not (manifest.parent/'completed.json').exists(): raise ValueError('Incomplete worker')
        seen.update(ids)
        for i in ids:
            for method in ['backbone','inchworm']:
                png=run/method/f'{i:04d}.png'
                record=json.loads(png.with_suffix('.json').read_text())
                if record['specification']!=spec or record['image_sha256']!=sha(png):
                    raise ValueError(f'Missing or changed sample: {png}')
                if method=='inchworm': total_backtracks+=record['backtracks']
    if seen!=set(range(30)): raise ValueError('All thirty prompts are required')
    for method in ['backbone','inchworm']:
        if {p.name for p in (run/method).glob('*.png')}!={f'{i:04d}.png' for i in range(30)}:
            raise ValueError('Missing or extra images')
    return {'pairs':30,'images':60,'original_inchworm_total_backtracks':total_backtracks,
            'original_inchworm_mean_backtracks':total_backtracks/30,'status':'complete'}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(validate(a.run),indent=2))


if __name__=='__main__': main()

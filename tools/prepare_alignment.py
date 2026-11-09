"""CPU job: pinned official datasets; existing generators are reused unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alignment_protocol import build_manifest
from protocol import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args(); pins=json.loads((ROOT/'configs/alignment_sources.json').read_text())
    env=os.environ.copy()
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']: env.pop(k,None)
    for name in ['ELLA','TIIF-Bench','LlamaGen']:
        url,rev=pins[name]; dest=ROOT/'benchmarks'/name
        if not (dest/'.git').is_dir():
            if dest.exists() and any(dest.iterdir()): raise ValueError(f'Refusing to replace {dest}')
            dest.mkdir(parents=True,exist_ok=True)
            for cmd in [['git','init',dest],['git','-C',dest,'remote','add','origin',url],
                        ['git','-C',dest,'fetch','--depth','1','origin',rev],['git','-C',dest,'checkout','--detach','FETCH_HEAD']]:
                subprocess.run(list(map(str,cmd)),env=env,check=True)
        if subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()!=rev:
            raise ValueError(f'Wrong source revision: {name}')
        subprocess.run(['git','-C',str(dest),'diff','--exit-code','HEAD'],check=True)
    manifest=build_manifest(ROOT); manifest['pins']=pins
    target=ROOT/'benchmarks/setup_records/alignment_manifest.json'
    atomic_json(target,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['rows','source_sha256']},indent=2),flush=True)
    for f in ['models/Janus-Pro-1B/config.json','models/LlamaGen/t2i_XL_stage2_512.pt','models/LlamaGen/vq_ds16_t2i.pt']:
        if not (a.data_root/f).is_file(): raise ValueError(f'Existing generation asset missing: {f}')
    print('ALIGNMENT_DATA_READY',flush=True)


if __name__=='__main__': main()

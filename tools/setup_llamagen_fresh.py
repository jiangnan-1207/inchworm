"""Create an independent upstream LlamaGen environment without editing old assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_REVISION='ce98ec41803a74a90ce68c40ababa9eaeffeb4ec'
WEIGHT_REVISION='276f5c5a3d915b922899a03f1912605531574747'


def run(command,env):
    print('+',' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),env=env,check=True)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--assets-only',action='store_true')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args(); fresh=a.fresh_root.resolve(); data=a.data_root.resolve()
    if a.plan:
        print(json.dumps({'fresh_root':str(fresh),'source':SOURCE_REVISION,'torch':'2.1.2+cu121',
            'transformers':'4.36.2','stages':[256,512],'old_assets':'read only, verified files reused',
            'requirements':str(ROOT/'requirements/llamagen-fresh.txt')})); return
    if sys.platform!='linux' or sys.version_info[:2]!=(3,10): raise RuntimeError('Use Linux Python 3.10')
    if data==fresh or data in fresh.parents or ROOT==fresh or ROOT in fresh.parents:
        raise ValueError('Use an independent directory outside DATA_ROOT and the code checkout')
    fresh.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PIP_CONSTRAINT','PIP_EXTRA_INDEX_URL']:
        env.pop(key,None)
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://pypi.org/simple',PIP_NO_CACHE_DIR='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(fresh/'cache/huggingface'),CUDA_VISIBLE_DEVICES='')
    source=fresh/'source'; python=fresh/'env/bin/python'
    if not a.assets_only:
        if (fresh/'env').exists() or source.exists(): raise ValueError('Use a new fresh root; do not overwrite a partial deployment')
        run([sys.executable,'-m','venv',fresh/'env'],env)
        run([python,'-m','pip','install','pip==24.0','setuptools==69.5.1','wheel==0.43.0'],env)
        run([python,'-m','pip','install','torch==2.1.2','torchvision==0.16.2',
            '--index-url','https://download.pytorch.org/whl/cu121'],env)
        run([python,'-m','pip','install','-r',ROOT/'requirements/llamagen-fresh.txt'],env)
        run([python,'-m','pip','check'],env)
        run(['git','init',source],env)
        run(['git','-C',source,'remote','add','origin','https://github.com/FoundationVision/LlamaGen.git'],env)
        run(['git','-C',source,'fetch','--depth','1','origin',SOURCE_REVISION],env)
        run(['git','-C',source,'checkout','--detach','FETCH_HEAD'],env)
        with (fresh/'environment.freeze.txt').open('w') as output:
            subprocess.run([str(python),'-m','pip','freeze'],env=env,stdout=output,check=True)
        run([python,Path(__file__),'--fresh-root',fresh,'--data-root',data,'--assets-only'],env)
        return
    # Previously verified large files are read, not redownloaded or modified.
    manifest=json.loads((ROOT/'configs/llama_asset_fingerprints.json').read_text())
    assets=fresh/'models'; assets.mkdir(exist_ok=True)
    records={}
    from tools.audit_llamagen_assets import audit_assets
    verified=audit_assets(data/'models/LlamaGen',manifest)
    if verified['status']!='assets_match_pinned_sources': raise ValueError('Existing assets failed verification')
    for record in verified['files']:
        old=Path(record['path']); relative=old.relative_to(data/'models/LlamaGen')
        target=assets/relative; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.symlink_to(old)
        elif target.resolve()!=old.resolve(): raise ValueError('Unexpected existing asset link')
        records[str(relative)]=record['actual_digest'] if record['algorithm']=='sha256' else digest(old)
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download('peizesun/llamagen_t2i','t2i_XL_stage1_256.pt',
        revision=WEIGHT_REVISION,local_dir=str(assets),local_dir_use_symlinks=False))
    records[path.name]=digest(path)
    (fresh/'setup_complete.json').write_text(json.dumps({'status':'ready_for_native_generation',
        'source_revision':SOURCE_REVISION,'weight_revision':WEIGHT_REVISION,'asset_sha256':records,
        'torch':'2.1.2+cu121','transformers':'4.36.2','scope':'fresh deployment, quality not yet established'},indent=2)+'\n')
    print('LLAMAGEN_FRESH_SETUP_OK',fresh,flush=True)


if __name__=='__main__': main()

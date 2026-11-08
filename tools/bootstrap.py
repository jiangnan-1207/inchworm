"""Create isolated Linux environments and fetch pinned public model sources.

Run with Python 3.10 on an allocated CPU/compute node. No GPU is requested.
The --plan option is standard-library-only and makes no changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    'Janus': ('https://github.com/deepseek-ai/Janus.git', '1daa72fa409002d40931bd7b36a9280362469ead'),
    'geneval': ('https://github.com/djghosh13/geneval.git', 'af4902f24d3ca90ebbb446dd9891a59e0f82725f'),
    'LlamaGen': ('https://github.com/FoundationVision/LlamaGen.git', 'ce98ec41803a74a90ce68c40ababa9eaeffeb4ec'),
    'mmdetection': ('https://github.com/open-mmlab/mmdetection.git', 'e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1'),
}


def run(command, env=None, stdout=None):
    print('+', ' '.join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)),check=True,env=env,stdout=stdout)


def ensure_repository(name, base_env):
    url, revision = REPOSITORIES[name]
    target = ROOT/'benchmarks'/name
    if (target/'.git').is_dir():
        remote = subprocess.check_output(['git','-C',str(target),'remote','get-url','origin'],text=True).strip()
        head = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if remote != url or head != revision:
            raise RuntimeError(f'{name}: existing source differs; inspect it instead of overwriting')
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f'Refusing to replace {target}')
    target.mkdir(parents=True,exist_ok=True)
    run(['git','init',target],base_env)
    run(['git','-C',target,'remote','add','origin',url],base_env)
    run(['git','-C',target,'fetch','--depth','1','origin',revision],base_env)
    run(['git','-C',target,'checkout','--detach','FETCH_HEAD'],base_env)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--components',nargs='+',choices=['janus','llamagen','geneval'],default=['janus','geneval','llamagen'])
    p.add_argument('--skip-downloads',action='store_true')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    data=a.data_root.expanduser().resolve()
    if data==ROOT or ROOT in data.parents:
        raise ValueError('Put model/env/output data outside the source repository')
    if a.plan:
        print(json.dumps({'repo':str(ROOT),'data_root':str(data),'components':a.components,
                          'sources':REPOSITORIES,'downloads':not a.skip_downloads},indent=2))
        return
    if sys.platform!='linux' or sys.version_info[:2]!=(3,10):
        raise RuntimeError('Use Python 3.10 on Linux for this pinned CUDA environment recipe')
    data.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['PIP_EXTRA_INDEX_URL','PIP_CONSTRAINT','PYTHONPATH','PYTHONHOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:
        env.pop(key,None)
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://pypi.org/simple',
               PIP_NO_CACHE_DIR='1',PIP_RETRIES='2',PIP_TIMEOUT='60',GIT_TERMINAL_PROMPT='0',
               PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(data/'hf_cache'),
               TORCH_HOME=str(data/'torch_cache'),XDG_CACHE_HOME=str(data/'.cache'),
               TMPDIR=str(data/'tmp'),OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    (data/'tmp').mkdir(exist_ok=True)
    (ROOT/'benchmarks/setup_records').mkdir(parents=True,exist_ok=True)
    for name in REPOSITORIES:
        ensure_repository(name,env)
    for component in dict.fromkeys(a.components):
        print('INSTALL_COMPONENT',component,flush=True)
        prefix=data/'envs'/component
        python=prefix/'bin/python'
        if not python.exists():
            if prefix.exists():
                raise RuntimeError(f'Partial environment at {prefix}; inspect before retrying')
            run([sys.executable,'-m','venv',prefix],env)
        scoped=env.copy()
        run([python,'-m','pip','install','pip==24.3.1','setuptools==69.5.1','wheel==0.45.1'],scoped)
        if component=='geneval':
            scoped['PIP_CONSTRAINT']=str(ROOT/'requirements/geneval-constraints.txt')
            run([python,'-m','pip','install','torch==2.0.0+cu118','torchvision==0.15.1+cu118',
                 '--index-url','https://download.pytorch.org/whl/cu118'],scoped)
            run([python,'-m','pip','install','--only-binary=:all:','mmcv-full==1.7.2',
                 '--find-links','https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html'],scoped)
        else:
            run([python,'-m','pip','install','torch==2.5.1+cu124','torchvision==0.20.1+cu124',
                 '--index-url','https://download.pytorch.org/whl/cu124'],scoped)
        run([python,'-m','pip','install','-r',ROOT/f'requirements/{component}.txt'],scoped)
        if component in ['janus','geneval']:
            source=ROOT/'benchmarks'/('Janus' if component=='janus' else 'mmdetection')
            run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',source],scoped)
        run([python,'-m','pip','check'],scoped)
        with (ROOT/f'benchmarks/setup_records/{component}.freeze.txt').open('w') as out:
            run([python,'-m','pip','freeze'],scoped,out)
        if not a.skip_downloads:
            run([python,ROOT/'tools/download_assets.py','--component',component,'--data-root',data],scoped)
        print('COMPONENT_SETUP_OK',component,flush=True)
    print('BOOTSTRAP_COMPLETE: GPU generation/evaluation still require actual execution',flush=True)


if __name__=='__main__':
    main()

"""CPU-node-only isolated Lumina install/download; no existing environment edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(args,env):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--assets-only',action='store_true')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args(); data=a.data_root.expanduser().resolve()
    pin=json.loads((ROOT/'configs/lumina_sources.json').read_text())
    if data==ROOT or ROOT in data.parents: raise ValueError('Data must be outside checkout')
    if a.plan:
        print(json.dumps({'data_root':str(data),'environment':str(data/'envs/lumina-mgpt2'),'pins':pin},indent=2)); return
    env=os.environ.copy()
    env.update(HF_HOME=str(data/'hf_cache'),TORCH_HOME=str(data/'torch_cache'),TMPDIR=str(data/'tmp'),
        XDG_CACHE_HOME=str(data/'.cache'),PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://pypi.org/simple',
        PIP_NO_CACHE_DIR='1',CUDA_VISIBLE_DEVICES='',HF_HUB_DOWNLOAD_TIMEOUT='120')
    for key in ['PYTHONPATH','PYTHONHOME','PIP_EXTRA_INDEX_URL','PIP_CONSTRAINT','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:
        env.pop(key,None)
    (data/'tmp').mkdir(parents=True,exist_ok=True)
    source=ROOT/'benchmarks/Lumina-mGPT-2.0'
    if not (source/'.git').exists():
        if source.exists() and any(source.iterdir()): raise ValueError('Existing source is not a git checkout')
        source.mkdir(parents=True,exist_ok=True)
        run(['git','init',source],env)
        run(['git','-C',source,'remote','add','origin',pin['source_url']],env)
        run(['git','-C',source,'fetch','--depth','1','origin',pin['source_revision']],env)
        run(['git','-C',source,'checkout','--detach','FETCH_HEAD'],env)
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['source_revision']: raise ValueError('Lumina source revision differs')
    python=data/'envs/lumina-mgpt2/bin/python'
    if not a.assets_only:
        if sys.platform!='linux' or sys.version_info[:2]!=(3,10): raise ValueError('Use Linux Python 3.10')
        if not python.exists():
            if python.parent.parent.exists(): raise ValueError('Inspect incomplete Lumina environment')
            run([sys.executable,'-m','venv',python.parent.parent],env)
        run([python,'-m','pip','install','pip==24.3.1','setuptools==69.5.1','wheel==0.45.1'],env)
        run([python,'-m','pip','install','torch==2.3.1+cu121','torchvision==0.18.1+cu121',
             '--index-url','https://download.pytorch.org/whl/cu121'],env)
        run([python,'-m','pip','install','-r',ROOT/'requirements/lumina.txt'],env)
        run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',source],env)
        run([python,'-m','pip','check'],env)
        run([python,Path(__file__),'--data-root',data,'--assets-only'],env)
        return
    from huggingface_hub import snapshot_download,hf_hub_download
    target=data/'models/Lumina-mGPT-2.0'
    snapshot_download(pin['model_repo'],revision=pin['model_revision'],local_dir=str(target),max_workers=2,
        allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
    index=json.loads((target/'model.safetensors.index.json').read_text())
    for filename in set(index['weight_map'].values()):
        if not (target/filename).is_file(): raise ValueError(f'Missing shard {filename}')
    vq=Path(hf_hub_download(pin['vq_repo'],'movqgan_270M.ckpt',revision=pin['vq_revision'],local_dir=str(data/'models/MoVQGAN')))
    link=source/'lumina_mgpt/movqgan/270M/movqgan_270M.ckpt'; link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink():
        if link.resolve()!=vq.resolve(): raise ValueError('Wrong MoVQ link')
    elif link.exists(): raise ValueError('Refusing to replace existing MoVQ checkpoint')
    else: link.symlink_to(vq.resolve())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(target),local_files_only=True)
    assert len(tokenizer)>CODE_START_FOR_CHECK
    sys.path.insert(0,str(source)); sys.path.insert(0,str(source/'lumina_mgpt'))
    from inference_solver import FlexARInferenceSolver
    import torch
    weights=torch.load(vq,map_location='cpu'); del weights
    record=ROOT/'benchmarks/setup_records/lumina_assets.json'; record.parent.mkdir(parents=True,exist_ok=True)
    record.write_text(json.dumps({'pins':pin,'torch':torch.__version__,'gpu_validated':False},indent=2)+'\n')
    with (record.parent/'lumina.freeze.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],check=True,stdout=f)
    print('LUMINA_SETUP_OK',flush=True)


CODE_START_FOR_CHECK=155000
if __name__=='__main__': main()

"""Isolated CPU installation of mPLUG and Qwen/PaddleOCR scorers, no generator changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--assets',choices=['dpg','tiif']); a=p.parse_args(); data=a.data_root.resolve()
    if data==ROOT or ROOT in data.parents: raise ValueError('Data outside checkout required')
    env=os.environ.copy()
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH','PYTHONHOME','PIP_CONSTRAINT','PIP_EXTRA_INDEX_URL']: env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='',HF_HOME=str(data/'hf_cache'),MODELSCOPE_CACHE=str(data/'modelscope_cache'),
        PADDLEOCR_HOME=str(data/'paddleocr'),PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL='https://pypi.org/simple',PIP_NO_CACHE_DIR='1')
    def run(cmd): subprocess.run(list(map(str,cmd)),env=env,check=True)
    pins=json.loads((ROOT/'configs/alignment_sources.json').read_text())
    if not a.assets:
        if sys.platform!='linux' or sys.version_info[:2]!=(3,10): raise ValueError('Linux Python 3.10 required')
        for kind in ['dpg','tiif']:
            folder=data/f'envs/alignment-{kind}-v1'; python=folder/'bin/python'
            if not python.exists():
                if folder.exists(): raise ValueError(f'Inspect partial environment {folder}')
                run([sys.executable,'-m','venv',folder])
            run([python,'-m','pip','install','pip==24.0','setuptools==69.5.1','wheel==0.45.1'])
            if kind=='dpg':
                run([python,'-m','pip','install','torch==2.0.1+cu118','torchvision==0.15.2+cu118','torchaudio==2.0.2+cu118',
                    '--index-url','https://download.pytorch.org/whl/cu118'])
            else:
                run([python,'-m','pip','install','torch==2.5.1+cu124','torchvision==0.20.1+cu124',
                    '--index-url','https://download.pytorch.org/whl/cu124'])
            run([python,'-m','pip','install','-r',ROOT/f'requirements/alignment-{kind}.txt'])
            run([python,'-m','pip','check'])
            run([python,Path(__file__),'--data-root',data,'--assets',kind])
        print('ALIGNMENT_JUDGES_READY',flush=True); return
    kind=a.assets
    if kind=='dpg':
        from modelscope import snapshot_download
        path=Path(snapshot_download('damo/mplug_visual-question-answering_coco_large_en',revision='v1.0.0',
            cache_dir=str(data/'modelscope_cache')))
        from mplug_direct import MPlugDirect
        from PIL import Image
        judge=MPlugDirect(path,device='cpu')
        probe=judge({'image':Image.new('RGB',(384,384)), 'question':'Is there a cat?'})
        if not isinstance(probe.get('text'),str) or not probe['text'].strip(): raise ValueError('mPLUG CPU probe failed')
        print('MPLUG_DIRECT_CPU_INFERENCE_OK',flush=True)
        assets=dict(model_path=str(path),repo='damo/mplug_visual-question-answering_coco_large_en',revision='v1.0.0',
            runtime='direct ModelScope mPLUG; official resize/normalize/tokenizer/top-beam decode; no OFA/fairseq')
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        path=Path(snapshot_download(pins['qwen_repo'],revision=pins['qwen_revision'],local_dir=str(data/'models/TIIF-Qwen2.5-VL-7B'),
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=2))
        AutoProcessor.from_pretrained(path,local_files_only=True)
        from paddleocr import PaddleOCR
        ocr=data/'models/TIIF-PaddleOCR'
        PaddleOCR(lang='en',use_gpu=False,show_log=False,use_angle_cls=False,
            det_model_dir=str(ocr/'det'),rec_model_dir=str(ocr/'rec'),cls_model_dir=str(ocr/'cls'))
        assets=dict(model_path=str(path),repo=pins['qwen_repo'],revision=pins['qwen_revision'],
            ocr_path=str(ocr),ocr_sha256={str(f.relative_to(ocr)):sha(f) for f in sorted(ocr.rglob('*')) if f.is_file()})
    import torch
    assets['torch']=torch.__version__
    target=ROOT/f'benchmarks/setup_records/alignment_{kind}_judge.json'; atomic_json(target,assets)
    with target.with_suffix('.freeze.txt').open('w') as out:
        subprocess.run([sys.executable,'-m','pip','freeze'],check=True,stdout=out)
    print(f'ALIGNMENT_{kind.upper()}_JUDGE_ASSETS_OK',flush=True)


if __name__=='__main__': main()

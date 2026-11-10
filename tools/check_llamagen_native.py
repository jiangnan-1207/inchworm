"""Run pinned official sample_t2i.py on three simple prompts, without Inchworm.

Only the prompt list and image saving are intercepted. Upstream model loading,
T5 encoding, padding, native sampling, and VQ decoding execute unmodified.
This is a diagnostic, not a benchmark or a claim of seed-wise image identity.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json,sha

PROMPTS=['a photo of a bench','a photo of a red apple','a photo of a dog']
SOURCE_REVISION='ce98ec41803a74a90ce68c40ababa9eaeffeb4ec'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source-root','data-root','out']: p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); source=a.source_root.resolve(); data=a.data_root.resolve(); out=a.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty diagnostic output directory')
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_REVISION:
        raise ValueError('Unexpected upstream LlamaGen version')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True)
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    import torch
    import torchvision
    from language.t5 import T5Embedder
    if not torch.cuda.is_available(): raise RuntimeError('Use an allocated GPU, not a login node')
    directory=data/'models/LlamaGen'
    checkpoint=directory/'t2i_XL_stage2_512.pt'
    vq=directory/'vq_ds16_t2i.pt'; t5=directory/'t5-ckpt'
    for f in [checkpoint,vq,t5/'flan-t5-xl/config.json']:
        if not f.is_file(): raise ValueError(f'Missing asset {f}')
    script=source/'autoregressive/sample/sample_t2i.py'
    native_args=[str(script),'--vq-ckpt',str(vq),'--gpt-ckpt',str(checkpoint),
        '--t5-path',str(t5),'--t5-model-type','flan-t5-xl','--gpt-model','GPT-XL',
        '--image-size','512','--precision','bf16','--cfg-scale','7.5',
        '--top-k','1000','--top-p','1.0','--temperature','1.0','--seed','42']
    metadata=dict(status='started',scope='diagnostic only; not benchmark',prompts=PROMPTS,seed=42,
        checkpoint=str(checkpoint),batch_size=3,resolution=512,source_revision=SOURCE_REVISION,
        official_script_sha256=sha(script),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        arguments=native_args,interceptions=['replace hardcoded demonstration prompts','save individual images plus native grid'],
        notice='Same seed does not imply identity with earlier batch=1 outputs; initialization and batch sizes differ')
    atomic_json(out/'diagnostic.json',metadata)
    original_embed=T5Embedder.get_text_embeddings
    original_save=torchvision.utils.save_image
    calls={'embed':0,'save':0}
    def embed(self,ignored_texts):
        calls['embed']+=1
        features,masks=original_embed(self,PROMPTS)
        if not torch.isfinite(features).all(): raise ValueError('Non-finite text features')
        metadata['text_token_counts']=[int(x) for x in masks.sum(-1).tolist()]
        print('AUDIT_PROMPTS',json.dumps(PROMPTS),flush=True)
        print('T5_VALID_TOKENS',metadata['text_token_counts'],flush=True)
        return features,masks
    def save(samples,filename,**kwargs):
        calls['save']+=1
        if samples.shape[0]!=3: raise ValueError('Expected all three diagnostic images')
        original_save(samples,out/'official_grid.png',**kwargs)
        for i,image in enumerate(samples): original_save(image,out/f'{i:02d}.png',**kwargs)
    previous=os.getcwd()
    try:
        os.chdir(out)
        with patch.object(T5Embedder,'get_text_embeddings',embed),patch.object(torchvision.utils,'save_image',save),patch.object(sys,'argv',native_args):
            runpy.run_path(str(script),run_name='__main__')
    finally: os.chdir(previous)
    if calls!={'embed':1,'save':1}: raise ValueError('Official demo interface changed')
    metadata.update(status='completed',image_sha256={p.name:sha(p) for p in sorted(out.glob('*.png'))})
    atomic_json(out/'diagnostic.json',metadata)
    print('LLAMAGEN_OFFICIAL_NATIVE_AUDIT_OK',out,flush=True)


if __name__=='__main__': main()

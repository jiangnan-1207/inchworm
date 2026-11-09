"""Thirty Lumina 2.0 backbone/original-Inchworm pairs, with a GPU parity gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import subprocess

from inchworm_core import Config,generate
from lumina_layout import Layout
from protocol import atomic_json,bind_run,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    root=a.root.resolve(); data=a.data_root.resolve(); out=a.out.resolve()
    if not 0<=a.shard_index<a.shards<=30: raise ValueError('Invalid thirty-prompt shard')
    prompts=json.loads((root/'configs/lumina_prompts.json').read_text())
    if len(prompts)!=30 or len(set(prompts))!=30: raise ValueError('Expected thirty distinct Lumina prompts')
    ids=list(range(a.shard_index,30,a.shards)); layout=Layout()
    pin=json.loads((root/'configs/lumina_sources.json').read_text())
    source=root/'benchmarks/Lumina-mGPT-2.0'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=pin['source_revision']:
        raise ValueError('Lumina source revision differs')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True)
    spec={'model':'Lumina-mGPT-2.0','source':pin,'layout':asdict(layout),'methods':['backbone','inchworm'],
        'controller_config':asdict(Config()),'shards':a.shards,'shard_index':a.shard_index,
        'prompt_ids':ids,'prompts':prompts,'seed_rule':'22026 + original_prompt_id',
        'cfg':4.0,'top_k':4096,'temperature':1.0,'dtype':'bf16','attention':'sdpa',
        'fixed_size_header':True,'search_tokens':'image codes only; fixed newlines enter model cache',
        'quantized':False,'jacobi':False,'source_sha256':{f.name:sha(f) for f in sorted(root.glob('*.py'))},
        'notice':'Original engineering Inchworm; no guard, adaptive temperature, or mixed H/G sites'}
    record_dir=out/'workers'/str(a.shard_index)
    bind_run(record_dir,spec)
    import torch
    import numpy as np
    from lumina_backend import LuminaBackend,State
    from lumina_runtime import load_solver,prepare_prompt,native_generate,check_probabilities
    if not torch.cuda.is_available(): raise RuntimeError('Run in an allocated GPU job')
    # Use the full-precision source model; no low-memory quantized substitution.
    print('GPU',torch.cuda.get_device_name(0),'TOTAL_BYTES',torch.cuda.get_device_properties(0).total_memory,flush=True)
    with torch.inference_mode():
        solver=load_solver(root,data,layout)
        prompt,header=prepare_prompt(solver,prompts[ids[0]],layout)
        torch.manual_seed(123)
        parity=check_probabilities(solver,prompt,header,layout)
        atomic_json(record_dir/'parity.json',parity)
        for i in ids:
            prompt,header=prepare_prompt(solver,prompts[i],layout)
            for method in ['backbone','inchworm']:
                path=out/method/f'{i:04d}.png'; receipt=path.with_suffix('.json')
                if receipt.exists():
                    old=json.loads(receipt.read_text())
                    if old['specification']!=spec or old['image_sha256']!=sha(path): raise ValueError('Resume mismatch')
                    print('SKIP_COMPLETE',method,i,flush=True); continue
                seed=22026+i
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                if method=='backbone':
                    native=native_generate(solver,prompt,layout.native_length)
                    sequence=native.sequences[0,prompt.shape[1]:].tolist(); del native
                    layout.validate_native(sequence)
                    packed=header+sequence; backtracks=0; events=[]
                else:
                    backend=LuminaBackend(solver.model,prompt,header,layout)
                    result=generate(backend,State(),layout.length,Config())
                    packed=layout.pack(header,result['tokens'])
                    backtracks=result['backtracks']; events=result['events']
                    del backend,result
                _,images=solver.decode_ids(packed)
                if len(images)!=1 or images[0].size!=(768,768): raise ValueError('Incomplete Lumina image decode')
                torch.cuda.synchronize(); seconds=time.perf_counter()-start
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.png.part'); images[0].save(temporary,format='PNG'); temporary.replace(path)
                atomic_json(receipt,{'specification':spec,'prompt':prompts[i],'prompt_id':i,'seed':seed,
                    'method':method,'seconds_including_decode':seconds,'backtracks':backtracks,'events':events,
                    'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'image_sha256':sha(path)})
                del images,packed,events
                print('LUMINA_PAIR_IMAGE_OK',i,method,'backtracks=',backtracks,flush=True)
    atomic_json(record_dir/'completed.json',{'status':'completed','pairs':len(ids),'images':2*len(ids)})
    print('LUMINA_PAIRS_SHARD_OK',a.shard_index,len(ids),flush=True)


if __name__=='__main__': main()

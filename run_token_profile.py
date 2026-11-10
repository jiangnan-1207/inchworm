"""Supplement original Inchworm's existing 100-image profile with token counts."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path

from experiment_plan import select_100
from inchworm_core import Config
from protocol import atomic_json, bind_run, commit_image, completed, load_prompts, paths, sha
from token_accounting import summarize_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root',required=True,type=Path,
        help='Existing checkout holding benchmarks/geneval, LlamaGen and setup_records')
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--model',required=True,choices=['janus1b','llamagen'])
    args=parser.parse_args()
    rows=load_prompts(args.assets_root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    ids=select_100(rows)
    import numpy as np
    import torch
    from variant_model_runner import load_model
    if not torch.cuda.is_available(): raise RuntimeError('An allocated CUDA GPU is required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    root=Path(__file__).resolve().parent
    spec={'experiment':'original_inchworm_token_accounting_v1','model':args.model,
        'config':asdict(Config()),'prompt_ids':ids,'seed_rule':'42 + 4 * prompt_id',
        'prompt_sha256':sha(args.assets_root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl'),
        'source_sha256':{p.name:sha(p) for p in sorted(root.glob('*.py'))},
        'asset_revisions':json.loads((args.assets_root/'benchmarks/setup_records/model_revisions_v1.json').read_text()),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'batch_size':1,'warmup_images_excluded':1,
        'notice':'Separate accounting run; does not replace existing latency/memory measurements. '
                 'Backbone counts are an analytical reference, not a new measured baseline.'}
    out=args.out/args.model
    bind_run(out,spec)
    records=[]
    pending=[]
    for i in ids:
        png,receipt=paths(out,'inchworm',i,0)
        if completed(png,receipt,spec,rows[i]):
            record=json.loads(receipt.read_text(encoding='utf-8'))
            if record['sample_id']!=i or record['seed']!=42+4*i:
                raise ValueError('Stored sample identity mismatch')
            records.append(record)
        else: pending.append(i)
    if pending:
        with torch.inference_mode():
            # Existing native-prefix checks remain mandatory, including LlamaGen's gate.
            run=load_model(args.model,args.assets_root,args.data_root,collect_token_metrics=True)

            def produce(text,seed):
                gc.collect(); torch.cuda.empty_cache()
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                return run(text,'inchworm')

            warmup,_,_=produce(rows[ids[0]]['prompt'],0)
            del warmup
            for i in pending:
                image,_,diag=produce(rows[i]['prompt'],42+4*i)
                # Deliberately omit timings: this is an instrumented accounting run.
                commit_image(out,'inchworm',i,0,rows[i],spec,image,
                    {'sample_id':i,'seed':42+4*i,'method':'inchworm',**diag})
                del image
                _,receipt=paths(out,'inchworm',i,0)
                record=json.loads(receipt.read_text(encoding='utf-8'))
                records.append(record)
                m=record['token_metrics']['inchworm']
                print('TOKEN_SAMPLE_OK',args.model,i,'backtracks=',m['backtracks'],
                    'regeneration=',m['regeneration_tokens'],
                    'image_inputs_per_branch=',m['model_input_image_token_positions_per_branch'],flush=True)
    if {r['sample_id'] for r in records}!=set(ids): raise ValueError('Wrong completed prompt set')
    summary=summarize_metrics(records)
    atomic_json(out/'token_summary.json',{'specification':spec,'results':summary,
        'notice':'100-image token profiling subset; not a benchmark quality score'})
    atomic_json(out/'completed.json',{'status':'completed','images':100,'method':'original_inchworm'})
    print(json.dumps(summary,indent=2),flush=True)
    print('ORIGINAL_INCHWORM_TOKEN_PROFILE_OK',args.model,flush=True)


if __name__=='__main__': main()

"""Independent variant comparison: pairs by default, profiling only on request."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

from experiment_plan import pair_prompts_for_model, select_100, summarize
from inchworm_variant import VariantConfig
from inchworm_variant_rules import TemperatureConfig
from protocol import atomic_json, bind_run, load_prompts, sha

METHODS=['backbone','inchworm','inchworm_variant']


def summarize_counts(records,expected):
    result=summarize(records,expected)
    for method,summary in result.items():
        group=[r for r in records if r['method']==method]
        for r in group:
            if r['checks']!=r['accepted_edits']+r['kept_original']:
                raise ValueError('Inconsistent check accounting')
        for field in ['checks','accepted_edits','kept_original']:
            total=sum(r[field] for r in group)
            summary['total_'+field]=total
            summary['mean_'+field]=total/len(group)
        summary['backtracks_definition']='accepted window replacements'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen','janus7b'],required=True)
    p.add_argument('--scope',choices=['pairs','profile','both'],default='pairs')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--tmin',type=float,default=0.5)
    p.add_argument('--tmax',type=float,default=1.0)
    p.add_argument('--h0',type=float,default=2.0)
    p.add_argument('--delta',type=float,default=0.0)
    a=p.parse_args()
    if len(a.methods)!=len(set(a.methods)): raise ValueError('Duplicate methods')
    config=VariantConfig(temperature=TemperatureConfig(a.tmin,a.tmax,a.h0),improvement_margin=a.delta)
    config.validate()
    rows=load_prompts(a.root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    group={'janus1b':0,'llamagen':1,'janus7b':2}[a.model]
    cases=[]
    if a.scope in ['profile','both']:
        cases.extend(('profile',i,rows[i]['prompt'],42+4*i) for i in select_100(rows))
    if a.scope in ['pairs','both']:
        cases.extend(('pairs',i,text,10042+i) for i,text in pair_prompts_for_model(a.model))
    import torch
    import numpy as np
    from variant_model_runner import load_model
    if not torch.cuda.is_available(): raise RuntimeError('Run on an allocated CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=a.out/a.model
    specification={'experiment':'separate_variant_v1','model':a.model,'scope':a.scope,
        'methods':a.methods,'variant_config':asdict(config),'cases':cases,'batch_size':1,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'sources':{f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'asset_revisions':json.loads((a.root/'benchmarks/setup_records/model_revisions_v1.json').read_text()),
        'timing':'same batch1 synchronized end-to-end measurement as requested experiments; excludes model loading, GC, cache clearing and PNG I/O',
        'warmup_images_per_method':1,'backtracks_definition':'accepted window replacements'}
    specification=json.loads(json.dumps(specification))
    bind_run(out,specification)
    with torch.inference_mode():
        run=load_model(a.model,a.root,a.data_root,config)

        def measure(text,method,seed):
            gc.collect(); torch.cuda.empty_cache()
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            image,token_seconds,diag=run(text,method)
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            return image,{'latency_seconds':seconds,'token_seconds':token_seconds,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**diag}

        for method in a.methods:
            image,_=measure(cases[0][2],method,0); del image
        records=[]
        for phase,i,text,seed in cases:
            offset=i%len(a.methods)
            for method in a.methods[offset:]+a.methods[:offset]:
                path=out/method/phase/f'{i:05d}.json'; png=path.with_suffix('.png')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['specification']!=specification or record['image_sha256']!=sha(png):
                        raise ValueError('Resume configuration or image differs')
                else:
                    image,diag=measure(text,method,seed)
                    png.parent.mkdir(parents=True,exist_ok=True)
                    temp=png.with_suffix('.png.part'); image.save(temp,format='PNG'); temp.replace(png)
                    record={'phase':phase,'sample_id':i,'prompt':text,'seed':seed,'method':method,
                        'specification':specification,'image_sha256':sha(png),**diag}
                    atomic_json(path,record); del image
                records.append(record)
                print('VARIANT_SAMPLE_OK',a.model,phase,i,method,'checks=',record['checks'],
                    'accepted=',record['accepted_edits'],'kept=',record['kept_original'],flush=True)
    for phase in sorted({c[0] for c in cases}):
        selected=[r for r in records if r['phase']==phase]
        expected=[(method,i) for p,i,_,_ in cases if p==phase for method in a.methods]
        summary=summarize_counts(selected,expected)
        atomic_json(out/f'{phase}_summary.json',{'specification':specification,'results':summary,
            'notice':'Qualitative subset only; not a GenEval score' if phase=='pairs' else '100-image profiling subset; not a full GenEval score'})
    atomic_json(out/'completed.json',{'status':'completed','images':len(records),'methods':a.methods,'scope':a.scope})
    print('VARIANT_EXPERIMENT_OK',a.model,flush=True)


if __name__=='__main__': main()

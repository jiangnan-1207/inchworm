"""Task 4: validate all shards, then call the pinned upstream GenEval evaluator."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

from four_task_protocol import variant_method
from merge_full import merge
from ablation_plan import BENCHMARK_VERSIONS,ablation_flags
from protocol import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen'],required=True)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--variant',choices=BENCHMARK_VERSIONS,required=True)
    p.add_argument('--evaluation-root',type=Path,help='Write links and scores here while reading existing shards')
    a=p.parse_args()
    if not 1<=a.shards<=553: raise ValueError('Invalid shard count')
    method=variant_method(a.model,a.variant)
    evaluation_root=a.evaluation_root if a.evaluation_root is not None else a.run/'merged'
    merge(a.root,a.run,method,a.shards,evaluation_root)
    common=['--root',str(a.root),'--scratch',str(a.data_root),
        '--out',str(evaluation_root),'--scope','full','--stage','preflight']
    # Separate process releases preflight models before full scoring loads them.
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'evaluate_benchmark.py'),*common],check=True)
    import evaluate_benchmark
    evaluate_benchmark.METHODS=[method]
    common[-1]='evaluate'
    sys.argv=[sys.argv[0],*common]
    evaluate_benchmark.main()
    result=json.loads((evaluation_root/'evaluation'/f'{method}.summary.json').read_text())
    if not (result['formal_complete_benchmark'] and result['prompts']==553 and result['images']==2212):
        raise ValueError('Full benchmark coverage was not established')
    result.update({'model':a.model,'method':method,'variant':a.variant,'ablation_flags':ablation_flags(a.variant)})
    atomic_json(evaluation_root/'evaluation'/f'{method}.summary.json',result)
    print('TASK4_FULL_GENEVAL_OK',method,'overall=',result['overall'],flush=True)


if __name__=='__main__': main()

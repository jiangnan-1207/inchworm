"""Deterministic samples and reporting for the three requested experiments."""
import json
from pathlib import Path
import random
import statistics

PAIR_MODELS=['janus1b','llamagen','janus7b']
PAIR_COUNT=30
PROMPTS_BY_MODEL=json.loads((Path(__file__).parent/'configs/qualitative_prompts.json').read_text(encoding='utf-8'))
PAIR_PROMPTS=[p for model in PAIR_MODELS for p in PROMPTS_BY_MODEL[model]]
if any(len(PROMPTS_BY_MODEL[m])!=PAIR_COUNT for m in PAIR_MODELS) or len(set(PAIR_PROMPTS))!=90:
    raise ValueError('Expected thirty distinct prompts per model, ninety globally')


def pair_prompts_for_model(model,count=PAIR_COUNT):
    if model not in PAIR_MODELS or not 1<=count<=PAIR_COUNT: raise ValueError('Invalid qualitative selection')
    offset=PAIR_MODELS.index(model)*PAIR_COUNT
    return [(offset+i,prompt) for i,prompt in enumerate(PROMPTS_BY_MODEL[model][:count])]


def select_100(rows):
    # Proportional six-task coverage, rather than the first 100 official rows.
    counts={'single_object':14,'two_object':18,'counting':14,'colors':17,'position':18,'color_attr':19}
    chosen=[]
    rng=random.Random(20261108)
    for task,count in counts.items():
        candidates=[i for i,row in enumerate(rows) if row['tag']==task]
        chosen.extend(rng.sample(candidates,count))
    return sorted(chosen)


def summarize(records,expected):
    keys=[(r['method'],r['sample_id']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
        raise ValueError('Missing or duplicate measured samples')
    output={}
    for method in sorted({m for m,_ in expected}):
        group=[r for r in records if r['method']==method]
        output[method]={
            'images':len(group),
            'mean_latency_seconds':statistics.mean(r['latency_seconds'] for r in group),
            'median_latency_seconds':statistics.median(r['latency_seconds'] for r in group),
            'mean_token_seconds':statistics.mean(r['token_seconds'] for r in group),
            'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in group),
            'max_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in group),
            'mean_backtracks':statistics.mean(r['backtracks'] for r in group),
            'total_backtracks':sum(r['backtracks'] for r in group),
        }
    return output

"""Deterministic samples and reporting for the three requested experiments."""
from collections import Counter
import random
import statistics

PAIR_PROMPTS = [
    'A photo of a red apple to the left of a blue cup on a wooden table.',
    'A photo of three yellow bananas on a white plate.',
    'A photo of a green backpack and a red chair.',
    'A photo of a blue bicycle to the right of a yellow car.',
    'A photo of two red apples and one green apple in a bowl.',
    'A photo of a black cat under a white table.',
    'A photo of a red book on top of a blue suitcase.',
    'A photo of four orange carrots on a white kitchen counter.',
    'A photo of a yellow umbrella next to a purple backpack.',
]


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

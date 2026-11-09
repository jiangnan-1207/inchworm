"""Read finished task results; never regenerate images just to report task 2."""
import argparse
import json
from pathlib import Path


def task2(run):
    path=Path(run)/'llamagen/profile_summary.json'
    data=json.loads(path.read_text())
    result=data['results']['inchworm']
    if result['images']!=100: raise ValueError('Need exactly 100 completed Inchworm images')
    if abs(result['mean_backtracks']-result['total_backtracks']/100)>1e-12:
        raise ValueError('Inconsistent backtrack summary')
    return {'method':'original_inchworm','model':'llamagen','images':100,
        'total_backtracks':result['total_backtracks'],'mean_backtracks':result['mean_backtracks'],
        'definition':'triggered window replacement events; zero-count images included','source':str(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(task2(a.run),indent=2))


if __name__=='__main__': main()

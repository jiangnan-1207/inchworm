"""Dataset identities and strict completion checks for DPG/full and TIIF/testmini."""
import csv
import hashlib
import json
from pathlib import Path
from protocol import sha


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def build_manifest(root):
    root=Path(root); rows=[]; hashes={}
    for p in sorted((root/'benchmarks/ELLA/dpg_bench/prompts').glob('*.txt')):
        hashes[str(p.relative_to(root))]=sha(p)
        rows.append(dict(benchmark='dpg',key=p.stem,prompt=p.read_text(encoding='utf-8').strip(),samples=4))
    if len(rows)!=1065: raise ValueError('Expected all 1065 DPG prompts')
    csv_path=root/'benchmarks/ELLA/dpg_bench/dpg_bench.csv'
    hashes[str(csv_path.relative_to(root))]=sha(csv_path)
    # Keep all CSV questions, including the first data row skipped by upstream's parser.
    for p in sorted((root/'benchmarks/TIIF-Bench/data/testmini_prompts').glob('*.jsonl')):
        hashes[str(p.relative_to(root))]=sha(p)
        data=jsonl(p)
        if not data: raise ValueError('Empty TIIF file')
        kind=data[0]['type']
        ep=root/f'benchmarks/TIIF-Bench/data/testmini_eval_prompts/{kind}_eval_prompts.jsonl'
        hashes[str(ep.relative_to(root))]=sha(ep); questions=jsonl(ep)
        if len(questions)!=len(data): raise ValueError(f'TIIF prompt/question mismatch {kind}')
        for i,(line,q) in enumerate(zip(data,questions)):
            if line['type']!=kind or q['type']!=kind: raise ValueError('Mixed TIIF types')
            if not q['yn_question_list'] or len(q['yn_question_list'])!=len(q['yn_answer_list']):
                raise ValueError('Missing TIIF questions/answers')
            for length in ['short','long']:
                rows.append(dict(benchmark='tiif',key=f'{kind}/{length}/{i}',kind=kind,index=i,
                    length=length,prompt=line[f'{length}_description'],samples=1,
                    questions=q['yn_question_list'],answers=q['yn_answer_list']))
    tiif=[r for r in rows if r['benchmark']=='tiif']
    if len({r['kind'] for r in tiif})!=39 or not tiif: raise ValueError('Expected 39 TIIF dimensions')
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return dict(schema=1,rows=rows,source_sha256=hashes,prompt_sha256=identity,
        counts={b:sum(r['samples'] for r in rows if r['benchmark']==b) for b in ['dpg','tiif']},
        tiif_scope='official testmini; both short and long; not full TIIF')


def sample_paths(run,model,method,row,j):
    base=Path(run)/'images'/model/method/row['benchmark']/row['key']
    return base/f'{j}.png',base/f'{j}.json'


def check_sample(run,model,method,row,j,spec):
    image,receipt=sample_paths(run,model,method,row,j)
    if not image.is_file() or not receipt.is_file(): return False
    record=json.loads(receipt.read_text(encoding='utf-8'))
    if record['spec']!=spec or record['row']!=row or record['image_sha256']!=sha(image):
        raise ValueError(f'Mismatched receipt {image}')
    return True


def validate_set(run,model,method,rows,spec):
    for row in rows:
        for j in range(row['samples']):
            if not check_sample(run,model,method,row,j,spec):
                raise ValueError(f'Incomplete generation: {row["key"]} / {j}')
    return sum(r['samples'] for r in rows)

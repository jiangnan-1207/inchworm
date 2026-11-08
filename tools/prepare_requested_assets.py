"""Download Janus-Pro-1B only; reuse the already verified cluster assets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json, load_prompts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--existing-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args()
    from huggingface_hub import HfApi,snapshot_download
    bench=ROOT/'benchmarks'; bench.mkdir(exist_ok=True)
    for name,revision in [('geneval','af4902f24d3ca90ebbb446dd9891a59e0f82725f'),
                          ('LlamaGen','ce98ec41803a74a90ce68c40ababa9eaeffeb4ec')]:
        source=a.existing_root/'benchmarks'/name
        head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if head!=revision: raise ValueError(f'{name} revision differs')
        target=bench/name
        if target.is_symlink():
            if target.resolve()!=source.resolve(): raise ValueError('Wrong source symlink')
        elif target.exists():
            raise ValueError(f'Refusing to replace {target}; use a fresh experiment checkout')
        else: target.symlink_to(source.resolve(),target_is_directory=True)
    load_prompts(bench/'geneval/prompts/evaluation_metadata.jsonl')
    destination=bench/'setup_records/model_revisions_v1.json'
    if destination.exists(): revisions=json.loads(destination.read_text())
    else: revisions=json.loads((a.existing_root/'benchmarks/setup_records/model_revisions_v1.json').read_text())
    if 'deepseek-ai/Janus-Pro-7B' not in revisions:
        revisions['deepseek-ai/Janus-Pro-7B']='5c3eb3fb2a3b61094328465ba61fcd4272090d67'
    repo='deepseek-ai/Janus-Pro-1B'
    if repo not in revisions: revisions[repo]=HfApi().model_info(repo).sha
    atomic_json(destination,revisions)
    target=a.data_root/'models/Janus-Pro-1B'
    snapshot_download(repo,revision=revisions[repo],local_dir=str(target),max_workers=2,
        allow_patterns=['*.json','*.model','*.txt','*.bin','*.safetensors'])
    indices=list(target.glob('*.index.json'))
    weights=list(target.glob('*.safetensors'))+list(target.glob('*.bin'))
    if not weights: raise ValueError('Missing 1B checkpoint weights')
    for index in indices:
        for filename in set(json.loads(index.read_text())['weight_map'].values()):
            if not (target/filename).is_file(): raise ValueError(f'Missing {filename}')
    from janus.models import VLChatProcessor
    VLChatProcessor.from_pretrained(str(target),local_files_only=True)
    print('REQUESTED_ASSETS_OK',flush=True)


if __name__=='__main__': main()

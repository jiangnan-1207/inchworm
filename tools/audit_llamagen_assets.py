"""Read-only offline comparison with pinned official LlamaGen/T5 fingerprints."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import atomic_json

SOURCE_REVISION='ce98ec41803a74a90ce68c40ababa9eaeffeb4ec'
T5_REQUIRED=['config.json','tokenizer_config.json','special_tokens_map.json','spiece.model']


def verify_file(path,expected):
    record={'path':str(path),'expected':expected}
    if not path.is_file(): return {**record,'status':'missing'}
    size=path.stat().st_size
    if size!=expected['size']: return {**record,'status':'size_mismatch','actual_size':size}
    # LFS objects use their content SHA256; ordinary repository files use the
    # Git blob digest, including the header. The pointer's blob ID is NOT the
    # digest of a downloaded LFS weight file.
    lfs=bool(expected.get('sha256'))
    digest=hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs: digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    actual=digest.hexdigest()
    wanted=expected['sha256'] if lfs else expected['git_blob_sha1']
    return {**record,'status':'match' if actual==wanted else 'hash_mismatch',
        'algorithm':'sha256' if lfs else 'git_blob_sha1','actual_digest':actual}


def audit_assets(model_dir,manifest):
    records=[]
    errors=[]
    for model in manifest['models']:
        folder=model_dir/model['subdir']
        expected={f['name']:f for f in model['files']}
        if model['repo']=='peizesun/llamagen_t2i':
            names=['t2i_XL_stage2_512.pt','vq_ds16_t2i.pt']
        elif model['repo']=='google/flan-t5-xl':
            names=T5_REQUIRED.copy()
            # Check all installed inference-related files, including a stale
            # alternate format that Transformers might prefer at load time.
            names+=sorted(p.name for p in folder.iterdir() if p.is_file() and (
                p.name.startswith(('model','pytorch_model','tokenizer','added_tokens','adapter_'))
                or p.name=='special_tokens_map.json')) if folder.is_dir() else []
            if not any((folder/name).is_file() for name in [
                    'model.safetensors','model.safetensors.index.json','pytorch_model.bin','pytorch_model.bin.index.json']):
                errors.append('No complete supported T5 weight entrypoint found')
            for name in ['model.safetensors.index.json','pytorch_model.bin.index.json']:
                index=folder/name
                if not index.exists(): continue
                try:
                    shards=set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                    if not shards or not all(isinstance(n,str) and Path(n).name==n and n in expected for n in shards):
                        raise ValueError('Unexpected shard names')
                    names+=sorted(shards)
                except (ValueError,KeyError,TypeError): errors.append('Invalid T5 index: '+str(index))
        else: raise ValueError('Unknown manifest model')
        for name in dict.fromkeys(names):
            if name not in expected:
                record={'path':str(folder/name),'status':'unrecognized_inference_file'}
            else: record=verify_file(folder/name,expected[name])
            records.append(record)
            print(record['status'].upper(),model['repo'],name,flush=True)
    passed=not errors and all(r['status']=='match' for r in records)
    return {'status':'assets_match_pinned_sources' if passed else 'asset_audit_failed',
        'manifest':manifest,'files':records,'errors':errors,
        'notice':'File integrity only. Matching files do not prove correct model execution or image quality.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'configs/llama_asset_fingerprints.json').read_text(encoding='utf-8-sig'))
    report=audit_assets(args.data_root/'models/LlamaGen',manifest)
    source=args.source_root.resolve()
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    clean=subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--']).returncode==0
    report['source']={'directory':str(source),'head':head,'tracked_files_clean':clean}
    if head!=SOURCE_REVISION or not clean:
        report['errors'].append('Upstream source differs from pinned clean version')
        report['status']='asset_audit_failed'
    packages={}
    for name in ['torch','torchvision','transformers','tokenizers','accelerate','huggingface-hub','sentencepiece','numpy']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]='not installed'
    modules={}
    for name in ['torch','transformers','tokenizers','sentencepiece']:
        spec=importlib.util.find_spec(name)
        modules[name]=None if spec is None else spec.origin
    report['environment']={'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
        'packages':packages,'module_locations':modules}
    atomic_json(args.out,report)
    print('REPORT',args.out,flush=True)
    if report['status']!='assets_match_pinned_sources':
        print('LLAMAGEN_ASSET_AUDIT_FAILED',flush=True)
        raise SystemExit(1)
    print('LLAMAGEN_ASSET_HASHES_AND_SOURCE_OK',flush=True)


if __name__=='__main__': main()

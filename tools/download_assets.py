"""Official asset downloads. Called by bootstrap in the matching isolated env."""
import argparse
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def download_janus():
    import json
    from huggingface_hub import snapshot_download
    root = Path(os.environ['BENCH_ROOT'])
    data = Path(os.environ['BENCH_SCRATCH'])
    revision = '5c3eb3fb2a3b61094328465ba61fcd4272090d67'
    record = root/'setup_records/model_revisions_v1.json'
    revisions = json.loads(record.read_text()) if record.exists() else {}
    if revisions.get('deepseek-ai/Janus-Pro-7B', revision) != revision:
        raise RuntimeError('Existing Janus revision differs')
    revisions['deepseek-ai/Janus-Pro-7B'] = revision
    record.write_text(json.dumps(revisions,indent=2)+'\n')
    target = data/'models/Janus-Pro-7B'
    snapshot_download('deepseek-ai/Janus-Pro-7B',revision=revision,local_dir=str(target),max_workers=2,
        allow_patterns=['*.json','*.model','*.txt','*.bin','*.safetensors'])
    indices = list(target.glob('*.index.json'))
    if not indices:
        raise RuntimeError('Missing Janus checkpoint weight index')
    for index in indices:
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            if not (target/name).is_file() or (target/name).stat().st_size == 0:
                raise RuntimeError(f'Missing model shard: {name}')
    from janus.models import VLChatProcessor
    VLChatProcessor.from_pretrained(str(target),local_files_only=True)
    print('JANUS_ASSETS_OK',flush=True)

def download_llamagen():
    from pathlib import Path
    import hashlib, json, os
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    scratch = Path(os.environ['BENCH_SCRATCH'])
    root = Path(os.environ['BENCH_ROOT'])
    record = root/'setup_records/model_revisions_v1.json'
    revisions = json.loads(record.read_text()) if record.exists() else {}
    api = HfApi()
    for repo in ['peizesun/llamagen_t2i', 'google/flan-t5-xl']:
        if repo not in revisions:
            revisions[repo] = api.model_info(repo).sha
            record.write_text(json.dumps(revisions, indent=2)+'\n')
    llama = scratch/'models/LlamaGen'
    for name in ['vq_ds16_t2i.pt', 't2i_XL_stage2_512.pt']:
        p = Path(hf_hub_download('peizesun/llamagen_t2i', name,
                 revision=revisions['peizesun/llamagen_t2i'], local_dir=str(llama)))
        assert p.stat().st_size > 1000000
        print('LLAMA_WEIGHT_READY:', name, p.stat().st_size, flush=True)
    t5 = llama/'t5-ckpt/flan-t5-xl'
    info = api.model_info('google/flan-t5-xl', revision=revisions['google/flan-t5-xl'])
    names = {x.rfilename for x in info.siblings}
    formats = ['*.bin', '*.bin.index.json'] if any(x.endswith('.bin') for x in names) else ['*.safetensors', '*.safetensors.index.json']
    snapshot_download('google/flan-t5-xl', revision=revisions['google/flan-t5-xl'],
        local_dir=str(t5), max_workers=2,
        allow_patterns=['config.json', 'tokenizer*.json', 'special_tokens_map.json', 'spiece.model'] + formats)
    assert (t5/'config.json').is_file() and (t5/'spiece.model').is_file()
    indices = list(t5.glob('*.index.json'))
    for index in indices:
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            assert (t5/name).stat().st_size > 0, name
    assert indices or (t5/'pytorch_model.bin').exists() or (t5/'model.safetensors').exists()
    from transformers import AutoTokenizer, T5Config
    tok = AutoTokenizer.from_pretrained(str(t5), local_files_only=True)
    conf = T5Config.from_pretrained(str(t5), local_files_only=True)
    assert conf.d_model == 2048 and len(tok('a red apple')['input_ids']) > 0
    manifest = {'revisions': revisions, 't5_directory': str(t5), 'llama_directory': str(llama),
                't5_weight_bytes': sum(p.stat().st_size for p in t5.glob('*') if p.suffix in ['.bin', '.safetensors'])}
    (root/'setup_records/llama_assets_v1.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('LLAMAGEN_ASSETS_OK', flush=True)

def download_geneval():
    from pathlib import Path
    import hashlib, json, os, urllib.request
    import torch, open_clip
    scratch = Path(os.environ['BENCH_SCRATCH'])
    root = Path(os.environ['BENCH_ROOT'])
    weights = scratch/'models/GenEval'
    weights.mkdir(parents=True, exist_ok=True)
    target = weights/'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
    url = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
    receipt = weights/'detector_receipt_v1.json'
    def sha256(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    verified = target.exists() and receipt.exists() and sha256(target) == json.loads(receipt.read_text())['sha256']
    if not verified:
        part = target.with_suffix('.pth.part')
        with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as out:
            for block in iter(lambda: response.read(8*1024*1024), b''):
                out.write(block)
        assert part.stat().st_size > 1000000
        part.replace(target)
    checkpoint = torch.load(str(target), map_location='cpu')
    assert isinstance(checkpoint, dict) and 'state_dict' in checkpoint
    del checkpoint
    receipt.write_text(json.dumps({'url':url, 'path':str(target), 'sha256':sha256(target)}, indent=2)+'\n')
    print('GENEVAL_DETECTOR_WEIGHT_OK', flush=True)
    clipdir = weights/'clip_cache'
    model, _, transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device='cpu', cache_dir=str(clipdir))
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    with torch.inference_mode():
        features = model.encode_text(tokenizer(['a red apple']))
    assert features.shape == (1,768) and torch.isfinite(features).all()
    print('GENEVAL_CLIP_WEIGHT_OK', flush=True)
    report = {'torch':torch.__version__, 'detector':str(target), 'clip_arch':'ViT-L-14',
              'clip_pretrained':'openai', 'clip_cache':str(clipdir), 'clip_benchmark':'1.6.1',
              'gpu_inference_validated':False, 'official_553_prompt_sha256':'5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13'}
    (root/'setup_records/geneval_assets_v1.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=['janus','llamagen','geneval'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    data=args.data_root.expanduser().resolve()
    (data/'models').mkdir(parents=True,exist_ok=True)
    (ROOT/'benchmarks/setup_records').mkdir(parents=True,exist_ok=True)
    os.environ['BENCH_ROOT']=str(ROOT/'benchmarks')
    os.environ['BENCH_SCRATCH']=str(data)
    {'janus':download_janus,'llamagen':download_llamagen,'geneval':download_geneval}[args.component]()


if __name__=='__main__':
    main()

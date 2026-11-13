"""Use upstream GenEval unchanged; redirect only the CLIP cache location."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from protocol import METHODS, TASKS, atomic_json, load_prompts, select_ids, validate_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--scope',choices=['pilot','full'],required=True)
    parser.add_argument('--stage',choices=['preflight','evaluate'],required=True)
    args = parser.parse_args()
    root, scratch, out = Path(args.root), Path(args.scratch), Path(args.out)
    rows = load_prompts(root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl')
    ids = select_ids(rows,args.scope)
    samples = 4 if args.scope == 'full' else 1
    import torch
    import open_clip
    from mmcv.ops import nms
    assert torch.cuda.is_available(), 'GPU evaluation requires a CUDA GPU'
    boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]],device='cuda')
    _, keep = nms(boxes,torch.tensor([0.9,0.8],device='cuda'),0.5)
    assert keep.tolist() == [0]
    print('GENEVAL_CUDA_NMS_OK',flush=True)
    create = open_clip.create_model_and_transforms

    def cached_create(*positional, **kwargs):
        kwargs.setdefault('cache_dir',str(scratch/'models/GenEval/clip_cache'))
        return create(*positional,**kwargs)

    open_clip.create_model_and_transforms = cached_create
    path = root/'benchmarks/geneval/evaluation/evaluate_images.py'
    spec = importlib.util.spec_from_file_location('upstream_geneval',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = root/'benchmarks/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    module.args = SimpleNamespace(model_config=str(config),model_path=str(scratch/'models/GenEval'),options={})
    load_start = time.perf_counter()
    module.object_detector, (module.clip_model,module.transform,module.tokenizer), module.classnames = module.load_models(module.args)
    module.clip_model.eval()
    module.THRESHOLD = 0.3
    module.COUNTING_THRESHOLD = 0.9
    module.MAX_OBJECTS = 16
    module.NMS_THRESHOLD = 1.0
    module.POSITION_THRESHOLD = 0.1
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-load_start
    if args.stage == 'preflight':
        # Test the helpers against the old GenEval environment's mathematical
        # path for list templates; this is not a claim about all helper APIs.
        from torch.nn import functional as F
        classes = ['red','blue','green']
        templates = ['a photo of a {c} apple','a photo of a {c}-colored apple','a photo of a {c} object']
        reference = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for color in classes:
                encoded = module.clip_model.encode_text(module.tokenizer([x.format(c=color) for x in templates]).cuda())
                vector = F.normalize(encoded,dim=-1).mean(dim=0)
                reference.append(vector/vector.norm())
            reference = torch.stack(reference,dim=1).cuda()
        candidate = module.zsc.zero_shot_classifier(module.clip_model,module.tokenizer,classes,templates,'cuda')
        torch.testing.assert_close(candidate,reference,atol=3e-4,rtol=3e-4)
        print('COLOR_HELPER_LIST_TEMPLATE_FORMULA_CHECK_OK',flush=True)
        from PIL import Image
        work_tmp = Path(os.environ.get('TMPDIR',str(scratch/'tmp')))
        work_tmp.mkdir(parents=True,exist_ok=True)
        apple = work_tmp/f'geneval_cuda_preflight_{os.getpid()}.png'
        Image.new('RGB',(256,256),(220,30,30)).save(apple)
        # Synthetic input exercises GPU code; it is never a benchmark sample.
        module.evaluate_image(str(apple),{'tag':'single_object','prompt':'a photo of an apple',
                                       'include':[{'class':'apple','count':1}]})
        from PIL import Image
        with Image.open(apple) as image:
            w,h = image.size
            # Exercise the crop/classification CUDA path even if detector did
            # not find an apple. This forced crop is a test, not a scored result.
            labels = module.color_classification(image.convert('RGB'),[(torch.tensor([0.,0.,w,h,1.]).numpy(),None)],'apple')
        assert len(labels)==1 and labels[0] in module.COLORS
        apple.unlink()
        print('GENEVAL_DETECTOR_AND_COLOR_GPU_PREFLIGHT_OK',flush=True)
        return

    results_dir = out/'evaluation'
    summaries = {}
    for method in METHODS:
        count = validate_outputs(out,method,rows,ids,samples)
        module.args.imagedir = str(out/method)
        results_dir.mkdir(parents=True,exist_ok=True)
        # Upstream main writes only at the end. Use a temporary filename so an
        # interrupted evaluator cannot leave a completed-looking final result.
        final = results_dir/f'{method}.jsonl'
        temp = final.with_name(final.name+'.part')
        module.args.outfile = str(temp)
        torch.cuda.synchronize()
        start = time.perf_counter()
        module.main(module.args)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        records = [json.loads(line) for line in temp.read_text().splitlines() if line.strip()]
        expected = {str(out/method/f'{i:05d}'/'samples'/f'{j:04d}.png') for i in ids for j in range(samples)}
        actual = [r['filename'] for r in records]
        if len(actual)!=count or len(set(actual))!=count or set(actual)!=expected:
            raise ValueError('Evaluation result has missing, duplicate or unexpected images')
        by_task = {tag:[bool(r['correct']) for r in records if r['tag']==tag] for tag in TASKS}
        assert all(by_task.values()), 'All six task categories must be evaluated'
        task_scores = {tag:sum(values)/len(values) for tag,values in by_task.items()}
        full = args.scope == 'full'
        summary = {'scope':args.scope,'formal_complete_benchmark':full,
                   'prompts':len(ids),'images':count,'evaluation_seconds':seconds,
                   'load_seconds_shared':load_seconds,
                   'task_scores':task_scores if full else None,
                   'overall':sum(task_scores.values())/6 if full else None,
                   'pilot_notice':None if full else 'Integration check only; not a benchmark score'}
        temp.replace(final)
        atomic_json(results_dir/f'{method}.summary.json',summary)
        summaries[method]=summary
        print('EVALUATION_METHOD_OK',method,'images=',count,'seconds=',round(seconds,2),flush=True)
    # Each method can be evaluated in a different process. Per-method summary
    # files are authoritative; never overwrite a shared summary with one method.
    if args.scope=='pilot':
        print('PILOT_THREE_GROUP_CHAIN_OK: NOT A COMPLETE BENCHMARK SCORE',flush=True)
    else:
        print('FULL_GENEVAL_METHODS_OK',','.join(summaries),flush=True)


if __name__=='__main__':
    main()

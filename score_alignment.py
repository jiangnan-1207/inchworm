"""Local scoring only; TIIF Qwen2.5-VL-7B/PaddleOCR is a separately labelled protocol."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
from alignment_metrics import aggregate_tiif,dependency_score,dpg_questions,parse_answers
from alignment_protocol import sample_paths,validate_set
from protocol import atomic_json,bind_run,sha
from run_alignment_generation import generation_spec


def qwen_judge(directory):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    processor=AutoProcessor.from_pretrained(directory,local_files_only=True,min_pixels=256*28*28,max_pixels=1280*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(directory,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    def judge(path,questions):
        # Fixed upstream prompt; no random template selection, no target caption.
        prompt=('You are tasked with conducting a careful examination of the provided image. Based on the content '
            'of the image, please answer the following yes or no questions:\n\nQuestions:\n'+'\n'.join(questions)+
            '\n\nEach answer should be on a separate line, starting with yes or no, followed by the reason. '
            'Keep the exact question order. Return exactly one answer per question, without additional content.')
        messages=[dict(role='system',content='You are a professional image critic.'),dict(role='user',
            content=[dict(type='image',image=str(path)),dict(type='text',text=prompt)])]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        with Image.open(path) as image:
            inputs=processor(text=[text],images=[image.convert('RGB')],return_tensors='pt').to('cuda')
        with torch.inference_mode(): result=model.generate(**inputs,do_sample=False,max_new_tokens=2048)
        response=processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        return parse_answers(response,len(questions)),response
    return judge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','data-root','run']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',choices=['janus1b','llamagen'],required=True)
    p.add_argument('--benchmark',choices=['dpg','tiif'],required=True); a=p.parse_args()
    manifest,spec=generation_spec(a.root,a.model)
    rows=[r for r in manifest['rows'] if r['benchmark']==a.benchmark]
    assets_path=a.root/f'benchmarks/setup_records/alignment_{a.benchmark}_judge.json'
    assets=json.loads(assets_path.read_text())
    judge_spec=dict(generator=spec,benchmark=a.benchmark,assets=assets,scoring_source={
        f:sha(a.root/f) for f in ['score_alignment.py','alignment_metrics.py']},
        dpg_first_question='included; upstream first-data-row skip corrected',
        tiif_protocol='Qwen2.5-VL-7B greedy, fixed prompt; PaddleOCR English/GNED; NOT GPT-4o or TIIF-Evaluator leaderboard')
    out=a.run/'scores'/a.model/a.benchmark; bind_run(out,judge_spec)
    for method in ['backbone','inchworm']: validate_set(a.run,a.model,method,rows,spec)
    if a.benchmark=='dpg':
        questions=dpg_questions(a.root/'benchmarks/ELLA/dpg_bench/dpg_bench.csv')
        if set(questions)!={r['key'] for r in rows}: raise ValueError('DPG prompt/question key mismatch')
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        judge=pipeline(Tasks.visual_question_answering,model=assets['model_path'],device='gpu:0')
    else:
        judge=qwen_judge(assets['model_path'])
        from paddleocr import PaddleOCR
        ocr_path=Path(assets['ocr_path'])
        for f,h in assets['ocr_sha256'].items():
            if sha(ocr_path/f)!=h: raise ValueError('OCR assets changed')
        ocr=PaddleOCR(lang='en',use_gpu=False,show_log=False,use_angle_cls=False,
            det_model_dir=str(ocr_path/'det'),rec_model_dir=str(ocr_path/'rec'),cls_model_dir=str(ocr_path/'cls'))
        source=a.root/'benchmarks/TIIF-Bench/eval/cal_gned_and_recall_models.py'
        module_spec=importlib.util.spec_from_file_location('official_gned',source)
        gned=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(gned)
    summary={}
    for method in ['backbone','inchworm']:
        records=[]
        for row in rows:
            for j in range(row['samples']):
                image,_=sample_paths(a.run,a.model,method,row,j)
                receipt=out/'records'/method/row['key']/f'{j}.json'
                if receipt.exists():
                    record=json.loads(receipt.read_text())
                    if record['image_sha256']!=sha(image) or record['judge_spec']!=judge_spec: raise ValueError('Scoring resume mismatch')
                else:
                    record=dict(image_sha256=sha(image),judge_spec=judge_spec,key=row['key'],sample=j)
                    if a.benchmark=='dpg':
                        from PIL import Image
                        q=questions[row['key']]
                        with Image.open(image) as im:
                            answers={qid:judge({'image':im.convert('RGB'),'question':item['question']})['text'].strip().lower()
                                for qid,item in q.items()}
                        if any(s not in ['yes','no'] for s in answers.values()): raise ValueError('Non-binary mPLUG answer')
                        record.update(score=dependency_score(q,answers),answers=answers)
                    else:
                        record.update(kind=row['kind'],length=row['length'])
                        if row['kind']=='text':
                            words=[]
                            for q in row['questions']:
                                match=re.fullmatch(r"Is the text '(.*)' in the image\?",q.strip())
                                if not match: raise ValueError('Unrecognized text ground truth')
                                words.append(match[1].lower())
                            raw=ocr.ocr(str(image),cls=False); results=raw[0] if raw and raw[0] else []
                            predicted=[s[1][0].lower() for s in results] or ['']
                            record.update(text_score=1-gned.matching_based_nled(words,predicted),ocr=predicted,ground_truth=words)
                        else:
                            answers,response=judge(image,row['questions'])
                            expected=[s.strip().lower() for s in row['answers']]
                            record.update(correct=sum(x==y for x,y in zip(answers,expected)),questions=len(expected),
                                answers=answers,response=response)
                    atomic_json(receipt,record)
                records.append(record)
                print('ALIGNMENT_SCORE_OK',a.model,method,a.benchmark,row['key'],j,flush=True)
        if a.benchmark=='dpg':
            # Each prompt has exactly four samples, so the per-sample mean equals mean-of-prompt-means.
            summary[method]=dict(overall=100*sum(r['score'] for r in records)/len(records),images=len(records),prompts=len(rows))
        else: summary[method]=aggregate_tiif(records)
    atomic_json(out/'summary.json',dict(status='complete',scope=manifest['tiif_scope'] if a.benchmark=='tiif' else 'full 1065x4',
        judge_spec=judge_spec,results=summary))
    print(json.dumps(summary,indent=2),flush=True); print('ALIGNMENT_SCORING_COMPLETE',flush=True)


if __name__=='__main__': main()

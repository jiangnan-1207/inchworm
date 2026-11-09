"""Strict aggregation: DPG dependency-aware prompt mean; TIIF nine-group macro mean."""
import csv
import re

GROUPS={
 'attribute':['shape+color','color+texture','texture+color','shape+texture'],
 'relation':['2d_spatial_relation','3d_spatial_relation','action+2d','action+3d'],
 'reasoning':['numeracy','negation','differentiation','comparison'],
 'attribute_relation':['action+color','action+texture','color+2d','color+3d','shape+2d','shape+3d','texture+2d','texture+3d'],
 'attribute_reasoning':['numeracy+color','numeracy+texture','comparison+color','comparison+texture',
                          'differentiation+color','differentiation+texture','negation+color','negation+texture'],
 'relation_reasoning':['numeracy+2d','numeracy+3d','comparison+2d','comparison+3d',
                         'differentiation+2d','differentiation+3d','negation+2d','negation+3d'],
 'style':['style'],'text':['text'],'designer':['real_world']}


def dpg_questions(path):
    groups={}
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            qid=int(r['proposition_id']); deps=[int(x.strip()) for x in r['dependency'].split(',')]
            group=groups.setdefault(r['item_id'],{})
            if qid in group: raise ValueError('Duplicate DPG question')
            group[qid]=dict(question=r['question_natural_language'],dependencies=deps,category=r['tuple'])
    for group in groups.values():
        for qid,q in group.items():
            if any(x and x not in group for x in q['dependencies']): raise ValueError('Missing DPG dependency')
    return groups


def dependency_score(questions,answers):
    if set(questions)!=set(answers): raise ValueError('Missing DPG answers')
    # Follow upstream's single pass in CSV order. The actual CSV contains cycles;
    # recursive DAG propagation would change its protocol or reject valid items.
    scores={qid:float(answer=='yes') for qid,answer in answers.items()}
    for qid,item in questions.items():
        if any(scores[parent]==0 for parent in item['dependencies'] if parent): scores[qid]=0
    return sum(scores.values())/len(scores)


def parse_answers(text,count):
    answers=re.findall(r'^\s*(yes|no)\b',text,flags=re.M|re.I)
    if len(answers)!=count: raise ValueError(f'Judge returned {len(answers)} answers, expected {count}')
    return [s.lower() for s in answers]


def aggregate_tiif(records):
    output={}
    for length in ['short','long']:
        dim={}
        for kind in sum(GROUPS.values(),[]):
            selected=[r for r in records if r['kind']==kind and r['length']==length]
            if not selected: raise ValueError(f'Missing TIIF dimension {kind}/{length}')
            if kind=='text': value=sum(r['text_score'] for r in selected)/len(selected)
            else:
                value=sum(r['correct'] for r in selected)/sum(r['questions'] for r in selected)
            dim[kind]=value
        group={g:sum(dim[k] for k in ks)/len(ks) for g,ks in GROUPS.items()}
        output[length]=dict(overall=100*sum(group.values())/9,groups={k:100*v for k,v in group.items()},
            dimensions={k:100*v for k,v in dim.items()})
    return output

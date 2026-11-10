"""Original Inchworm token accounting; no sampling or selection changes.

Regeneration counts every alternative and its resampled suffix, including losers.
Prediction evaluations count actual distributions, including lookahead, excluding
ready-cache hits. Model input positions are separate (prefill is not an image
token input). CFG doubles model positions but not logical sampled image tokens.
"""


class MeteredBackend:
    def __init__(self,backend,prefill_length):
        if prefill_length<1: raise ValueError('Positive prefill length required')
        self.backend=backend; self.prefill_length=prefill_length
        self.prefill_calls=0; self.image_input_positions=0; self.forward_calls=0
        self.ready_hits=0; self.sample_calls=0

    def predict(self,state):
        if getattr(state,'ready_prediction',None) is not None: self.ready_hits+=1
        else:
            self.forward_calls+=1
            if state.cache is None: self.prefill_calls+=1
            else: self.image_input_positions+=1
        return self.backend.predict(state)

    def sample(self,prediction):
        self.sample_calls+=1
        return self.backend.sample(prediction)

    def alternatives(self,*args): return self.backend.alternatives(*args)
    def append(self,*args): return self.backend.append(*args)
    def ready(self,*args): return self.backend.ready(*args)


def original_metrics(result,meter):
    length=len(result['tokens']); suffix=0; forced=0; selected=0; lookahead=0; reused=0
    events=[e for e in result['events'] if e['kind']=='backtrack']
    if len(events)!=result['backtracks']: raise ValueError('Backtrack event mismatch')
    for e in events:
        for c in e['candidates']:
            if not e['s']<c['position']<=e['f']: raise ValueError('Candidate position outside window')
            tail=e['f']-c['position']; suffix+=tail; forced+=1
            if c['id']==e['selected_id']: selected+=1+tail
        lookahead+=sum(c['lookahead_entropy'] is not None for c in e['survivors'])
        reused+=int(not e['terminal_mean_only'])
    if meter.sample_calls!=length+suffix or meter.ready_hits!=reused:
        raise ValueError('Sampling/ready accounting differs from controller events')
    if meter.forward_calls!=length+suffix+lookahead-reused or meter.prefill_calls!=1:
        raise ValueError('Model evaluation accounting differs from controller events')
    if meter.image_input_positions!=meter.forward_calls-1: raise ValueError('Unexpected input sequence shape')
    return make_metrics(length,meter.prefill_length,len(events),forced,suffix,selected,
        lookahead,meter.forward_calls,meter.image_input_positions,meter.sample_calls,meter.ready_hits,
        'instrumented backend; original controller events cross-checked')


def make_metrics(length,prefill,backtracks,forced,suffix,selected,lookahead,evaluations,image_inputs,samples,reused,origin):
    return dict(final_image_tokens=length,backtracks=backtracks,
        regeneration_tokens=forced+suffix,forced_alternative_tokens=forced,resampled_suffix_tokens=suffix,
        selected_candidate_regeneration_tokens=selected,
        logical_generated_image_tokens=length+forced+suffix,stochastic_sample_calls=samples,
        lookahead_prediction_requests=lookahead,cached_prediction_reuses=reused,
        image_probability_evaluations_per_branch=evaluations,
        model_input_image_token_positions_per_branch=image_inputs,
        prompt_prefill_token_positions_per_branch=prefill,cfg_branches=2,
        model_input_image_token_positions_with_cfg=2*image_inputs,
        model_input_token_positions_with_cfg=2*(prefill+image_inputs),count_origin=origin)


def backbone_reference(length,prefill):
    if length<1 or prefill<1: raise ValueError('Positive lengths required')
    return make_metrics(length,prefill,0,0,0,0,0,length,length-1,length,0,
        'analytical reference for fixed-length native autoregressive loop; no new baseline GPU run')


def summarize_metrics(records):
    if len(records)!=100 or len({r['sample_id'] for r in records})!=100:
        raise ValueError('Exactly 100 distinct completed images required')
    output={}
    for method in ['inchworm','backbone_reference']:
        values=[r['token_metrics'][method] for r in records]
        numeric=[k for k,v in values[0].items() if isinstance(v,(int,float))]
        output[method]=dict(images=100,totals={k:sum(v[k] for v in values) for k in numeric},
            means={k:sum(v[k] for v in values)/100 for k in numeric},count_origin=values[0]['count_origin'])
    return output

"""Independent Inchworm variant: guarded original, adaptive proposals, mixed sites."""
from dataclasses import asdict, dataclass, field
from math import isfinite

from inchworm_core import Config, trigger
from inchworm_variant_rules import TemperatureConfig, mixed_positions
from inchworm_variant_scoring import score_path, strictly_better

METHOD = 'inchworm_variant'


@dataclass(frozen=True)
class VariantConfig:
    controller: Config = field(default_factory=Config)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    improvement_margin: float = 0.0
    max_checks: int | None = None

    def validate(self):
        self.controller.validate()
        self.temperature.validate()
        if self.controller.positions != 4:
            raise ValueError('This confirmed variant uses M=4 (two H plus two G)')
        if self.controller.max_backtracks is not None:
            raise ValueError('Use max_checks for the variant, not max_backtracks')
        if not isfinite(self.improvement_margin) or self.improvement_margin < 0:
            raise ValueError('Improvement margin must be finite and nonnegative')
        if self.max_checks is not None and (not isinstance(self.max_checks,int) or self.max_checks<0):
            raise ValueError('max_checks must be a nonnegative integer or None')


def generate(backend, initial_state, length, cfg=VariantConfig(), on_event=None):
    """Backend adds gap(pred, token) and with-replacement temperature proposals.

    Keep N alternative finalists by window mean; original is an extra finalist.
    A rejected check still confirms the current window. Entropies and all scores
    use the base sampling distribution, never the sharpened proposal entropy.
    """
    cfg.validate()
    if not isinstance(length,int) or length<1:
        raise ValueError('length must be a positive integer')
    base=cfg.controller
    tokens,entropies,records,events=[],[],[],[]
    state=initial_state
    s=checks=accepted=retained=0

    def emit(event):
        events.append(event)
        if on_event: on_event(event)

    def predict(current):
        pred=backend.predict(current)
        if not isfinite(pred.entropy) or pred.entropy<0:
            raise ValueError('Invalid predictive entropy')
        return pred

    while len(tokens)<length:
        pred=predict(state)
        token=backend.sample(pred)
        tokens.append(token); entropies.append(pred.entropy)
        state=backend.append(pred,token)
        f=len(tokens)
        if f%96==0: emit({'kind':'progress','f':f})
        if f<=base.warmup:
            s=f
            if f==min(base.warmup,length): emit({'kind':'warmup','confirmed':f})
            continue
        if cfg.max_checks is not None and checks>=cfg.max_checks:
            records.clear(); s=f
            continue
        records.append(pred)
        should_check,diagnostics=trigger(entropies,s,base)
        if not should_check:
            if f-s>=base.max_window or f==length:
                emit({'kind':'confirm','s':s,'f':f,**diagnostics})
                s=f; records.clear()
            continue

        checks+=1
        original_tokens=tokens[s:f].copy()
        original_entropies=entropies[s:f].copy()
        original_state=state
        gaps=[backend.gap(records[j-s],tokens[j]) for j in range(s,f)]
        offsets=mixed_positions(original_entropies,gaps)
        finalists=[]; summaries=[]; proposals=[]
        candidate_id=0
        for offset in offsets:
            i=s+offset
            original_pred=records[offset]
            draws,proposal=backend.alternatives(original_pred,tokens[i],base.alternatives)
            if len(draws)!=base.alternatives:
                raise ValueError('Expected exactly k draws, including duplicates')
            proposals.append({'position':i+1,'gap':gaps[offset],**proposal})
            for draw_index,alternative in enumerate(draws):
                candidate_tokens=tokens[s:i]+[alternative]
                candidate_entropies=entropies[s:i+1].copy()
                candidate_state=backend.append(original_pred,alternative)
                for _ in range(i+1,f):
                    p=predict(candidate_state)
                    z=backend.sample(p)  # Base sampling: no adaptive suffix temperature.
                    candidate_tokens.append(z); candidate_entropies.append(p.entropy)
                    candidate_state=backend.append(p,z)
                mean=sum(candidate_entropies)/(f-s)
                candidate={'id':candidate_id,'position':i+1,'draw_index':draw_index,
                    'alternative':alternative,'mean_entropy':mean,'tokens':candidate_tokens,
                    'entropies':candidate_entropies,'state':candidate_state}
                summaries.append({k:candidate[k] for k in ['id','position','draw_index','alternative','mean_entropy']})
                finalists.append(candidate)
                finalists.sort(key=lambda c:(c['mean_entropy'],c['id']))
                del finalists[base.survivors:]
                candidate_id+=1
        if not finalists: raise RuntimeError('No candidate paths')
        terminal=f==length
        # Original path is never subjected to the mean-only N-candidate filter.
        original=score_path(backend,original_state,original_entropies,terminal,base.mean_weight)
        for candidate in finalists:
            candidate.update(score_path(backend,candidate['state'],candidate['entropies'],terminal,base.mean_weight))
        winner=min(finalists,key=lambda c:(c['score'],c['id']))
        changed=strictly_better(winner['score'],original['score'],cfg.improvement_margin)
        if changed:
            tokens[s:f]=winner['tokens']; entropies[s:f]=winner['entropies']
            state=winner['state']; accepted+=1
        else:
            # Restore original cache/pending token, or its ready lookahead cache.
            state=original['state']; retained+=1
        emit({'kind':'check','s':s,'f':f,**diagnostics,'accepted':changed,
            'original_tokens':original_tokens,'original_entropies':original_entropies,
            'original_score':original['score'],'original_mean_entropy':original['mean_entropy'],
            'original_lookahead_entropy':original['lookahead_entropy'],
            'selected_tokens':tokens[s:f].copy(),'selected_id':winner['id'] if changed else 'original',
            'selected_score':winner['score'] if changed else original['score'],
            'best_candidate_score':winner['score'],'improvement_margin':cfg.improvement_margin,
            'positions':[s+j+1 for j in offsets],'proposals':proposals,'candidates':summaries,
            'survivors':[{k:c[k] for k in ['id','mean_entropy','lookahead_entropy','score']} for c in finalists],
            'terminal_mean_only':terminal})
        s=f; records.clear(); finalists.clear()
        del original,original_state,winner,candidate,candidate_state,original_pred

    assert checks==accepted+retained
    return {'method':METHOD,'tokens':tokens,'entropies':entropies,'events':events,
        'checks':checks,'accepted_edits':accepted,'kept_original':retained,
        'backtracks':accepted,'backtracks_definition':'accepted window replacements',
        'config':asdict(cfg),'search_cap_reached':cfg.max_checks is not None and checks>=cfg.max_checks}

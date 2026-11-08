"""The agreed entropy score and strict original-path safeguard."""
from math import isfinite


def strictly_better(candidate_score,original_score,margin=0.0):
    if not all(isfinite(x) for x in [candidate_score,original_score,margin]) or margin<0:
        raise ValueError('Scores must be finite and margin nonnegative')
    return candidate_score < original_score-margin


def score_path(backend,state,entropies,terminal,mean_weight=0.5):
    """Score both original and replacement paths using the same distribution.

    Proposal temperature never replaces the base distribution's entropy in the
    objective. End-of-image scoring follows the original mean-only convention.
    """
    entropies=list(entropies)
    if not entropies or any(not isfinite(h) or h<0 for h in entropies):
        raise ValueError('Need finite nonnegative window entropies')
    if not isfinite(mean_weight) or not 0<=mean_weight<=1:
        raise ValueError('Invalid mean weight')
    mean=sum(entropies)/len(entropies)
    if terminal:
        return {'mean_entropy':mean,'lookahead_entropy':None,'score':mean,'state':state}
    lookahead=backend.predict(state)
    if not isfinite(lookahead.entropy) or lookahead.entropy<0:
        raise ValueError('Invalid lookahead entropy')
    return {'mean_entropy':mean,'lookahead_entropy':lookahead.entropy,
            'score':mean_weight*mean+(1-mean_weight)*lookahead.entropy,
            'state':backend.ready(lookahead)}

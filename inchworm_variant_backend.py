"""Wrap existing model backends without changing their original sampling path."""
from math import isfinite

from inchworm_variant_rules import TemperatureConfig, adaptive_temperature


class VariantBackend:
    def __init__(self,base,temperature=TemperatureConfig()):
        temperature.validate()
        self.base=base
        self.temperature=temperature

    def predict(self,state): return self.base.predict(state)
    def sample(self,prediction): return self.base.sample(prediction)
    def append(self,prediction,token): return self.base.append(prediction,token)
    def ready(self,prediction): return self.base.ready(prediction)

    def gap(self,prediction,token):
        p=prediction.probabilities[0].double()
        chosen=float(p[token].item())
        if chosen<=0 or not isfinite(chosen):
            raise ValueError('Original token must have positive base sampling probability')
        return float((p.max().log()-p[token].log()).item())

    def alternatives(self,prediction,original,k):
        import torch
        temperature=adaptive_temperature(prediction.entropy,self.temperature)
        # FP64 preserves small support probabilities when sharpening. Neither
        # remove the original token nor truncate/deduplicate repeated draws.
        p=prediction.probabilities[0].double()
        if not torch.isfinite(p).all() or (p<0).any() or p.sum()<=0:
            raise ValueError('Invalid base sampling probabilities')
        q=torch.softmax(p.log()/temperature,dim=-1)
        draws=torch.multinomial(q,k,replacement=True).tolist()
        return draws,{'temperature':temperature,'base_entropy':prediction.entropy,
                      'original_token':original,'sampled_tokens':draws,
                      'with_replacement':True,'original_token_allowed':True}

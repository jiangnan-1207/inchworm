"""Original Inchworm adapter for Lumina's fixed-header, CFG=4 image sampling."""
from dataclasses import dataclass
import torch
from lumina_cache import ForkableCache
from lumina_layout import CODE_START,CODE_COUNT


@dataclass(frozen=True)
class State:
    conditional:tuple=()
    unconditional:tuple=()
    pending:tuple=()
    count:int=0
    ready_prediction:object=None


@dataclass
class Prediction:
    probabilities:object
    entropy:float
    conditional:tuple
    unconditional:tuple
    count:int


class LuminaBackend:
    def __init__(self,model,prompt,header,layout):
        self.model=model; self.prompt=prompt; self.header=header; self.layout=layout
        self.calls=0

    def _forward(self,inputs,snapshot):
        cache=ForkableCache(snapshot)
        output=self.model(input_ids=inputs,past_key_values=cache,use_cache=True,return_dict=True)
        self.calls+=1
        return output.logits[:,-1,:],cache.snapshot()

    @torch.inference_mode()
    def predict(self,state):
        if state.ready_prediction is not None: return state.ready_prediction
        if not state.pending:
            conditional=self.prompt
            unconditional=torch.tensor([self.header],device=self.prompt.device,dtype=torch.long)
        else:
            conditional=unconditional=torch.tensor([state.pending],device=self.prompt.device,dtype=torch.long)
        c,cs=self._forward(conditional,state.conditional)
        u,us=self._forward(unconditional,state.unconditional)
        logits=u+4.0*(c-u)
        logits[:,:CODE_START]=-torch.inf; logits[:,CODE_START+CODE_COUNT:]=-torch.inf
        threshold=torch.topk(logits,4096).values[:,-1,None]
        logits=logits.masked_fill(logits<threshold,-torch.inf)
        probabilities=torch.softmax(logits,dim=-1)
        p=probabilities.float(); p=p/p.sum(-1,keepdim=True)
        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum().item()
        return Prediction(probabilities,entropy,cs,us,state.count)

    def sample(self,prediction):
        return int(torch.multinomial(prediction.probabilities,1).item())

    def alternatives(self,prediction,original,k):
        p=prediction.probabilities[0].clone(); p[original]=-torch.inf
        return torch.topk(p,k).indices.tolist()

    def append(self,prediction,token):
        count=prediction.count+1
        return State(prediction.conditional,prediction.unconditional,self.layout.pending(token,count),count)

    def ready(self,prediction):
        return State(count=prediction.count,ready_prediction=prediction)

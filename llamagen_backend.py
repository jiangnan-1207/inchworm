"""Functional prefix caches for the pinned upstream LlamaGen GPT.

Adapted from FoundationVision/LlamaGen (MIT; licenses/LlamaGen.txt).
Branch states never share mutable upstream KVCache buffers.
"""
from dataclasses import dataclass
import torch
from torch import nn


class PrefixCache(nn.Module):
    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix

    def update(self, positions, key, value):
        offset = 0 if self.prefix is None else self.prefix[0].shape[2]
        expected = torch.arange(offset, offset+key.shape[2], device=positions.device)
        if not torch.equal(positions, expected):
            raise ValueError('Non-contiguous LlamaGen cache update')
        if self.prefix is not None:
            key = torch.cat([self.prefix[0], key], dim=2)
            value = torch.cat([self.prefix[1], value], dim=2)
        self.prefix = (key, value)
        return key, value


@dataclass
class State:
    cache: object = None
    pending: int | None = None
    ready_prediction: object = None


@dataclass
class Prediction:
    probabilities: object
    entropy: float
    cache: object


class LlamaGenBackend:
    def __init__(self, model, cond, masks, length=1024):
        self.model = model
        self.cond = torch.cat([cond, torch.zeros_like(cond)+model.cls_embedding.uncond_embedding])
        self.text_length = cond.shape[1]
        size = self.text_length+length
        self.mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=cond.device))[None].repeat(2,1,1)
        self.mask[:,:,:self.text_length] &= torch.cat([masks,masks]).bool().unsqueeze(1)
        diagonal = torch.arange(size,device=cond.device)
        self.mask[:,diagonal,diagonal] = True
        model.freqs_cis = model.freqs_cis.to(cond.device)
        self.forward_calls = 0
        from autoregressive.models.generate import logits_to_probs
        self.logits_to_probs = logits_to_probs

    @torch.inference_mode()
    def predict(self, state):
        if state.ready_prediction is not None:
            return state.ready_prediction
        prefix = state.cache
        offset = 0 if prefix is None else prefix[0][0].shape[2]
        steps = self.text_length if prefix is None else 1
        positions = torch.arange(offset, offset+steps, device=self.cond.device)
        for i,layer in enumerate(self.model.layers):
            layer.attention.kv_cache = PrefixCache(None if prefix is None else prefix[i])
        # Crop only invisible trailing key positions. Preserve all mask rows
        # because upstream indexes rows by absolute input_pos.
        self.model.causal_mask = self.mask[:,:,:offset+steps]
        tokens = None if prefix is None else torch.tensor([[state.pending],[state.pending]],device=self.cond.device)
        # Upstream uses math SDPA for image-token decoding.
        if prefix is None:
            logits,_ = self.model(tokens,self.cond,positions)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
                logits,_ = self.model(tokens,None,positions)
        self.forward_calls += 1
        guided = logits[1:2,-1,:]+7.5*(logits[0:1,-1,:]-logits[1:2,-1,:])
        probabilities = self.logits_to_probs(guided,temperature=1.0,top_k=1000,top_p=1.0)
        p = probabilities.float(); p = p/p.sum(-1,keepdim=True)
        entropy = -(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum().item()
        cache = tuple(layer.attention.kv_cache.prefix for layer in self.model.layers)
        return Prediction(probabilities,entropy,cache)

    def sample(self,prediction):
        return int(torch.multinomial(prediction.probabilities,1).item())

    def alternatives(self,prediction,original,k):
        scores=prediction.probabilities[0].clone(); scores[original]=-torch.inf
        return torch.topk(scores,k).indices.tolist()

    def append(self,prediction,token):
        return State(cache=prediction.cache,pending=int(token))

    def ready(self,prediction):
        return State(ready_prediction=prediction)

    def close(self):
        for layer in self.model.layers:
            layer.attention.kv_cache = None
        self.model.causal_mask = None


@torch.inference_mode()
def check_native_prefixes(model,cond,masks,length=32):
    """Compare probabilities at identical prefixes, including a restored branch.

Native full-capacity and functional cropped-cache attention have different
reduction shapes, so compare numerically, not by assuming bitwise identity.
"""
    from autoregressive.models.generate import logits_to_probs
    combined=torch.cat([cond,torch.zeros_like(cond)+model.cls_embedding.uncond_embedding])
    text=cond.shape[1]
    with torch.device(cond.device):
        model.setup_caches(2,text+1024,model.tok_embeddings.weight.dtype)
    model.causal_mask[:,:,:text] &= torch.cat([masks,masks]).bool().unsqueeze(1)
    diagonal=torch.arange(model.max_seq_length,device=cond.device)
    model.causal_mask[:,diagonal,diagonal]=True
    expected=[]; tokens=[]
    for t in range(length):
        position=torch.arange(text,device=cond.device) if t==0 else torch.tensor([text+t-1],device=cond.device)
        x=None if t==0 else torch.tensor([[tokens[-1]],[tokens[-1]]],device=cond.device)
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
            logits,_=model(x,combined if t==0 else None,position)
        guided=logits[1:2,-1,:]+7.5*(logits[0:1,-1,:]-logits[1:2,-1,:])
        p=logits_to_probs(guided,temperature=1.0,top_k=1000,top_p=1.0)
        expected.append(p.clone()); tokens.append(int(p.argmax().item()))
    backend=LlamaGenBackend(model,cond,masks)
    state=State(); saved=None
    for t in range(length):
        pred=backend.predict(state)
        # Total variation bounds distribution changes even at top-k boundaries.
        tv=(pred.probabilities-expected[t]).abs().sum().item()/2
        if tv>0.01:
            raise AssertionError(f'LLAMAGEN_CACHE_PARITY_FAILED t={t} total_variation={tv}')
        if t==8: saved=state
        state=backend.append(pred,tokens[t])
    restored=backend.predict(saved)
    if (restored.probabilities-expected[8]).abs().sum().item()/2>0.01:
        raise AssertionError('LlamaGen cache restoration failed')
    backend.close()
    print('LLAMAGEN_NATIVE_PREFIX_AND_RESTORE_CHECK_OK',flush=True)

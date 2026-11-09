"""Forkable Transformers cache with shared immutable 128-token chunks.

Only the final partial chunk is copied on append. Dense attention keys/values
are materialized transiently per layer, not retained in every branch snapshot.
"""
import torch
from transformers.cache_utils import Cache


class ForkableCache(Cache):
    def __init__(self,snapshot=(),chunk_size=128):
        super().__init__()
        if chunk_size<1: raise ValueError('chunk_size must be positive')
        self.layers=list(snapshot)
        self.chunk_size=chunk_size

    def snapshot(self): return tuple(self.layers)
    def __len__(self): return len(self.layers)
    def get_seq_length(self,layer_idx=0):
        return sum(t.shape[-2] for t in self.layers[layer_idx][0]) if layer_idx<len(self.layers) else 0
    def get_max_cache_shape(self): return None
    def get_max_length(self): return None

    def _append(self,chunks,new):
        result=list(chunks)
        at=0
        if result and result[-1].shape[-2]<self.chunk_size:
            take=min(self.chunk_size-result[-1].shape[-2],new.shape[-2])
            result[-1]=torch.cat([result[-1],new[...,:take,:]],dim=-2)
            at=take
        while at<new.shape[-2]:
            end=min(at+self.chunk_size,new.shape[-2])
            result.append(new[...,at:end,:].clone())
            at=end
        return tuple(result)

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        offset=self.get_seq_length(layer_idx)
        if layer_idx==0 and cache_kwargs and cache_kwargs.get('cache_position') is not None:
            pos=cache_kwargs['cache_position']
            if not torch.equal(pos,torch.arange(offset,offset+key_states.shape[-2],device=pos.device)):
                raise ValueError('Non-contiguous cache update')
        while len(self.layers)<=layer_idx: self.layers.append(((),()))
        old_k,old_v=self.layers[layer_idx]
        keys=self._append(old_k,key_states); values=self._append(old_v,value_states)
        self.layers[layer_idx]=(keys,values)
        return (keys[0] if len(keys)==1 else torch.cat(keys,dim=-2),
                values[0] if len(values)==1 else torch.cat(values,dim=-2))

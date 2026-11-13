import importlib.util
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import sys
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is installed on compute nodes')
class PrefixCacheTests(unittest.TestCase):
    def test_native_capacity_padding_preserves_branches_and_rejects_overflow(self):
        import torch
        from llamagen_backend import PrefixCache
        key=torch.arange(24).reshape(2,1,3,4).to(torch.bfloat16)
        value=key+1
        saved=(key.clone(),value.clone())
        left=PrefixCache(saved,capacity=8); right=PrefixCache(saved,capacity=8)
        lk,lv=left.update(torch.tensor([3]),torch.zeros(2,1,1,4,dtype=key.dtype),
            torch.ones(2,1,1,4,dtype=key.dtype))
        rk,rv=right.update(torch.tensor([3]),torch.ones(2,1,1,4,dtype=key.dtype)*9,
            torch.ones(2,1,1,4,dtype=key.dtype)*10)
        self.assertEqual(lk.shape,(2,1,8,4))
        self.assertEqual(left.prefix[0].shape,(2,1,4,4))
        self.assertTrue(torch.equal(lk[:,:,:3],key))
        self.assertTrue(torch.equal(lv[:,:,:3],value))
        self.assertEqual(lk[:,:,4:].abs().sum().item(),0)
        self.assertEqual(lv[:,:,4:].abs().sum().item(),0)
        self.assertTrue(torch.equal(saved[0],key))
        self.assertEqual(lk[:,:,3].sum().item(),0)
        self.assertEqual(rk[:,:,3].sum().item(),72)
        self.assertEqual(rv[:,:,3].sum().item(),80)
        with self.assertRaises(ValueError):
            PrefixCache(saved,capacity=3).update(torch.tensor([3]),key[:,:,:1],value[:,:,:1])

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'benchmarks/LlamaGen/autoregressive/models/gpt.py').is_file(),
        'Requires the pinned upstream source checkout')
    def test_real_native_small_gpt_prefill_decode_and_restore_in_both_dtypes(self):
        import torch
        from llamagen_backend import check_native_prefixes
        source=Path(__file__).resolve().parents[1]/'benchmarks/LlamaGen'
        with patch.object(sys,'path',[str(source),*sys.path]):
            from autoregressive.models.gpt import ModelArgs,Transformer
            for dtype in [torch.float32,torch.bfloat16]:
                with self.subTest(dtype=dtype),torch.inference_mode():
                    torch.manual_seed(42)
                    model=Transformer(ModelArgs(dim=32,n_layer=2,n_head=4,multiple_of=16,
                        caption_dim=8,cls_token_num=3,block_size=16,vocab_size=32,model_type='t2i'))
                    # Native initialization zeros the head; a nonzero head is
                    # essential to catch cache/attention errors in this test.
                    torch.nn.init.normal_(model.output.weight,std=0.1)
                    model=model.to(dtype=dtype).eval()
                    cond=torch.randn(1,3,8,dtype=dtype)
                    masks=torch.tensor([[0,1,1]])
                    check_native_prefixes(model,cond,masks,length=10)
                    self.assertTrue(all(layer.attention.kv_cache is None for layer in model.layers))

    def test_native_prefill_gate_uses_the_same_kernel_policy_as_generation(self):
        """Regression: a math-only reference prefill falsely rejected the adapter."""
        import torch
        from llamagen_backend import check_native_prefixes,PrefixCache
        active={'math':False}; trace=[]

        @contextmanager
        def kernel(**kwargs):
            previous=active['math']; active['math']=True
            try: yield
            finally: active['math']=previous

        class Model:
            def __init__(self):
                self.layers=[SimpleNamespace(attention=SimpleNamespace(kv_cache=None))]
                self.tok_embeddings=torch.nn.Embedding(3,2)
                self.cls_embedding=SimpleNamespace(uncond_embedding=torch.zeros(3,2))
                self.freqs_cis=torch.zeros(3)

            def setup_caches(self,batch,size,dtype):
                self.max_seq_length=size
                self.causal_mask=torch.ones(batch,size,size,dtype=torch.bool)
                self.layers[0].attention.kv_cache=PrefixCache()

            def __call__(self,tokens,condition,positions):
                trace.append(('prefill' if condition is not None else 'decode',active['math']))
                count=len(positions)
                self.layers[0].attention.kv_cache.update(positions,
                    torch.zeros(2,1,count,2),torch.zeros(2,1,count,2))
                logits=torch.zeros(2,count,3)
                # Simulate a backend-dependent prefill difference large enough
                # to fail the original TV gate. Decode stays deterministic.
                logits[0,:,0]=0.5 if condition is not None and active['math'] else 0.1
                return logits,None

        generate=SimpleNamespace(logits_to_probs=lambda logits,**kwargs:logits.softmax(-1))
        with patch.dict(sys.modules,{'autoregressive.models.generate':generate}), \
                patch.object(torch.backends.cuda,'sdp_kernel',kernel):
            check_native_prefixes(Model(),torch.ones(1,3,2),torch.ones(1,3),length=10)
        prefill=[math for kind,math in trace if kind=='prefill']
        self.assertEqual(prefill,[False,False])
        self.assertTrue(all(math for kind,math in trace if kind=='decode'))

    def test_restored_branches_do_not_mutate_their_shared_prefix(self):
        import torch
        from llamagen_backend import PrefixCache
        key=torch.ones(2,2,3,4); value=key*2
        prefix=(key.clone(),value.clone())
        left=PrefixCache(prefix); right=PrefixCache(prefix)
        left.update(torch.tensor([3]),torch.zeros(2,2,1,4),torch.zeros(2,2,1,4))
        right.update(torch.tensor([3]),torch.ones(2,2,1,4)*9,torch.ones(2,2,1,4)*9)
        self.assertTrue(torch.equal(prefix[0],key))
        self.assertEqual(left.prefix[0].shape[2],4)
        self.assertEqual(right.prefix[0][0,0,3,0].item(),9)
        self.assertEqual(left.prefix[0][0,0,3,0].item(),0)
        with self.assertRaises(ValueError):
            right.update(torch.tensor([7]),torch.ones(2,2,1,4),torch.ones(2,2,1,4))

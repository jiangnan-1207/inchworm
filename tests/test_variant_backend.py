import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires the compute-node PyTorch environment')
class VariantBackendTests(unittest.TestCase):
    def test_point_mass_can_draw_original_repeatedly_without_deduplication(self):
        import torch
        from inchworm_variant_backend import VariantBackend
        base=SimpleNamespace()
        backend=VariantBackend(base)
        pred=SimpleNamespace(probabilities=torch.tensor([[1.,0.,0.]]),entropy=0.)
        draws,record=backend.alternatives(pred,0,8)
        self.assertEqual(draws,[0]*8)
        self.assertTrue(record['with_replacement'])
        self.assertEqual(backend.gap(pred,0),0)
        self.assertTrue(torch.equal(pred.probabilities,torch.tensor([[1.,0.,0.]])))

    def test_gap_uses_untempered_base_and_forward_sampling_is_delegated(self):
        import torch
        from inchworm_variant_backend import VariantBackend
        from math import log
        pred=SimpleNamespace(probabilities=torch.tensor([[.6,.3,.1]],dtype=torch.float64),entropy=1.)
        calls=[]
        base=SimpleNamespace(sample=lambda p:calls.append(p) or 2)
        backend=VariantBackend(base)
        self.assertAlmostEqual(backend.gap(pred,1),log(2))
        self.assertEqual(backend.sample(pred),2)
        self.assertEqual(calls,[pred])

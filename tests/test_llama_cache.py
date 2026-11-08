import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is installed on compute nodes')
class PrefixCacheTests(unittest.TestCase):
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

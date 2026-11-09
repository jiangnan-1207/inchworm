import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
    'Requires Lumina Torch/Transformers environment')
class LuminaCacheTests(unittest.TestCase):
    def test_forks_share_full_prefix_blocks_but_never_mutate_original(self):
        import torch
        from lumina_cache import ForkableCache
        cache=ForkableCache(chunk_size=2)
        key=torch.arange(12).reshape(1,1,3,4).float()
        cache.update(key,key+1,0,{'cache_position':torch.arange(3)})
        snapshot=cache.snapshot()
        left=ForkableCache(snapshot,chunk_size=2); right=ForkableCache(snapshot,chunk_size=2)
        a,_=left.update(torch.zeros(1,1,1,4),torch.zeros(1,1,1,4),0)
        b,_=right.update(torch.ones(1,1,1,4)*99,torch.ones(1,1,1,4)*99,0)
        self.assertTrue(torch.equal(torch.cat(snapshot[0][0],-2),key))
        self.assertIs(left.layers[0][0][0],snapshot[0][0][0])
        self.assertIs(right.layers[0][0][0],snapshot[0][0][0])
        self.assertEqual(a[0,0,-1,0],0)
        self.assertEqual(b[0,0,-1,0],99)
        self.assertEqual(cache.get_seq_length(),3)
        self.assertEqual(left.get_seq_length(),4)

    def test_chunked_values_match_dense_append_across_multiple_layers(self):
        import torch
        from lumina_cache import ForkableCache
        cache=ForkableCache(chunk_size=3)
        expected=[None,None]
        for length in [2,1,4,2]:
            for layer in range(2):
                update=torch.randn(1,2,length,4)
                before=cache.get_seq_length(layer)
                dense,_=cache.update(update,update*2,layer,{'cache_position':torch.arange(before,before+length)})
                expected[layer]=update if expected[layer] is None else torch.cat([expected[layer],update],-2)
                torch.testing.assert_close(dense,expected[layer])
        self.assertEqual(cache.get_seq_length(),9)
        self.assertIsNone(cache.get_max_cache_shape())
        with self.assertRaises(ValueError):
            cache.update(torch.ones(1,2,1,4),torch.ones(1,2,1,4),0,{'cache_position':torch.tensor([0])})

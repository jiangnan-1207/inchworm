import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('llama_runtime',ROOT/'tools/diagnose_llamagen_runtime.py')
runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires PyTorch')
class RuntimeProbeTests(unittest.TestCase):
    def test_teacher_forced_alignment_starts_at_last_text_position(self):
        import torch
        structure_spec=importlib.util.spec_from_file_location('llama_structure',ROOT/'tools/check_llamagen_structure.py')
        structure=importlib.util.module_from_spec(structure_spec); structure_spec.loader.exec_module(structure)
        logits=torch.arange(6).reshape(1,6,1)
        selected=structure.image_prediction_logits(logits,text_length=3,image_length=4)
        self.assertEqual(selected.flatten().tolist(),[2,3,4,5])
        with self.assertRaises(ValueError): structure.image_prediction_logits(logits,3,5)

    def test_official_left_padding_keeps_valid_token_order(self):
        import torch
        features=torch.arange(2*5*3).reshape(2,5,3).float()
        masks=torch.tensor([[1,1,0,0,0],[1,1,1,0,0]])
        actual,actual_masks=runtime.prepare_condition(features,masks)
        self.assertEqual(actual_masks.tolist(),[[0,0,0,1,1],[0,0,1,1,1]])
        self.assertTrue(torch.equal(actual[0,3:],features[0,:2]))
        self.assertTrue(torch.equal(actual[1,2:],features[1,:3]))
        self.assertEqual(actual[0,:3].abs().sum().item(),0)

    def test_feature_stats_exclude_padding_and_detect_nonfinite(self):
        import torch
        features=torch.tensor([[[1.,0.],[999.,999.]],[[0.,1.],[999.,999.]]])
        masks=torch.tensor([[1,0],[1,0]])
        stats=runtime.feature_stats(features,masks)
        self.assertTrue(stats['finite'])
        self.assertEqual(stats['valid_abs_max'],1)
        self.assertEqual(stats['pooled_prompt_cosines'],[[1,0],[0,1]])
        features[0,0,0]=float('nan')
        self.assertFalse(runtime.feature_stats(features,masks)['finite'])


if __name__=='__main__': unittest.main()

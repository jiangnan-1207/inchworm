from types import SimpleNamespace
import unittest

from inchworm_variant_scoring import score_path, strictly_better


class Backend:
    def __init__(self): self.calls=[]
    def predict(self,state):
        self.calls.append(state)
        return SimpleNamespace(entropy=2,state=state)
    def ready(self,prediction): return ('ready',prediction.state)


class VariantScoringTests(unittest.TestCase):
    def test_strict_improvement_and_margin_boundary(self):
        self.assertFalse(strictly_better(1,1))
        self.assertFalse(strictly_better(2,1))
        self.assertTrue(strictly_better(0.99,1))
        self.assertFalse(strictly_better(0.75,1,0.25))
        self.assertTrue(strictly_better(0.74,1,0.25))

    def test_same_score_formula_and_ready_state_for_original_or_candidate(self):
        backend=Backend()
        result=score_path(backend,'original',[1,3,5],False)
        self.assertEqual(result['score'],2.5)
        self.assertEqual(result['state'],('ready','original'))
        result=score_path(backend,'replacement',[1,3,5],False)
        self.assertEqual(result['score'],2.5)
        self.assertEqual(result['state'],('ready','replacement'))

    def test_terminal_window_never_looks_past_image(self):
        backend=Backend()
        result=score_path(backend,'original',[1,3],True)
        self.assertEqual(result['score'],2)
        self.assertEqual(result['state'],'original')
        self.assertIsNone(result['lookahead_entropy'])
        self.assertEqual(backend.calls,[])

    def test_nonfinite_scores_cannot_pass_safeguard(self):
        for values in [(float('nan'),1,0),(0,float('inf'),0),(0,1,-1)]:
            with self.assertRaises(ValueError): strictly_better(*values)

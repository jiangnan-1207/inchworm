from dataclasses import replace
import unittest

from inchworm_core import Config
from inchworm_variant import VariantConfig, generate
from test_core import FakeBackend, Pred


class Backend(FakeBackend):
    draws=(1,2)
    def gap(self,pred,token): return float(len(pred.prefix)%3)
    def alternatives(self,pred,original,k):
        tokens=list(self.draws[:k])
        return tokens,{'sampled_tokens':tokens,'original_token':original}


class WorseBackend(Backend):
    draws=(1,1)
    def predict(self,state):
        if isinstance(state,Pred): return state
        pred=super().predict(state)
        return Pred(pred.prefix,10.0 if 1 in state else pred.entropy)


class SameBackend(Backend): draws=(0,0)


class VariantCoreTests(unittest.TestCase):
    def cfg(self,**kwargs):
        return VariantConfig(controller=Config(warmup=4,max_window=3),**kwargs)

    def test_worse_candidates_keep_original_cache_and_continue_original_prefix(self):
        backend=WorseBackend()
        result=generate(backend,(),12,self.cfg(max_checks=1))
        self.assertEqual(result['tokens'],[0]*12)
        self.assertEqual((result['checks'],result['accepted_edits'],result['kept_original']),(1,0,1))
        self.assertEqual(list(backend.calls[-1]),[0]*11)
        event=next(e for e in result['events'] if e['kind']=='check')
        self.assertEqual(event['selected_id'],'original')
        self.assertLess(event['original_score'],event['best_candidate_score'])

    def test_equal_candidates_keep_original_and_duplicate_draws_are_not_removed(self):
        result=generate(SameBackend(),(),12,self.cfg(max_checks=1))
        event=next(e for e in result['events'] if e['kind']=='check')
        self.assertFalse(event['accepted'])
        self.assertEqual(event['best_candidate_score'],event['original_score'])
        self.assertEqual(len(event['candidates']),6)  # Three window sites * two draws.
        self.assertTrue(all(c['alternative']==0 for c in event['candidates']))
        self.assertEqual(len(event['survivors']),2)
        self.assertEqual(len(result['tokens']),12)

    def test_better_candidate_replaces_history_and_continues_winning_cache(self):
        backend=Backend()
        result=generate(backend,(),12,self.cfg(max_checks=1))
        self.assertEqual(result['accepted_edits'],1)
        self.assertEqual(result['kept_original'],0)
        event=next(e for e in result['events'] if e['kind']=='check')
        self.assertLess(event['selected_score'],event['original_score'])
        self.assertEqual(list(backend.calls[-1]),result['tokens'][:-1])
        self.assertTrue(any(h<0.8 for h in result['entropies'][4:7]))

    def test_terminal_window_has_no_out_of_image_prediction_for_any_path(self):
        backend=Backend()
        result=generate(backend,(),7,self.cfg())
        event=next(e for e in result['events'] if e['kind']=='check')
        self.assertTrue(event['terminal_mean_only'])
        self.assertIsNone(event['original_lookahead_entropy'])
        self.assertTrue(all(c['lookahead_entropy'] is None for c in event['survivors']))
        self.assertLess(max(map(len,backend.calls)),7)

    def test_disabled_checks_are_forward_only_and_short_warmup_is_safe(self):
        result=generate(Backend(),(),12,self.cfg(max_checks=0))
        self.assertEqual(result['tokens'],[0]*12)
        self.assertEqual(result['checks'],0)
        self.assertEqual(generate(Backend(),(),3,self.cfg())['checks'],0)

    def test_margin_can_prevent_otherwise_accepted_replacement(self):
        result=generate(Backend(),(),12,self.cfg(max_checks=1,improvement_margin=100))
        self.assertEqual(result['kept_original'],1)

    def test_invalid_config_does_not_silently_change_agreed_M(self):
        for cfg in [VariantConfig(controller=Config(positions=6)),
                    VariantConfig(controller=Config(max_backtracks=0)),
                    VariantConfig(improvement_margin=float('nan')),
                    VariantConfig(max_checks=-1)]:
            with self.assertRaises(ValueError): cfg.validate()

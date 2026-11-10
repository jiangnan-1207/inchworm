"""Check counters against independent backend traces, without Torch or GPUs."""
from dataclasses import dataclass
import random
import unittest

from inchworm_core import Config, generate
from token_accounting import MeteredBackend, backbone_reference, original_metrics, summarize_metrics


@dataclass
class State:
    prefix: tuple=()
    cache: object=None
    ready_prediction: object=None


@dataclass
class Prediction:
    prefix: tuple
    entropy: float


class Backend:
    def __init__(self):
        self.forwards=[]
        self.samples=[]
        self.rng=random.Random(42)

    def predict(self,state):
        if state.ready_prediction is not None: return state.ready_prediction
        self.forwards.append(state.prefix)
        t=len(state.prefix)
        h=1.0 if t<4 else (0.8 if t<6 else 4.0)
        if 1 in state.prefix: h=0.3
        elif 2 in state.prefix: h=0.4
        return Prediction(state.prefix,h)

    def sample(self,prediction):
        # Advance a real RNG while keeping the test's entropy/branch pattern fixed.
        self.samples.append(self.rng.random())
        return 0

    def alternatives(self,prediction,original,k): return [1,2][:k]
    def append(self,prediction,token): return State(prediction.prefix+(token,),cache=True)
    def ready(self,prediction): return State(ready_prediction=prediction)


class TokenAccountingTests(unittest.TestCase):
    def run_case(self,length,config):
        original=Backend()
        expected=generate(original,State(),length,config)
        actual=Backend()
        meter=MeteredBackend(actual,19)
        result=generate(meter,State(),length,config)
        self.assertEqual(result,expected)
        self.assertEqual(actual.forwards,original.forwards)
        self.assertEqual(actual.samples,original.samples)
        metrics=original_metrics(result,meter)
        self.assertEqual(metrics['image_probability_evaluations_per_branch'],len(actual.forwards))
        self.assertEqual(metrics['stochastic_sample_calls'],len(actual.samples))
        return metrics

    def test_losing_candidates_lookahead_reuse_and_unchanged_rng(self):
        m=self.run_case(12,Config(warmup=4,max_window=3,max_backtracks=1))
        # Three positions x two alternatives: suffix sizes 2,2,1,1,0,0.
        self.assertEqual(m['forced_alternative_tokens'],6)
        self.assertEqual(m['resampled_suffix_tokens'],6)
        self.assertEqual(m['regeneration_tokens'],12)
        self.assertEqual(m['selected_candidate_regeneration_tokens'],3)
        self.assertEqual(m['logical_generated_image_tokens'],24)
        self.assertEqual(m['lookahead_prediction_requests'],2)
        self.assertEqual(m['cached_prediction_reuses'],1)
        self.assertEqual(m['image_probability_evaluations_per_branch'],19)
        self.assertEqual(m['model_input_image_token_positions_per_branch'],18)
        self.assertEqual(m['model_input_token_positions_with_cfg'],2*(19+18))

    def test_terminal_window_has_no_extra_prediction(self):
        m=self.run_case(7,Config(warmup=4,max_window=3))
        self.assertEqual(m['lookahead_prediction_requests'],0)
        self.assertEqual(m['cached_prediction_reuses'],0)
        self.assertEqual(m['image_probability_evaluations_per_branch'],13)

    def test_disabled_controller_matches_analytical_reference(self):
        m=self.run_case(12,Config(warmup=4,max_backtracks=0))
        reference=backbone_reference(12,19)
        for key in m:
            if key!='count_origin': self.assertEqual(m[key],reference[key])

    def test_multiple_backtracks_and_short_image(self):
        m=self.run_case(40,Config(warmup=4,max_window=3))
        self.assertGreater(m['backtracks'],1)
        m=self.run_case(1,Config())
        self.assertEqual(m['model_input_image_token_positions_per_branch'],0)
        self.assertEqual(m['image_probability_evaluations_per_branch'],1)

    def test_means_include_zeros_and_incomplete_or_duplicate_set_fails(self):
        records=[{'sample_id':i,'token_metrics':{
            'inchworm':backbone_reference(576,19),'backbone_reference':backbone_reference(576,19)}} for i in range(100)]
        records[-1]['token_metrics']['inchworm']['regeneration_tokens']=100
        summary=summarize_metrics(records)
        self.assertEqual(summary['inchworm']['means']['regeneration_tokens'],1)
        self.assertEqual(summary['backbone_reference']['means']['regeneration_tokens'],0)
        with self.assertRaises(ValueError): summarize_metrics(records[:-1])
        with self.assertRaises(ValueError): summarize_metrics(records[:-1]+[records[0]])

    def test_inconsistent_event_trace_is_rejected(self):
        meter=MeteredBackend(Backend(),19)
        result=generate(meter,State(),12,Config(warmup=4,max_window=3,max_backtracks=1))
        meter.forward_calls+=1
        with self.assertRaises(ValueError): original_metrics(result,meter)


if __name__=='__main__': unittest.main()

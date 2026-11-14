from dataclasses import asdict
import unittest
from types import SimpleNamespace

from ablation_plan import DEFAULT_ABLATIONS, ablation_flags, benchmark_config, controller_method, variant_config
from four_task_protocol import variant_method
from inchworm_variant_backend import VariantBackend
from inchworm_variant import generate, VariantConfig
from inchworm_core import Config
from test_variant_core import Backend


class AblationTests(unittest.TestCase):
    def test_pure_backbone_is_distinct_from_every_inchworm_version(self):
        self.assertEqual(controller_method('backbone'),'backbone')
        self.assertFalse(any(ablation_flags('backbone').values()))
        self.assertEqual(variant_method('janus1b','backbone'),'janus1b_backbone')
        self.assertNotIn('backbone',DEFAULT_ABLATIONS)
        self.assertNotEqual(variant_method('janus1b','backbone'),variant_method('janus1b','original'))

    def test_each_ablation_changes_only_its_intended_modes(self):
        expected={'guard_only':('entropy','topk'),'guard_temperature':('entropy','temperature'),
            'guard_gap':('mixed','topk'),'combined':('mixed','temperature')}
        self.assertEqual(len(DEFAULT_ABLATIONS),3)
        for version,modes in expected.items():
            config=variant_config(version)
            self.assertEqual((config.position_mode,config.proposal_mode),modes)
            self.assertEqual(config.improvement_margin,0)
            self.assertEqual(asdict(config.controller),asdict(Config()))
        with self.assertRaises(ValueError): variant_config('typo')

    def test_six_score_groups_are_distinct_and_combined_id_is_preserved(self):
        methods={variant_method(model,v) for model in ['janus1b','llamagen'] for v in DEFAULT_ABLATIONS}
        self.assertEqual(len(methods),6)
        self.assertEqual(6*553*4,13272)
        self.assertEqual(variant_method('janus1b','combined'),'janus1b_inchworm_variant')

    def test_third_version_is_guard_and_H_G_without_temperature(self):
        self.assertEqual(ablation_flags('guard_gap'),{'original_path_guard':True,
            'adaptive_temperature':False,'mixed_high_H_high_G':True})
        self.assertEqual(ablation_flags('guard_temperature')['mixed_high_H_high_G'],False)
        self.assertEqual(ablation_flags('guard_only'),{'original_path_guard':True,
            'adaptive_temperature':False,'mixed_high_H_high_G':False})
        self.assertNotIn('combined',DEFAULT_ABLATIONS)

    def test_original_benchmark_uses_original_controller_and_has_no_new_changes(self):
        self.assertEqual(controller_method('original'),'inchworm')
        self.assertEqual(asdict(benchmark_config('original')),asdict(Config()))
        self.assertFalse(any(ablation_flags('original').values()))
        self.assertEqual(variant_method('janus1b','original'),'janus1b_original_inchworm')
        self.assertNotIn('original',DEFAULT_ABLATIONS)
        for version in DEFAULT_ABLATIONS:
            self.assertEqual(controller_method(version),'inchworm_variant')

    def test_topk_delegates_old_proposals_without_importing_torch(self):
        calls=[]
        base=SimpleNamespace(alternatives=lambda p,z,k:calls.append((p,z,k)) or [2,3])
        backend=VariantBackend(base,proposal_mode='topk')
        pred=SimpleNamespace(entropy=2)
        tokens,diag=backend.alternatives(pred,1,2)
        self.assertEqual(tokens,[2,3])
        self.assertFalse(diag['with_replacement'])
        self.assertFalse(diag['original_token_allowed'])
        self.assertEqual(calls,[(pred,1,2)])

    def test_entropy_only_selection_never_uses_G(self):
        class NoGap(Backend):
            def gap(self,pred,token): raise AssertionError('G should not be used')
        config=VariantConfig(controller=Config(warmup=4,max_window=3),position_mode='entropy',max_checks=1)
        result=generate(NoGap(),(),12,config)
        event=next(e for e in result['events'] if e['kind']=='check')
        self.assertEqual(event['positions'],[7,5,6])
        self.assertTrue(all(p['gap'] is None for p in event['proposals']))

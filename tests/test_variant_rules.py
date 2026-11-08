from math import isclose, log
import unittest

from inchworm_variant_rules import (TemperatureConfig,adaptive_temperature,
    mixed_positions,probability_gap,temperature_probabilities)


class VariantRuleTests(unittest.TestCase):
    def test_high_entropy_lowers_temperature_with_bounded_mapping(self):
        values=[adaptive_temperature(h) for h in [0,1,2,4,8,1000]]
        self.assertEqual(values[0],1)
        self.assertTrue(all(0.5<=t<=1 for t in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertAlmostEqual(adaptive_temperature(2),0.5+0.5/2.718281828459045)

    def test_temperature_power_sharpens_odds_and_preserves_zero_support(self):
        p=[0.6,0.3,0.1,0]
        q=temperature_probabilities(p,0.5)
        self.assertAlmostEqual(sum(q),1)
        self.assertEqual(q[-1],0)
        self.assertGreater(q[0],p[0])
        self.assertAlmostEqual(q[0]/q[1],(p[0]/p[1])**2)
        for actual,expected in zip(temperature_probabilities(p,1),p):
            self.assertAlmostEqual(actual,expected)

    def test_uniform_distribution_stays_uniform_even_at_low_temperature(self):
        self.assertEqual(temperature_probabilities([0.25]*4,0.5),[0.25]*4)

    def test_tiny_nonzero_probabilities_use_stable_log_ratio(self):
        gap=probability_gap(1e-100,1e-300)
        self.assertAlmostEqual(gap,200*log(10))
        self.assertEqual(probability_gap(0.6,0.6),0)

    def test_mixed_positions_uses_remaining_gap_positions(self):
        # Gap rank is highest at the two H-selected positions: do not duplicate.
        self.assertEqual(mixed_positions([9,8,1,2,3],[100,90,1,5,4]),[0,1,3,4])

    def test_short_windows_and_ties_are_deterministic(self):
        for count in range(5):
            selected=mixed_positions([1]*count,[1]*count)
            self.assertEqual(selected,list(range(count)))
        self.assertEqual(mixed_positions([1]*8,[1]*8),[0,1,2,3])

    def test_invalid_inputs_fail_instead_of_silently_changing_rules(self):
        for cfg in [TemperatureConfig(minimum=0),TemperatureConfig(maximum=2),
                    TemperatureConfig(entropy_scale=0),TemperatureConfig(minimum=float('nan'))]:
            with self.assertRaises(ValueError): cfg.validate()
        for h in [-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError): adaptive_temperature(h)
        with self.assertRaises(ValueError): probability_gap(0.5,0)
        with self.assertRaises(ValueError): temperature_probabilities([0,0],0.5)
        with self.assertRaises(ValueError): mixed_positions([1],[])

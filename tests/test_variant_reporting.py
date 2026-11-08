import unittest
from run_variant_experiments import summarize_counts


class ReportingTests(unittest.TestCase):
    def test_zero_counts_are_included_and_checks_are_not_called_accepted_edits(self):
        rows=[{'method':'inchworm_variant','sample_id':i,'latency_seconds':1,'token_seconds':0.5,
            'peak_allocated_bytes':100,'peak_reserved_bytes':200,'backtracks':accepted,
            'checks':checks,'accepted_edits':accepted,'kept_original':checks-accepted}
            for i,(checks,accepted) in enumerate([(0,0),(4,1)])]
        result=summarize_counts(rows,[('inchworm_variant',0),('inchworm_variant',1)])['inchworm_variant']
        self.assertEqual(result['mean_checks'],2)
        self.assertEqual(result['mean_accepted_edits'],0.5)
        self.assertEqual(result['mean_kept_original'],1.5)
        rows[1]['kept_original']=0
        with self.assertRaises(ValueError): summarize_counts(rows,[('inchworm_variant',0),('inchworm_variant',1)])

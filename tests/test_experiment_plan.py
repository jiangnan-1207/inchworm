import unittest
from experiment_plan import PAIR_PROMPTS, select_100, summarize
from protocol import TASKS


class ExperimentTests(unittest.TestCase):
    def test_nine_distinct_pairs_and_exactly_100_stratified_prompts(self):
        self.assertEqual(len(PAIR_PROMPTS),9)
        self.assertEqual(len(set(PAIR_PROMPTS)),9)
        rows=[{'tag':t} for t in TASKS for _ in range(100)]
        ids=select_100(rows)
        self.assertEqual(ids,select_100(rows))
        self.assertEqual(len(ids),100)
        self.assertEqual(len(set(ids)),100)
        self.assertEqual({rows[i]['tag'] for i in ids},set(TASKS))

    def test_backtrack_mean_includes_zero_and_incomplete_runs_are_rejected(self):
        records=[{'method':'inchworm','sample_id':i,'latency_seconds':1,'token_seconds':0.5,
            'peak_allocated_bytes':100,'peak_reserved_bytes':200,'backtracks':b} for i,b in enumerate([0,2,4])]
        expected=[('inchworm',i) for i in range(3)]
        self.assertEqual(summarize(records,expected)['inchworm']['mean_backtracks'],2)
        with self.assertRaises(ValueError): summarize(records[:-1],expected)
        with self.assertRaises(ValueError): summarize(records+[records[0]],expected)

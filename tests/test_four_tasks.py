import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from four_task_protocol import full_shard_ids,variant_method
from tools.report_four_tasks import task2

ROOT=Path(__file__).resolve().parents[1]


class FourTaskTests(unittest.TestCase):
    def test_full_shards_cover_553_prompts_and_2212_samples_exactly_once(self):
        for shards in [1,2,4,32,553]:
            groups=[full_shard_ids(i,shards) for i in range(shards)]
            ids=[i for group in groups for i in group]
            self.assertEqual(sorted(ids),list(range(553)))
            samples={(i,j) for i in ids for j in range(4)}
            self.assertEqual(len(samples),2212)
        for index,shards in [(0,0),(2,2),(-1,2),(0,554)]:
            with self.assertRaises(ValueError): full_shard_ids(index,shards)

    def test_new_methods_cannot_be_confused_with_7b_or_original_method(self):
        self.assertEqual(variant_method('janus1b'),'janus1b_inchworm_variant')
        self.assertEqual(variant_method('llamagen'),'llamagen_inchworm_variant')
        with self.assertRaises(ValueError): variant_method('janus7b')

    def test_task2_reads_complete_100_image_mean_without_regenerating(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            target=Path(folder)/'llamagen/profile_summary.json'
            target.parent.mkdir()
            result={'images':100,'total_backtracks':25,'mean_backtracks':.25}
            target.write_text(json.dumps({'results':{'inchworm':result}}))
            self.assertEqual(task2(folder)['mean_backtracks'],.25)
            result['images']=99
            target.write_text(json.dumps({'results':{'inchworm':result}}))
            with self.assertRaises(ValueError): task2(folder)

    def test_fresh_setup_plan_can_download_both_janus_sizes(self):
        result=subprocess.run([sys.executable,ROOT/'tools/bootstrap.py','--data-root',str(ROOT.parent/'data-plan-only'),
            '--janus-models','1b','7b','--plan'],text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout)['janus_models'],['1b','7b'])

    def test_full_generation_and_scoring_help_do_not_require_gpu(self):
        for name in ['generate_variant_full.py','evaluate_variant_full.py']:
            result=subprocess.run([sys.executable,ROOT/name,'--help'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--shards',result.stdout)

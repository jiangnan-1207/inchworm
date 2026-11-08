"""Portable setup plans must not install models or mutate the checkout."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_bootstrap_plan_is_read_only_and_allows_partial_handoff(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent,prefix='inchworm-plan-test-') as folder:
            data = Path(folder)/'new-data-directory'
            result = subprocess.run([sys.executable, ROOT/'tools/bootstrap.py',
                '--data-root', str(data), '--components', 'janus', 'geneval', '--plan'],
                text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['components'], ['janus', 'geneval'])
            self.assertFalse(data.exists())
            self.assertTrue(all(len(repo[1]) == 40 for repo in plan['sources'].values()))

    def test_model_data_cannot_be_placed_inside_public_source_tree(self):
        result = subprocess.run([sys.executable, ROOT/'tools/bootstrap.py',
            '--data-root', str(ROOT/'models'), '--plan'], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outside the source repository', result.stderr)

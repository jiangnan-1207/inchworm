import json
from pathlib import Path
import subprocess
import sys
import unittest

from lumina_layout import Layout,IMAGE_START,IMAGE_END,NEWLINE,CODE_START

ROOT=Path(__file__).resolve().parents[1]


class LuminaLayoutTests(unittest.TestCase):
    def test_768_grid_has_9216_codes_and_96_fixed_newlines(self):
        grid=Layout(); header=[IMAGE_START,8824,8824]
        packed=grid.pack(header,[CODE_START]*grid.length)
        self.assertEqual(grid.length,9216)
        self.assertEqual(grid.native_length,9313)
        self.assertEqual(packed.count(NEWLINE),96)
        self.assertEqual(packed[-1],IMAGE_END)
        grid.validate_native(packed[3:])
        with self.assertRaises(ValueError): grid.validate_native(packed[3:-1])

    def test_newline_enters_pending_cache_but_is_not_a_search_position(self):
        grid=Layout()
        self.assertEqual(grid.pending(CODE_START,95),(CODE_START,))
        self.assertEqual(grid.pending(CODE_START,96),(CODE_START,NEWLINE))
        with self.assertRaises(ValueError): grid.pending(NEWLINE,1)
        with self.assertRaises(ValueError): Layout(767,768)

    def test_thirty_prompts_are_new_and_distinct(self):
        prompts=json.loads((ROOT/'configs/lumina_prompts.json').read_text())
        previous=json.loads((ROOT/'configs/qualitative_prompts.json').read_text())
        old={p for group in previous.values() for p in group}
        self.assertEqual(len(prompts),30)
        self.assertEqual(len(set(prompts)),30)
        self.assertFalse(set(prompts)&old)

    def test_install_plan_and_runner_help_need_no_gpu(self):
        result=subprocess.run([sys.executable,ROOT/'tools/setup_lumina.py','--data-root',str(ROOT.parent/'lumina-plan-data'),
            '--plan'],text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout)['pins']['source_revision'],'0a54b9e0f1bf4ea3577e8a3b1b27b652fbc1e6cc')
        result=subprocess.run([sys.executable,ROOT/'run_lumina_pairs.py','--help'],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

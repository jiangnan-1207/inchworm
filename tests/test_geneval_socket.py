import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix','Linux AF_UNIX regression')
class GenevalSocketTests(unittest.TestCase):
    def test_reproduce_long_path_and_fix_at_evaluator_entrypoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            long_tmp=Path(directory)/('x'*120); long_tmp.mkdir()
            env={**os.environ,'TMPDIR':str(long_tmp),'TMP':str(long_tmp),'TEMP':str(long_tmp)}
            control=subprocess.run([sys.executable,'-c',
                'from multiprocessing.connection import Listener; Listener(authkey=b"test")'],
                cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
            self.assertNotEqual(control.returncode,0)
            self.assertIn('AF_UNIX path too long',control.stderr)
            code='''
import os,tempfile
from pathlib import Path
from unittest.mock import patch
import evaluate_benchmark
original=tempfile.gettempdir()
def probe():
    from multiprocessing.connection import Listener
    directory=Path(tempfile.gettempdir())
    assert str(directory).startswith('/tmp/iw-')
    with Listener(authkey=b'test') as listener:
        assert len(os.fsencode(listener.address))<108
    return directory
with patch.object(evaluate_benchmark,'evaluate',side_effect=probe):
    created=evaluate_benchmark.main()
assert not created.exists()
assert tempfile.gettempdir()==original
assert os.environ['TMPDIR']==original
print('SOCKET_REGRESSION_OK')
'''
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,
                capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SOCKET_REGRESSION_OK',result.stdout)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch shared-storage regression')
    def test_torch_queue_can_share_tensor_with_long_inherited_tmp(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            long_tmp=Path(directory)/('x'*120); long_tmp.mkdir()
            code='''
from short_worker_tmp import short_worker_tmp
with short_worker_tmp():
    import torch
    import torch.multiprocessing as mp
    queue=mp.Queue()
    queue.put(torch.arange(8))
    actual=queue.get(timeout=20)
    assert torch.equal(actual,torch.arange(8))
    queue.close(); queue.join_thread()
print('TORCH_FD_TRANSFER_OK')
'''
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
                env={**os.environ,'TMPDIR':str(long_tmp)},text=True,capture_output=True,timeout=40)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('TORCH_FD_TRANSFER_OK',result.stdout)


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash integration check')
class ScoreOnlySlurmTests(unittest.TestCase):
    def test_three_score_jobs_no_generators_and_only_gap_depends_on_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            base=Path(directory); data=base/'scratch'; baseline=base/'baseline'; ablation=base/'ablation'
            for root,name in [(baseline,'janus1b_backbone'),(ablation,'janus1b_guard_temperature')]:
                path=root/'shards'/f'{name}_0/run_spec.json'; path.parent.mkdir(parents=True); path.write_text('{}')
            config=base/'local.sh'
            config.write_text(f'DATA_ROOT="{data}"\nGENEVAL_PYTHON="{sys.executable}"\n'
                'GPU_REQUEST="gpu:l40s:1"\nPARTITION=gpu-l40s\nCPUS_PER_TASK=4\nMEMORY=64G\n')
            bindir=base/'bin'; bindir.mkdir(); capture=base/'calls.jsonl'; stub=bindir/'sbatch'
            stub.write_text('''#!/usr/bin/env python3
import os,sys,json
with open(os.environ['CAPTURE'],'a') as f:
    f.write(json.dumps({'args':sys.argv[1:],'variant':os.environ['RESCORE_VARIANT'],
        'run':os.environ['EXISTING_RUN'],'out':os.environ['RESCORE_ROOT'],
        'shards':os.environ['RESCORE_SHARDS']})+'\\n')
print(2000)
'''); stub.chmod(0o755)
            env={**os.environ,'CONFIG_FILE':str(config),'ASSETS_ROOT':str(base/'assets'),
                'BASELINE_RUN_ROOT':str(baseline),'ABLATION_RUN_ROOT':str(ablation),
                'SCORE_LOG_ROOT':str(base/'logs'),'GAP_GENERATION_JOB':'15658228',
                'CAPTURE':str(capture),'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            def launch(mode):
                return subprocess.run(['bash',ROOT/'scripts/submit_janus_scores_only.sh',mode],
                    cwd=ROOT,env=env,text=True,capture_output=True)
            result=launch('--check-only'); self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((base/'logs/submissions.txt').exists())
            result=launch('--submit'); self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertTrue(all('--test-only' in row['args'] for row in rows[:3]))
            for row in rows[3:]:
                self.assertEqual(row['args'][-1],'scripts/rescore_existing_variant_worker.sh')
                self.assertIn('--time=02:00:00',row['args'])
                self.assertFalse(any(arg.startswith('--array') for arg in row['args']))
                deps=[arg for arg in row['args'] if arg.startswith('--dependency')]
                self.assertEqual(deps,['--dependency=afterok:15658228'] if row['variant']=='guard_gap' else [])
                self.assertEqual(row['run'],row['out'])
                self.assertEqual(row['shards'],'1' if row['variant']=='backbone' else '32')
            self.assertNotEqual(launch('--submit').returncode,0)
            self.assertEqual(len(capture.read_text().splitlines()),6)


if __name__=='__main__': unittest.main()

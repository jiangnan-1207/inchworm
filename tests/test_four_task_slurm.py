"""Exercise real Bash launchers with a fake sbatch; no jobs or GPUs used."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash integration check')
class SlurmLauncherTests(unittest.TestCase):
    def fixture(self,folder):
        folder=Path(folder)
        bindir=folder/'bin'; bindir.mkdir()
        stub=bindir/'sbatch'
        stub.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_SLURM_LOG'])
rows=p.read_text().splitlines() if p.exists() else []
record={'args':sys.argv[1:],'model':os.environ['EXPERIMENT_MODEL'],'phase':os.environ['FOUR_TASK_PHASE'],'variant':os.environ.get('FOUR_TASK_VARIANT')}
with p.open('a') as out: out.write(json.dumps(record)+'\\n')
print(100+len(rows))
''')
        stub.chmod(0o755)
        config=folder/'local.sh'
        config.write_text(f'''DATA_ROOT="{folder}/data"
GPU_REQUEST="gpu:1"
CPUS_PER_TASK=4
MEMORY="64G"
WALLTIME="02:00:00"
SHARDS=4
CONCURRENT=2
ACCOUNT=""
PARTITION=""
QOS=""
''')
        env={**os.environ,'CONFIG_FILE':str(config),'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
             'FAKE_SLURM_LOG':str(folder/'calls.jsonl')}
        return env

    def call(self,env,*args):
        return subprocess.run(['bash',ROOT/'scripts/submit_four_tasks.sh',*args],env=env,
            text=True,capture_output=True,cwd=ROOT)

    def test_full_scoring_depends_on_its_own_array_and_duplicate_submission_is_blocked(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            env=self.fixture(folder)
            result=self.call(env,'--submit','task4','testfull')
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(x) for x in Path(env['FAKE_SLURM_LOG']).read_text().splitlines()]
            self.assertEqual(len(rows),12)
            expected=[(model,version,phase) for model in ['janus1b','llamagen']
                for version in ['guard_only','guard_temperature','guard_gap'] for phase in ['generate','score']]
            self.assertEqual([(r['model'],r['variant'],r['phase']) for r in rows],expected)
            self.assertIn('--array=0-3%2',rows[0]['args'])
            self.assertIn('--dependency=afterok:100',rows[1]['args'])
            self.assertIn('--dependency=afterok:101',rows[2]['args'])
            self.assertIn('--dependency=afterok:102',rows[3]['args'])
            self.assertIn('--dependency=afterok:106',rows[7]['args'])
            self.assertFalse(any(arg.startswith('--dependency=') for arg in rows[6]['args']))
            self.assertNotEqual(self.call(env,'--submit','task4','testfull').returncode,0)
            self.assertEqual(len(Path(env['FAKE_SLURM_LOG']).read_text().splitlines()),12)

    def test_check_only_does_not_create_submission_journal(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            env=self.fixture(folder)
            result=self.call(env,'--check-only','task4','dryrun','janus1b')
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(x) for x in Path(env['FAKE_SLURM_LOG']).read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertTrue(all('--test-only' in r['args'] for r in rows))
            self.assertFalse((Path(folder)/'data/logs/dryrun.task4.submissions.txt').exists())

    def test_single_selected_version_does_not_submit_other_ablations(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            env=self.fixture(folder); env['ABLATIONS']='guard_temperature'
            result=self.call(env,'--submit','task4','single','llamagen')
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(x) for x in Path(env['FAKE_SLURM_LOG']).read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['variant']=='guard_temperature' for r in rows))

    def test_invalid_or_repeated_model_is_rejected_before_sbatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            env=self.fixture(folder)
            self.assertNotEqual(self.call(env,'--submit','task4','bad','janus7b').returncode,0)
            self.assertNotEqual(self.call(env,'--submit','task1','bad','janus1b','janus1b').returncode,0)
            self.assertFalse(Path(env['FAKE_SLURM_LOG']).exists())

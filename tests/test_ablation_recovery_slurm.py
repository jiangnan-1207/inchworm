import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash integration check')
class AblationRecoveryTests(unittest.TestCase):
    def test_independent_arrays_matching_scores_and_external_worker_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            base=Path(directory); data=base/'scratch'; assets=base/'assets'; out=base/'project_run'
            for suffix in ['benchmarks/geneval/prompts/evaluation_metadata.jsonl',
                    'benchmarks/setup_records/model_revisions_v1.json']:
                path=assets/suffix; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}')
            path=data/'models/Janus-Pro-1B/config.json'; path.parent.mkdir(parents=True); path.write_text('{}')
            bindir=base/'bin'; bindir.mkdir(); calls=base/'calls.jsonl'
            stub=bindir/'sbatch'
            stub.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['CAPTURE']); job=2000+(len(p.read_text().splitlines()) if p.exists() else 0)
with p.open('a') as f: f.write(json.dumps({'args':sys.argv[1:], 'job':job,
    'version':os.environ['RECOVERY_VARIANT'],'phase':os.environ['RECOVERY_PHASE']})+'\\n')
print(job)
'''); stub.chmod(0o755)
            config=base/'local.sh'
            config.write_text(f'DATA_ROOT="{data}"\nJANUS_PYTHON="{sys.executable}"\nGENEVAL_PYTHON="{sys.executable}"\n'
                'GPU_REQUEST="gpu:l40s:1"\nPARTITION=gpu-l40s\nCPUS_PER_TASK=4\nMEMORY=64G\n')
            env={**os.environ,'CONFIG_FILE':str(config),'ASSETS_ROOT':str(assets),'ABLATION_RUN_ROOT':str(out),
                'RECOVERY_CONCURRENT':'2','RECOVERY_SHARDS':'32','CAPTURE':str(calls),
                'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            def submit(mode,environment=env):
                return subprocess.run(['bash',ROOT/'scripts/submit_janus_ablation_recovery.sh',mode],
                    cwd=ROOT,env=environment,text=True,capture_output=True)
            result=submit('--check-only'); self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((out/'logs/submissions.txt').exists())
            rows=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertTrue(all('--test-only' in row['args'] for row in rows))
            result=submit('--submit'); self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in calls.read_text().splitlines()][4:]
            self.assertEqual(len(rows),4)
            for generation,score in [rows[:2],rows[2:]]:
                self.assertEqual(generation['phase'],'generate'); self.assertEqual(score['phase'],'score')
                self.assertEqual(generation['version'],score['version'])
                self.assertFalse(any(a.startswith('--dependency') for a in generation['args']))
                self.assertIn(f"--dependency=afterok:{generation['job']}",score['args'])
                self.assertIn('--array=0-31%2',generation['args'])
                self.assertIn('--cpus-per-task=4',generation['args'])
                self.assertTrue(any(a.startswith('--output='+str(out)) for a in generation['args']))
            self.assertEqual([rows[0]['version'],rows[2]['version']],['guard_temperature','guard_gap'])
            self.assertNotEqual(submit('--submit').returncode,0)
            self.assertEqual(len(calls.read_text().splitlines()),8)
            # Reject accidental writes back into the exhausted scratch tree.
            bad={**env,'ABLATION_RUN_ROOT':str(data/'bad_run')}
            self.assertNotEqual(submit('--check-only',bad).returncode,0)
            self.assertEqual(len(calls.read_text().splitlines()),8)
            fake=base/'fake_python'
            fake.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['WORKER_CAPTURE'],'a') as f: f.write(json.dumps({'args':sys.argv[1:],
    'tmp':os.environ['TMPDIR'],'cache':os.environ['XDG_CACHE_HOME']})+'\\n')
'''); fake.chmod(0o755)
            config.write_text(config.read_text().replace(str(sys.executable),str(fake)))
            env.update(REPO_ROOT=str(ROOT),WORKER_CAPTURE=str(base/'worker.jsonl'),SLURM_ARRAY_TASK_ID='7')
            for version in ['guard_temperature','guard_gap']:
                for phase in ['generate','score']:
                    result=subprocess.run(['bash',ROOT/'scripts/janus_ablation_recovery_worker.sh'],cwd=ROOT,
                        env={**env,'RECOVERY_VARIANT':version,'RECOVERY_PHASE':phase},text=True,capture_output=True)
                    self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in (base/'worker.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            for row in rows:
                args=row['args']
                self.assertEqual(args[args.index('--root')+1],str(assets))
                self.assertEqual(args[args.index('--data-root')+1],str(data))
                self.assertEqual(args[args.index('--run')+1],str(out))
                self.assertEqual(args[args.index('--model')+1],'janus1b')
                self.assertTrue(Path(row['tmp']).is_relative_to(out))
                self.assertTrue(Path(row['cache']).is_relative_to(out))
                if args[1]=='generate_variant_full.py':
                    self.assertEqual(args[args.index('--shard-index')+1],'7')
                else:
                    self.assertEqual(args[1],'evaluate_variant_full.py')
                    self.assertEqual(args[args.index('--evaluation-root')+1],str(out/'merged'))


if __name__=='__main__': unittest.main()

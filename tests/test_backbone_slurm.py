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
class BackboneSlurmTests(unittest.TestCase):
    def test_one_allocation_and_worker_runs_backbone_then_full_scoring(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            base=Path(directory); data=base/'data'; assets=base/'assets'; out=base/'project_run'
            for suffix in ['benchmarks/geneval/prompts/evaluation_metadata.jsonl',
                    'benchmarks/setup_records/model_revisions_v1.json']:
                path=assets/suffix; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}')
            path=data/'models/Janus-Pro-1B/config.json'; path.parent.mkdir(parents=True); path.write_text('{}')
            bindir=base/'bin'; bindir.mkdir(); calls=base/'calls.jsonl'
            stub=bindir/'sbatch'
            stub.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CAPTURE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
print(12345)
'''); stub.chmod(0o755)
            config=base/'local.sh'
            config.write_text(f'DATA_ROOT="{data}"\nJANUS_PYTHON="{sys.executable}"\nGENEVAL_PYTHON="{sys.executable}"\n'
                'GPU_REQUEST="gpu:l40s:1"\nPARTITION=gpu-l40s\nCPUS_PER_TASK=4\nMEMORY=64G\n')
            env={**os.environ,'CONFIG_FILE':str(config),'ASSETS_ROOT':str(assets),'BASELINE_RUN_ROOT':str(out),
                'CAPTURE':str(calls),'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            def submit(mode):
                return subprocess.run(['bash',ROOT/'scripts/submit_janus1b_backbone.sh',mode],
                    cwd=ROOT,env=env,text=True,capture_output=True)
            result=submit('--check-only'); self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((out/'logs/submission.txt').exists())
            result=submit('--submit'); self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertIn('--test-only',rows[0]); self.assertNotIn('--test-only',rows[1])
            self.assertIn('--time=08:00:00',rows[1]); self.assertIn('--gres=gpu:l40s:1',rows[1])
            self.assertNotEqual(submit('--submit').returncode,0)
            fake=base/'fake_python'
            fake.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['WORKER_CAPTURE'],'a') as f: f.write(json.dumps({'args':sys.argv[1:],
    'tmp':os.environ['TMPDIR'],'cache':os.environ['XDG_CACHE_HOME']})+'\\n')
'''); fake.chmod(0o755)
            config.write_text(config.read_text().replace(str(sys.executable),str(fake)))
            env.update(REPO_ROOT=str(ROOT),WORKER_CAPTURE=str(base/'worker.jsonl'))
            result=subprocess.run(['bash',ROOT/'scripts/janus1b_backbone_full_worker.sh'],cwd=ROOT,
                env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(line) for line in (base/'worker.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['args'][1],'generate_variant_full.py')
            self.assertEqual(rows[1]['args'][1],'evaluate_variant_full.py')
            for row in rows:
                args=row['args']
                self.assertEqual(args[args.index('--variant')+1],'backbone')
                self.assertEqual(args[args.index('--model')+1],'janus1b')
                self.assertEqual(args[args.index('--shards')+1],'1')
                self.assertTrue(Path(row['tmp']).is_relative_to(out))
                self.assertTrue(Path(row['cache']).is_relative_to(out))


if __name__=='__main__': unittest.main()

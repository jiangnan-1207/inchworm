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
class TokenSlurmTests(unittest.TestCase):
    def test_dry_run_then_single_job_and_duplicate_prevention(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            root=Path(folder)
            bindir=root/'bin'; bindir.mkdir()
            stub=bindir/'sbatch'
            stub.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                'with open(os.environ["FAKE_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(12345)\n')
            stub.chmod(0o755)
            metadata=root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl'
            metadata.parent.mkdir(parents=True); metadata.touch()
            revisions=root/'benchmarks/setup_records/model_revisions_v1.json'
            revisions.parent.mkdir(parents=True); revisions.touch()
            config=root/'local.sh'
            config.write_text(f'DATA_ROOT="{root}/data"\nJANUS_PYTHON="{sys.executable}"\n'
                'GPU_REQUEST="gpu:l40s:1"\nPARTITION="gpu-l40s"\nCPUS_PER_TASK=4\nMEMORY=64G\n')
            env={**os.environ,'CONFIG_FILE':str(config),'ASSETS_ROOT':str(root),
                'FAKE_LOG':str(root/'calls.jsonl'),'PATH':str(bindir)+os.pathsep+os.environ['PATH']}

            def call(mode,model='janus1b'):
                return subprocess.run(['bash',ROOT/'scripts/submit_token_profile.sh',mode,model,'tokens01'],
                    env=env,cwd=ROOT,text=True,capture_output=True)

            result=call('--check-only')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(list((root/'data/logs').iterdir()))
            result=call('--submit')
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls),2)
            self.assertIn('--test-only',calls[0]); self.assertNotIn('--test-only',calls[1])
            self.assertIn('--cpus-per-task=4',calls[1]); self.assertIn('--gres=gpu:l40s:1',calls[1])
            self.assertIn('--time=04:00:00',calls[1])
            self.assertNotEqual(call('--submit').returncode,0)
            self.assertNotEqual(call('--submit','unsupported').returncode,0)
            self.assertEqual(len((root/'calls.jsonl').read_text().splitlines()),2)


if __name__=='__main__': unittest.main()

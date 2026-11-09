"""Validate real Bash submission dependency wiring without a cluster."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash integration check')
class AlignmentSlurmTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder)/'checkout'; (root/'scripts').mkdir(parents=True)
        for f in ['submit_alignment.sh','common.sh','alignment_worker.sh']:
            shutil.copyfile(ROOT/'scripts'/f,root/'scripts'/f)
        records=root/'benchmarks/setup_records'; records.mkdir(parents=True)
        for name in ['manifest','dpg_judge','tiif_judge']: (records/f'alignment_{name}.json').write_text('{}')
        bindir=Path(folder)/'bin'; bindir.mkdir()
        stub=bindir/'sbatch'; stub.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_LOG'])
rows=p.read_text().splitlines() if p.exists() else []
with p.open('a') as f: f.write(json.dumps(dict(args=sys.argv[1:],model=os.environ['ALIGNMENT_MODEL'],stage=os.environ['ALIGNMENT_STAGE']))+'\\n')
print(700+len(rows))
'''); stub.chmod(0o755)
        config=Path(folder)/'local.sh'
        config.write_text(f'DATA_ROOT="{folder}/data"\nJANUS_PYTHON=/bin/true\nLLAMA_PYTHON=/bin/true\n')
        env={**os.environ,'CONFIG_FILE':str(config),'FAKE_LOG':str(Path(folder)/'calls.jsonl'),
            'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'ALIGNMENT_SHARDS':'4','ALIGNMENT_CONCURRENT':'2'}
        return root,env

    def call(self,root,env,*args):
        return subprocess.run(['bash',root/'scripts/submit_alignment.sh',*args],env=env,cwd=root,text=True,capture_output=True)

    def test_two_arrays_and_four_scorers_depend_on_correct_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root,env=self.fixture(folder)
            result=self.call(root,env,'--submit','generate','alignmenttest')
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(s) for s in Path(env['FAKE_LOG']).read_text().splitlines()]
            self.assertEqual([r['stage'] for r in rows],['generate','score','score']*2)
            self.assertEqual([r['model'] for r in rows],['janus1b']*3+['llamagen']*3)
            for i,parent in [(1,700),(2,700),(4,703),(5,703)]:
                self.assertIn(f'--dependency=afterok:{parent}',rows[i]['args'])
                self.assertFalse(any(x.startswith('--array=') for x in rows[i]['args']))
            self.assertNotEqual(self.call(root,env,'--submit','generate','alignmenttest').returncode,0)
            self.assertEqual(len(Path(env['FAKE_LOG']).read_text().splitlines()),6)

    def test_check_only_never_submits_real_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root,env=self.fixture(folder)
            result=self.call(root,env,'--check-only','generate','dryrun')
            self.assertEqual(result.returncode,0,result.stderr)
            rows=[json.loads(s) for s in Path(env['FAKE_LOG']).read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertTrue(all('--test-only' in r['args'] for r in rows))
            self.assertFalse((Path(folder)/'data/logs/dryrun.alignment-generate.txt').exists())


if __name__=='__main__': unittest.main()

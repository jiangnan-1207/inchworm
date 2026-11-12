"""Verify the quota workaround with a fake inference process, without a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash integration check')
class DiagnosticStorageTests(unittest.TestCase):
    def test_output_and_write_caches_move_but_asset_root_stays(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            folder=Path(directory); scratch=folder/'scratch'; project=folder/'project'
            scratch.mkdir(); project.mkdir()
            for name in ['tmp','logs']: (scratch/name).mkdir()
            reference=scratch/'reference.png'; reference.write_bytes(b'fixture')
            report=folder/'captured.json'
            stub=folder/'python_stub'
            stub.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
keys=['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE',
      'TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']
Path(os.environ['CAPTURE_FILE']).write_text(json.dumps({'args':sys.argv[1:],
    'env':{k:os.environ[k] for k in keys}}))
''')
            stub.chmod(0o755)
            config=folder/'local.sh'
            config.write_text(f'DATA_ROOT="{scratch}"\nLLAMA_PYTHON="{stub}"\n')
            before={str(p.relative_to(scratch)) for p in scratch.rglob('*')}
            env={**os.environ,'REPO_ROOT':str(ROOT),'CONFIG_FILE':str(config),
                'LLAMA_SOURCE_ROOT':str(folder/'source'),'LLAMA_ASSET_REPORT':str(folder/'audit.json'),
                'LLAMA_REFERENCE_IMAGE':str(reference),'LLAMA_DIAGNOSTIC_ROOT':str(project),
                'CAPTURE_FILE':str(report),'SLURM_JOB_ID':'12345'}
            result=subprocess.run(['bash',ROOT/'scripts/llama_structure_worker.sh'],cwd=ROOT,
                env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            captured=json.loads(report.read_text())
            args=captured['args']
            self.assertEqual(args[args.index('--data-root')+1],str(scratch))
            self.assertEqual(args[args.index('--out')+1],str(project/'llamagen_structure_12345'))
            for value in captured['env'].values():
                self.assertTrue(Path(value).is_relative_to(project))
            self.assertEqual(before,{str(p.relative_to(scratch)) for p in scratch.rglob('*')})


if __name__=='__main__': unittest.main()

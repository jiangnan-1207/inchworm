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


class FreshIdentityTests(unittest.TestCase):
    def test_stages_have_distinct_checkpoints_and_benchmark_names(self):
        from tools.run_llamagen_fresh import identity
        a=identity(256); b=identity(512)
        self.assertNotEqual(a,b)
        self.assertEqual(a[1],'t2i_XL_stage1_256.pt')
        self.assertEqual(b[1],'t2i_XL_stage2_512.pt')
        self.assertNotEqual(b[0],'llamagen_baseline')


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux deployment integration')
class FreshSubmissionTests(unittest.TestCase):
    def test_isolated_cpu_deploy_then_gpu_demo_and_full_array_then_score(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            base=Path(directory); fresh=base/'fresh'; config=base/'local.sh'; calls=base/'calls'
            config.write_text(f'DATA_ROOT="{base}/old"\nJANUS_PYTHON="{sys.executable}"\n'
                'GPU_REQUEST="gpu:l40s:1"\nPARTITION=gpu-l40s\nCPUS_PER_TASK=4\nMEMORY=64G\n')
            bindir=base/'bin'; bindir.mkdir(); stub=bindir/'sbatch'
            stub.write_text('''#!/usr/bin/env python3
import os,json,sys
from pathlib import Path
p=Path(os.environ['CAPTURE']); job=3000+(len(p.read_text().splitlines()) if p.exists() else 0)
with p.open('a') as f: f.write(json.dumps({'args':sys.argv[1:],'phase':os.environ['FRESH_PHASE'],'job':job})+'\\n')
print(job)
'''); stub.chmod(0o755)
            env={**os.environ,'CONFIG_FILE':str(config),'FRESH_LLAMA_ROOT':str(fresh),
                'ASSETS_ROOT':str(base/'assets'),'CAPTURE':str(calls),'PATH':str(bindir)+os.pathsep+os.environ['PATH']}
            def launch(mode,task):
                return subprocess.run(['bash',ROOT/'scripts/submit_llamagen_fresh.sh',mode,task],
                    cwd=ROOT,env=env,text=True,capture_output=True)
            r=launch('--check-only','deploy'); self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse((fresh/'logs/deploy.submissions.txt').exists())
            r=launch('--submit','deploy'); self.assertEqual(r.returncode,0,r.stderr)
            rows=[json.loads(x) for x in calls.read_text().splitlines()][2:]
            self.assertEqual([r['phase'] for r in rows],['setup','demo'])
            self.assertIn('--partition=cpu-small',rows[0]['args'])
            self.assertFalse(any(arg.startswith('--gres') for arg in rows[0]['args']))
            self.assertIn('--partition=gpu-l40s',rows[1]['args'])
            self.assertIn(f"--dependency=afterok:{rows[0]['job']}",rows[1]['args'])
            self.assertNotEqual(launch('--submit','deploy').returncode,0)
            (fresh/'setup_complete.json').write_text('{}')
            (fresh/'native_images').mkdir(); (fresh/'native_images/completed.json').write_text('{}')
            r=launch('--submit','full512'); self.assertEqual(r.returncode,0,r.stderr)
            rows=[json.loads(x) for x in calls.read_text().splitlines()][4:]
            self.assertEqual([r['phase'] for r in rows],['generate','score'])
            self.assertIn('--array=0-31%2',rows[0]['args'])
            self.assertIn(f"--dependency=afterok:{rows[0]['job']}",rows[1]['args'])


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ['torch','transformers','timm','ftfy','bs4'])
    and (ROOT/'benchmarks/LlamaGen').is_dir(),
    'Requires Torch and pinned official source')
class FreshUpstreamTests(unittest.TestCase):
    def test_upstream_imports_and_native_sampler_with_real_tiny_gpt(self):
        import torch
        sys.path.insert(0,str(ROOT/'benchmarks/LlamaGen'))
        from language.t5 import T5Embedder
        from tokenizer.tokenizer_image.vq_model import VQ_models
        from autoregressive.models.gpt import Transformer,ModelArgs
        from autoregressive.models.generate import generate
        self.assertIn('VQ-16',VQ_models)
        with torch.inference_mode():
            model=Transformer(ModelArgs(dim=32,n_layer=2,n_head=4,multiple_of=16,
                caption_dim=8,cls_token_num=120,block_size=16,vocab_size=32,model_type='t2i')).eval()
            torch.nn.init.normal_(model.output.weight,std=0.1)
            cond=torch.randn(1,120,8); mask=torch.tensor([[0]*117+[1,1,1]])
            result=generate(model,cond,4,mask,cfg_scale=7.5,temperature=1.,top_k=8,top_p=1.,sample_logits=True)
            self.assertEqual(tuple(result.shape),(1,4))
            self.assertTrue(bool(((result>=0)&(result<32)).all()))


if __name__=='__main__': unittest.main()

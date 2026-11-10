import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import run_token_profile
from token_accounting import backbone_reference


class FakeImage:
    def save(self,path,format): Path(path).write_bytes(b'fake PNG for receipt verification')


class TokenProfileTests(unittest.TestCase):
    def test_complete_resume_and_corrupt_receipt_without_loading_gpu(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder)
            metadata=root/'benchmarks/geneval/prompts/evaluation_metadata.jsonl'
            metadata.parent.mkdir(parents=True); metadata.write_text('fixture')
            revisions=root/'benchmarks/setup_records/model_revisions_v1.json'
            revisions.parent.mkdir(parents=True); revisions.write_text('{}')
            gpu=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda i:'fixture GPU',
                manual_seed_all=Mock(),empty_cache=Mock())
            torch=types.SimpleNamespace(cuda=gpu,__version__='fixture',version=types.SimpleNamespace(cuda='fixture'),
                backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
                    cudnn=types.SimpleNamespace()),manual_seed=Mock(),inference_mode=contextlib.nullcontext)
            numpy=types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock()))
            metric=backbone_reference(576,19)
            produce=Mock(side_effect=lambda *a:(FakeImage(),0,{'backtracks':0,'events':[],
                'token_metrics':{'inchworm':metric,'backbone_reference':metric}}))
            loader=Mock(return_value=produce)
            argv=['run_token_profile.py','--model','janus1b','--assets-root',str(root),
                '--data-root',str(root/'data'),'--out',str(root/'out')]
            rows=[{'tag':'single_object','prompt':f'prompt {i}'} for i in range(100)]
            with patch.dict('sys.modules',{'torch':torch,'numpy':numpy,
                    'variant_model_runner':types.SimpleNamespace(load_model=loader)}), \
                    patch('sys.argv',argv),patch.object(run_token_profile,'load_prompts',return_value=rows), \
                    patch.object(run_token_profile,'select_100',return_value=list(range(100))), \
                    patch.object(run_token_profile.gc,'collect'),contextlib.redirect_stdout(io.StringIO()):
                run_token_profile.main()
                loader.assert_called_once_with('janus1b',root,root/'data',collect_token_metrics=True)
                self.assertEqual(produce.call_count,101)  # one excluded warmup
                self.assertEqual(torch.manual_seed.call_args_list[0].args,(0,))
                self.assertEqual(torch.manual_seed.call_args_list[-1].args,(42+4*99,))
                summary=json.loads((root/'out/janus1b/token_summary.json').read_text())
                self.assertEqual(summary['results']['inchworm']['images'],100)
                run_token_profile.main()
                self.assertEqual(loader.call_count,1)  # complete resume does not load the model
                self.assertEqual(produce.call_count,101)
                png=root/'out/janus1b/inchworm/00000/samples/0000.png'
                png.write_bytes(b'corrupt')
                with self.assertRaises(ValueError): run_token_profile.main()


if __name__=='__main__': unittest.main()

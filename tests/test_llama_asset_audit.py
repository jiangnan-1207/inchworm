import hashlib
import importlib.util
import io
import contextlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('llama_asset_audit',ROOT/'tools/audit_llamagen_assets.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class AssetAuditTests(unittest.TestCase):
    def fixture(self,root):
        models=[]
        for repo,subdir,names in [('peizesun/llamagen_t2i','',['t2i_XL_stage2_512.pt','vq_ds16_t2i.pt']),
                ('google/flan-t5-xl','t5-ckpt/flan-t5-xl',audit.T5_REQUIRED+[
                    'pytorch_model.bin.index.json','pytorch_model-00001-of-00001.bin'])]:
            folder=root/subdir; folder.mkdir(parents=True,exist_ok=True)
            files=[]
            for name in names:
                data=(json.dumps({'weight_map':{'encoder.weight':'pytorch_model-00001-of-00001.bin'}}).encode()
                    if name.endswith('.index.json') else b'fixture')
                (folder/name).write_bytes(data)
                files.append({'name':name,'size':len(data),
                    'sha256':hashlib.sha256(data).hexdigest() if name.endswith(('.pt','.bin')) else None,
                    'git_blob_sha1':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
            models.append({'repo':repo,'revision':'fixture','subdir':subdir,'files':files})
        return {'schema':1,'models':models}

    def call(self,root,manifest):
        with contextlib.redirect_stdout(io.StringIO()): return audit.audit_assets(root,manifest)

    def test_all_hashes_match_and_no_files_changed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root=Path(directory); manifest=self.fixture(root)
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(self.call(root,manifest)['status'],'assets_match_pinned_sources')
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_same_size_corruption_is_detected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root=Path(directory); manifest=self.fixture(root)
            (root/'t2i_XL_stage2_512.pt').write_bytes(b'corrupt')
            result=self.call(root,manifest)
            self.assertEqual(result['status'],'asset_audit_failed')
            self.assertIn('hash_mismatch',[r['status'] for r in result['files']])

    def test_missing_shard_and_extra_tokenizer_are_detected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root=Path(directory); manifest=self.fixture(root)
            folder=root/'t5-ckpt/flan-t5-xl'
            (folder/'pytorch_model-00001-of-00001.bin').unlink()
            (folder/'added_tokens.json').write_text('{}')
            result=self.call(root,manifest)
            self.assertEqual(result['status'],'asset_audit_failed')
            self.assertIn('missing',[r['status'] for r in result['files']])
            self.assertIn('unrecognized_inference_file',[r['status'] for r in result['files']])

    def test_git_blob_digest_is_not_plain_sha1(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            path=Path(directory)/'config.json'; path.write_bytes(b'fixture')
            result=audit.verify_file(path,{'size':7,'sha256':None,
                'git_blob_sha1':hashlib.sha1(b'fixture').hexdigest()})
            self.assertEqual(result['status'],'hash_mismatch')


if __name__=='__main__': unittest.main()

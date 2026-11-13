"""A complete benchmark merge may write links elsewhere; shards stay untouched."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from merge_full import merge
from protocol import bind_run,commit_image,sha

ROOT=Path(__file__).resolve().parents[1]


class Image:
    def save(self,path,format): Path(path).write_bytes(b'fixture PNG')


@unittest.skipUnless(os.name=='posix','Requires POSIX symbolic links')
class ExternalEvaluationTests(unittest.TestCase):
    def test_553_prompt_merge_and_resume_write_only_to_destination(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            base=Path(directory); run=base/'scratch/run'; destination=base/'project/merged'
            method='janus1b_guard_only'
            rows=[{'tag':'single_object','prompt':f'test {i}'} for i in range(553)]
            for s in range(2):
                ids=list(range(s,553,2)); shard=run/'shards'/f'{method}_{s}'
                spec={'method':method,'shards':2,'shard_index':s,'prompt_ids':ids}
                bind_run(shard,spec)
                for i in ids:
                    for j in range(4): commit_image(shard,method,i,j,rows[i],spec,Image(),{})
            before={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()}
            with patch('merge_full.load_prompts',return_value=rows),contextlib.redirect_stdout(io.StringIO()):
                merge(base,run,method,2,destination)
                merge(base,run,method,2,destination)
            self.assertFalse((run/'merged').exists())
            self.assertTrue((destination/method/'00024').is_symlink())
            self.assertEqual((destination/method/'00024').resolve(),
                (run/'shards'/f'{method}_0'/method/'00024').resolve())
            coverage=json.loads((destination/f'{method}.coverage.json').read_text())
            self.assertEqual((coverage['prompts'],coverage['images']),(553,2212))
            self.assertEqual(before,{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
            # Existing unrelated real folders must never be overwritten.
            link=destination/method/'00024'; link.unlink(); link.mkdir()
            with patch('merge_full.load_prompts',return_value=rows),self.assertRaises(ValueError):
                merge(base,run,method,2,destination)


if __name__=='__main__': unittest.main()

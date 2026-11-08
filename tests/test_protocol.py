import json
from pathlib import Path
import tempfile
import unittest

from protocol import (TASKS, atomic_json, bind_run, commit_image, completed,
                      paths, seed_for, select_ids, validate_outputs)


class FakeImage:
    def save(self,path,format):
        assert format=='PNG'
        Path(path).write_bytes(b'fake test image, not a real exported PNG')


class ProtocolTests(unittest.TestCase):
    def test_unique_seeds_for_all_2212_samples(self):
        seeds={seed_for(i,j) for i in range(553) for j in range(4)}
        self.assertEqual(len(seeds),2212)

    def test_pilot_covers_every_task_without_renumbering(self):
        rows=[{'tag':tag} for tag in TASKS for _ in range(2)]
        self.assertEqual(select_ids(rows,'pilot'),[0,2,4,6,8,10])
        self.assertEqual(select_ids(rows,'full'),list(range(12)))

    def test_resume_and_checksum(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            spec={'scope':'pilot'}
            row={'tag':'single_object','prompt':'test'}
            bind_run(folder,spec)
            bind_run(folder,spec)
            image,receipt=paths(folder,'test',0,0)
            self.assertFalse(completed(image,receipt,spec,row))
            commit_image(folder,'test',0,0,row,spec,FakeImage(),{'generation_seconds':1.0})
            self.assertTrue(completed(image,receipt,spec,row))
            image.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                completed(image,receipt,spec,row)

    def test_cannot_mix_configurations(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            bind_run(folder,{'scope':'pilot'})
            with self.assertRaises(ValueError):
                bind_run(folder,{'scope':'full'})

    def test_missing_sample_is_an_error_not_a_lower_denominator(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            rows=[{'tag':'single_object','prompt':'test'}]
            commit_image(folder,'test',0,0,rows[0],{},FakeImage(),{})
            self.assertEqual(validate_outputs(folder,'test',rows,[0],1),1)
            with self.assertRaises(ValueError):
                validate_outputs(folder,'test',rows,[0],4)

    def test_merge_validation_rejects_a_receipt_from_another_configuration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            rows=[{'tag':'single_object','prompt':'test'}]
            bind_run(folder,{'cfg':5})
            commit_image(folder,'test',0,0,rows[0],{'cfg':7},FakeImage(),{})
            with self.assertRaises(ValueError):
                validate_outputs(folder,'test',rows,[0],1)

    def test_extra_prompt_and_conflicting_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            row={'prompt':'a'}
            commit_image(folder,'test',0,0,row,{},FakeImage(),{})
            with self.assertRaises(ValueError):
                commit_image(folder,'test',0,0,{'prompt':'b'},{},FakeImage(),{})
            (Path(folder)/'test/00001').mkdir()
            with self.assertRaises(ValueError):
                validate_outputs(folder,'test',[row],[0],1)


if __name__=='__main__':
    unittest.main()

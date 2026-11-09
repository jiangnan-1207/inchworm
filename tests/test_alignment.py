import csv
import json
from pathlib import Path
import tempfile
import unittest
from alignment_metrics import GROUPS,aggregate_tiif,dependency_score,dpg_questions,parse_answers
from alignment_protocol import build_manifest


class AlignmentTests(unittest.TestCase):
    def test_dependencies_propagate_and_missing_answers_fail(self):
        q={1:dict(dependencies=[0]),2:dict(dependencies=[1]),3:dict(dependencies=[2])}
        self.assertEqual(dependency_score(q,{1:'no',2:'yes',3:'yes'}),0)
        self.assertEqual(dependency_score(q,{1:'yes',2:'no',3:'yes'}),1/3)
        with self.assertRaises(ValueError): dependency_score(q,{1:'yes'})

    def test_csv_first_question_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'q.csv'
            path.write_text('item_id,proposition_id,dependency,tuple,question_natural_language\na,1,0,entity,Is it there?\na,2,1,color,Is it red?\n')
            self.assertEqual(set(dpg_questions(path)['a']),{1,2})

    def test_cyclic_dependencies_follow_official_single_pass(self):
        q={1:dict(dependencies=[2]),2:dict(dependencies=[1])}
        self.assertEqual(dependency_score(q,{1:'yes',2:'yes'}),1)
        self.assertEqual(dependency_score(q,{1:'yes',2:'no'}),0)

    def test_judge_format_is_strict(self):
        self.assertEqual(parse_answers('yes, visible\nno, missing',2),['yes','no'])
        with self.assertRaises(ValueError): parse_answers('yes',2)

    def test_nine_group_macro_and_missing_dimension_gate(self):
        records=[dict(kind=kind,length=length,correct=1,questions=1,text_score=0)
            for kind in sum(GROUPS.values(),[]) for length in ['short','long']]
        self.assertEqual(len(set(sum(GROUPS.values(),[]))),39)
        self.assertAlmostEqual(aggregate_tiif(records)['short']['overall'],800/9)
        with self.assertRaises(ValueError): aggregate_tiif(records[1:])

    def test_official_dataset_if_checked_out(self):
        root=Path(__file__).resolve().parents[1]
        if not (root/'benchmarks/ELLA/dpg_bench/dpg_bench.csv').exists():
            self.skipTest('Official sources are checked in the dataset CI job')
        manifest=build_manifest(root)
        self.assertEqual(manifest['counts']['dpg'],4260)
        questions=dpg_questions(root/'benchmarks/ELLA/dpg_bench/dpg_bench.csv')
        self.assertEqual({r['key'] for r in manifest['rows'] if r['benchmark']=='dpg'},set(questions))
        for group in questions.values():
            self.assertEqual(dependency_score(group,{qid:'yes' for qid in group}),1)
        import re
        for row in manifest['rows']:
            if row['benchmark']=='tiif':
                self.assertTrue(all(s.strip().lower() in ['yes','no'] for s in row['answers']))
                if row['kind']=='text':
                    for q in row['questions']: self.assertIsNotNone(re.fullmatch(r"Is the text '(.*)' in the image\?",q.strip()))
        print('ALIGNMENT_DATA_COUNTS',manifest['counts'],flush=True)


if __name__=='__main__': unittest.main()

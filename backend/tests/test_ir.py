import unittest
from pathlib import Path
from backend.data_loader import load_dataset
from backend.document_builder import build_documents
from backend.boolean_model import BooleanModel, QueryError
from backend.vector_model import VectorModel
from backend.preprocessing import tokenize
from backend.evaluation import evaluate

ROOT=Path(__file__).resolve().parents[2]/"Dataset for autism copy"

class IRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root,rows=load_dataset(ROOT); cls.docs=build_documents(root,rows)
        cls.boolean=BooleanModel(cls.docs); cls.vector=VectorModel(cls.docs)

    def test_dataset_and_documents(self):
        self.assertEqual(len(self.docs),100)
        self.assertEqual(sum(d['files']['image']['exists'] for d in self.docs),100)
        self.assertEqual(sum(d['files']['voice']['exists'] for d in self.docs),100)
        self.assertEqual(sum(d['files']['motion']['exists'] for d in self.docs),100)
        self.assertEqual(sum(d['files']['physio']['exists'] for d in self.docs),100)

    def test_preprocessing_preserves_research_terms_and_not(self):
        self.assertEqual(tokenize('Autism, ASD Level-1! NOT eye/gaze'),['autism','asd','level','1','not','eye','gaze'])

    def test_inverted_index_and_boolean_operators(self):
        self.assertEqual(len(self.boolean.search('mild_asd')),23)
        self.assertEqual(len(self.boolean.search('mild AND autism')),23)
        self.assertEqual(len(self.boolean.search('mild OR severe')),56)
        self.assertEqual(len(self.boolean.search('autism AND NOT voice')),0) # all sample docs advertise voice

    def test_boolean_invalid_and_empty(self):
        for q in ('','autism AND','AND voice'):
            with self.assertRaises(QueryError): self.boolean.search(q)

    def test_vector_ranking_cosine_top_k(self):
        found=self.vector.search('child autism screening',top_k=5)
        self.assertLessEqual(len(found),5); self.assertTrue(found)
        self.assertGreater(found[0]['score'],0); self.assertGreaterEqual(found[0]['score'],found[-1]['score'])
        self.assertLessEqual(found[0]['score'],1.000001)

    def test_modality_filter(self):
        result=self.vector.search('autism',100,'voice')
        self.assertEqual(len(result),100)
        self.assertEqual(self.vector.search('autism',100,'images'),[])

    def test_no_match_and_bad_top_k(self):
        self.assertEqual(self.vector.search('zzznomatch'),[])
        with self.assertRaises(ValueError): self.vector.search('autism',0)

    def test_evaluation_metrics(self):
        self.assertEqual(evaluate(['a','b'],['a','c'],2),{'precision':.5,'recall':.5,'f1':.5,'precision_at_k':.5,'recall_at_k':.5})

if __name__=='__main__': unittest.main()

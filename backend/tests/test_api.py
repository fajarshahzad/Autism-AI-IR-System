import unittest
from fastapi import HTTPException
from backend.main import health, stats, boolean_search, vector_search, document, serve_file, SearchRequest

class EndpointTests(unittest.TestCase):
    def test_health_and_stats(self):
        self.assertEqual(health()['status'],'ok')
        self.assertEqual(stats()['total_documents'],100)
        self.assertEqual(stats()['modalities'],{'image':100,'motion':100,'physio':100,'voice':100})
        self.assertEqual(stats()['missing_indexed_files'],100)

    def test_search_routes(self):
        b=boolean_search(SearchRequest(query='mild AND autism'))
        v=vector_search(SearchRequest(query='child autism screening',modality='image',top_k=3))
        self.assertEqual(b['model'],'boolean'); self.assertEqual(b['total_results'],23)
        self.assertEqual(v['model'],'vector_space'); self.assertEqual(len(v['results']),3)
        self.assertEqual(v['results'][0]['files']['image']['file_type'],'jpg')

    def test_document_and_file_routes(self):
        self.assertEqual(document('child_001')['document_id'],'child_001')
        self.assertTrue(serve_file('child_001','image').path.endswith('Child (1).jpg'))
        with self.assertRaises(HTTPException): document('missing')

    def test_invalid_boolean_route(self):
        with self.assertRaises(HTTPException) as err: boolean_search(SearchRequest(query='autism AND'))
        self.assertEqual(err.exception.status_code,400)

if __name__=='__main__': unittest.main()

import sys
import os
import unittest
import asyncio

# Setup sys.path to find backend services
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from services.rag_engine import RAGEngine, truncate_query_safely
from config import settings

class TestRAGEngineQueryTruncation(unittest.TestCase):
    def test_truncate_query_safely(self):
        # Case 1: string shorter than limit
        self.assertEqual(truncate_query_safely("hello world", 20), "hello world")
        
        # Case 2: string exactly limit
        self.assertEqual(truncate_query_safely("hello world", 11), "hello world")
        
        # Case 3: truncate on whitespace boundary
        # "hello world standard" with max 15 should slice to "hello world sta" -> last space -> "hello world"
        self.assertEqual(truncate_query_safely("hello world standard", 15), "hello world")
        
        # Case 4: hard truncation when no spaces exist
        self.assertEqual(truncate_query_safely("helloworldlonglonglong", 10), "helloworld")
        
        # Case 5: empty query
        self.assertEqual(truncate_query_safely("", 10), "")

    def test_query_validation_and_truncation(self):
        # We need an event loop to run async tests
        loop = asyncio.get_event_loop()
        rag = RAGEngine()
        
        # Query under 30 characters should abort/skip
        short_query = "hypertension"
        result = loop.run_until_complete(rag.query(short_query))
        self.assertEqual(result["documents"], [[]])
        self.assertEqual(result["timings"]["total_query_ms"], 0.0)
        
        # Query exactly 30 characters or longer should run
        long_query = "essential hypertension of the pulmonary artery system"
        self.assertTrue(len(long_query) >= 30)
        result = loop.run_until_complete(rag.query(long_query))
        
        self.assertIn("documents", result)
        self.assertGreater(result["timings"]["total_query_ms"], 0.0)

    def test_batch_query_validation(self):
        loop = asyncio.get_event_loop()
        rag = RAGEngine()
        
        # One valid query, one short query
        queries = [
            "essential hypertension of the pulmonary artery system", # valid, >=30 chars
            "diabetes" # invalid, <30 chars
        ]
        
        # Generate dummy 384-dimensional vectors
        dummy_vectors = [[0.1] * 384, [0.2] * 384]
        
        results = loop.run_until_complete(rag.batch_query(queries, dummy_vectors))
        
        # We expect a list of 2 results
        self.assertEqual(len(results), 2)
        
        # First one should be processed (not skipped)
        self.assertNotIn("skipped", results[0].get("timings", {}))
        
        # Second one should be skipped
        self.assertTrue(results[1].get("timings", {}).get("skipped"))
        self.assertEqual(results[1]["decision"]["confidence"]["score"], 0.0)

if __name__ == "__main__":
    unittest.main()
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
import asyncio
from services.rag_engine import RAGEngine

async def test():
    rag = RAGEngine()

    query = "hypertension"
    results = await rag.query(query, n_results=5)

    print("QUERY:", query)
    print("RESULTS:", results)

asyncio.run(test())
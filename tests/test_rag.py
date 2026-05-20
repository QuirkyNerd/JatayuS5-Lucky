import asyncio
from services.rag_engine import RAGEngine

async def test():
    rag = RAGEngine()

    query = "hypertension"
    results = await rag.query(query, n_results=5)

    print("QUERY:", query)
    print("RESULTS:", results)

asyncio.run(test())
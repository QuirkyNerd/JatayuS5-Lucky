import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_PATH = ROOT / "backend"

sys.path.insert(0, str(BACKEND_PATH))

from services.rag_engine import RAGEngine

print("\n=== INITIALIZING RAG ENGINE ===\n")

rag = RAGEngine()

print("Collection Counts:")
print(rag.collection_counts())

print("\n=== TEST QUERY ===\n")

query = "hypertension"

results = rag.query(query, n_results=5)

print(f"Query: {query}\n")

print(results)

print("\n=== TEST COMPLETE ===\n")
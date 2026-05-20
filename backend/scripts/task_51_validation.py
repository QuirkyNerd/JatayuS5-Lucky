import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from config import settings
import chromadb

def validate_icd_collections():
    persist_dir = settings.chroma_persist_dir
    print(f"Checking ChromaDB at: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        print(f"❌ ERROR: ChromaDB store not found at {persist_dir}")
        return

    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    
    print(f"ICD_COLLECTION_COUNT: {len(collections)}")
    for col in collections:
        count = col.count()
        print(f"COLLECTION: {col.name} | DOC_COUNT: {count}")
        
        if count > 0:
            sample = col.peek(1)
            print(f"  SAMPLE_DOCUMENT: {sample['documents'][0] if sample['documents'] else 'EMPTY'}")
            # print(f"  SAMPLE_METADATA: {sample['metadatas'][0] if sample['metadatas'] else 'EMPTY'}")

    # Test Query
    try:
        icd_col = client.get_collection("icd10_codes")
        if icd_col and icd_col.count() > 0:
            results = icd_col.query(query_texts=["femoral neck fracture"], n_results=5)
            print("\nTOP_QUERY_RESULTS for 'femoral neck fracture':")
            for i, doc in enumerate(results['documents'][0]):
                print(f"  {i+1}. {doc} (ID: {results['ids'][0][i]})")
    except:
        print("Could not query icd10_codes")

if __name__ == "__main__":
    validate_icd_collections()

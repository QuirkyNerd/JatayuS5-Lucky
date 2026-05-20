import chromadb

CHROMA_PATH = "./backend/backend/chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

print(f"\nChecking ChromaDB at: {CHROMA_PATH}\n")

collections = [
    "icd10_codes",
    "cpt_codes",
    "coding_guidelines",
    "symptoms"
]

for name in collections:
    try:
        collection = client.get_collection(name)
        print(f"{name}: {collection.count():,} documents")
    except Exception as e:
        print(f"{name}: ERROR -> {e}")
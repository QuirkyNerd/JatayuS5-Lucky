import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncio
from services.rag_engine import RAGEngine


async def main():

    rag = RAGEngine()

    while True:

        print("\n")
        query = input("Enter Query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        print("\n" + "=" * 100)
        print("QUERY:", query)
        print("=" * 100)

        try:
            result = await rag.query(query)

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            scores = result.get("scores", [[]])[0]

            if not docs:
                print("❌ NO RESULTS")
                continue

            TOP_K = min(5, len(docs))

            for i in range(TOP_K):

                doc = docs[i]
                meta = metas[i]
                score = scores[i]

                print("\n")
                print("-" * 80)
                print(f"RESULT {i+1}")
                print("-" * 80)

                print("Score:", round(score, 3))
                print("Instruction Family:", meta.get("instruction_family"))
                print("Clinical Domain:", meta.get("clinical_domain"))
                print("Rule Focus:", meta.get("rule_focus"))

                print("\nCONTENT:")
                print(doc[:500])

        except Exception as e:
            print("❌ ERROR:", str(e))


if __name__ == "__main__":
    asyncio.run(main())
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import asyncio
from services.rag_engine import get_rag_engine

async def main():
    rag = get_rag_engine()
    client = rag.q_client
    if not client:
        print("No Qdrant client")
        return
        
    print("Checking cpt_codes collection...")
    
    # Scroll through cpt_codes
    offset = None
    found_cpt = {}
    total_cpt = 0
    while True:
        res, next_offset = client.scroll(
            collection_name="cpt_codes",
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
        total_cpt += len(res)
        for p in res:
            payload = p.payload or {}
            code = payload.get("code") or payload.get("id") or ""
            if "52332" in code or "74176" in code:
                found_cpt[code] = payload
        if not next_offset:
            break
        offset = next_offset
        
    print(f"Total CPT codes in collection: {total_cpt}")
    print(f"Found CPT matches: {found_cpt}")
    
    # Scroll through icd10_codes for N13.2
    offset = None
    found_icd = {}
    total_icd = 0
    while True:
        res, next_offset = client.scroll(
            collection_name="icd10_codes",
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
        total_icd += len(res)
        for p in res:
            payload = p.payload or {}
            code = payload.get("code") or payload.get("id") or ""
            if "N13.2" in code or "N132" in code:
                found_icd[code] = payload
        if not next_offset or total_icd > 20000:  # Cap it to avoid scrolling 111k points if we find it early or just to check
            break
        offset = next_offset
        
    print(f"Total ICD codes scrolled: {total_icd}")
    print(f"Found ICD matches: {found_icd}")

if __name__ == "__main__":
    asyncio.run(main())

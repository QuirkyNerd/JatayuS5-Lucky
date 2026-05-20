#!/usr/bin/env python
"""
Full knowledge-base ingestion for CodePerfectAuditor.

Populates ChromaDB with:
- ICD10 datasets
- CPT datasets
- Coding guidelines
- Symptom datasets

PHASE 11: INFRASTRUCTURE HARDENING
- Implements Ingestion Safety Lock.
- Prevents accidental overwrite of production KB.
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = ROOT / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from config import settings
from services.guideline_loader import GuidelineLoader
from utils.logging import get_logger

logger = get_logger("ingest_guidelines")

def safe_int(value):
    try:
        return int(value)
    except:
        return 0

async def main():
    parser = argparse.ArgumentParser(description="Ingest medical datasets into ChromaDB")
    parser.add_argument("--reset-cpt", action="store_true", help="Reset and re-ingest CPT collection")
    parser.add_argument("--reset-guidelines", action="store_true", help="Reset and re-ingest Guidelines collection")
    parser.add_argument("--reset-all", action="store_true", help="Reset and re-ingest EVERYTHING")
    parser.add_argument("--force", action="store_true", help="FORCE ingestion even if data exists (Danger)")
    parser.add_argument("--backup", action="store_true", help="Create a backup before ingestion")
    args = parser.parse_args()

    logger.info("=== CodePerfectAuditor KB Ingestion Started ===")

    try:
        loader = GuidelineLoader()
        rag = loader.rag
        
        # 1. Backup if requested (ALWAYS safe)
        if args.backup:
            rag.backup_knowledge_base()

        # 2. Knowledge State Audit (Phase 11 Task 3)
        counts = rag.collection_counts()
        icd_count = safe_int(counts.get("icd10", 0))
        cpt_count = safe_int(counts.get("cpt", 0))
        guide_count = safe_int(counts.get("guidelines", 0))
        
        has_data = icd_count > 100000 or cpt_count > 8000 or guide_count > 500
        
        if has_data and not args.force:
            logger.error("INGESTION LOCK: Production-level data detected. Refusing to run without --force.")
            print("\n[ERROR] INGESTION ABORTED: Clinical knowledge base already contains production data.")
            print("To overwrite, use: python scripts/ingest_guidelines.py --force --reset-all\n")
            sys.exit(1)

        # 3. Targeted resets if requested (Administrative Manual Only)
        if args.force:
            if args.reset_all:
                logger.warning("FORCE RESET ALL: Recreating all collections...")
                rag._icd_col = rag.recreate_collection(settings.chroma_collection_icd)
                rag._cpt_col = rag.recreate_collection(settings.chroma_collection_cpt)
                rag._guide_col = rag.recreate_collection(settings.chroma_collection_guidelines)
                rag._symptom_col = rag.recreate_collection(settings.chroma_collection_symptoms)
            elif args.reset_cpt:
                logger.warning("FORCE RESET CPT: Recreating CPT collection...")
                rag._cpt_col = rag.recreate_collection(settings.chroma_collection_cpt)
            elif args.reset_guidelines:
                logger.warning("FORCE RESET Guidelines: Recreating Guidelines collection...")
                rag._guide_col = rag.recreate_collection(settings.chroma_collection_guidelines)

        # 4. Load all (will skip non-reset collections if counts are healthy)
        final_counts = await loader.load_all()

        icd_f = safe_int(final_counts.get("icd10", 0))
        cpt_f = safe_int(final_counts.get("cpt", 0))
        guide_f = safe_int(final_counts.get("guidelines", 0))
        symptom_f = safe_int(final_counts.get("symptoms", 0))

        total = icd_f + cpt_f + guide_f + symptom_f

        print("\n[SUCCESS] INGESTION COMPLETE\n")
        print(f"ICD10 Codes Loaded       : {icd_f:,}")
        print(f"CPT Codes Loaded         : {cpt_f:,}")
        print(f"Guideline Chunks Loaded  : {guide_f:,}")
        print(f"Symptom Entries Loaded   : {symptom_f:,}")
        print("------------------------------------")
        print(f"TOTAL DOCUMENTS          : {total:,}")
        print("\nChromaDB knowledge base ready.\n")

    except Exception as e:
        logger.exception("Ingestion failed")
        print("\n[ERROR] INGESTION FAILED")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
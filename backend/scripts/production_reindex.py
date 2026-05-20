import asyncio
import os
import csv
import sys
import json
import time
from typing import List, Dict, Any

# Ensure we can import from backend
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import settings
    from services.rag_engine import get_rag_engine
    from services.normalization import CodeNormalizer
    from services.embedding_service import get_embedding_service
except ImportError:
    from backend.config import settings
    from backend.services.rag_engine import get_rag_engine
    from backend.services.normalization import CodeNormalizer
    from backend.services.embedding_service import get_embedding_service

# Setup logger
try:
    from backend.utils.logging import get_logger
    logger = get_logger("production_reindex")
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("production_reindex")

CHECKPOINT_FILE = os.path.join(project_root, "scratch", "ingestion_checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(collection, row_index):
    data = load_checkpoint()
    data[collection] = row_index
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

async def ingest_collection(rag, embedding_service, csv_path, collection_type):
    logger.info(f"--- STARTING STABILIZED INGESTION: {collection_type} ---")
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    checkpoint = load_checkpoint()
    start_row = checkpoint.get(collection_type, 0)
    
    if start_row > 0:
        logger.info(f"Resuming {collection_type} from row {start_row}")

    ids = []
    documents = []
    metadatas = []
    texts_to_embed = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    total_records = len(all_rows)
    logger.info(f"Processing {total_records} records (Skipping first {start_row})")

    # PRE-PROCESS (Fast)
    for idx, row in enumerate(all_rows):
        if idx < start_row: continue
        
        raw_code = row.get('code') or row.get('\ufeffcode') or row.get('CPT') or row.get('HCPCS')
        description = row.get('description', '') or row.get('Description', '') or row.get('LONG DESCRIPTION', '')
        if not raw_code: continue

        clean_code = CodeNormalizer.normalize(raw_code)
        display_code = CodeNormalizer.format_display(raw_code)
        
        category = "General"
        intent = "General"
        
        if collection_type == "CPT":
            if raw_code.isdigit():
                c_int = int(raw_code)
                if 10021 <= c_int <= 19999: category = "Integumentary Surgery"
                elif 20000 <= c_int <= 29999: category = "Musculoskeletal Surgery"
                elif 30000 <= c_int <= 32999: category = "Respiratory Surgery"
                elif 33000 <= c_int <= 37799: category = "Cardiovascular Surgery"
                elif 40490 <= c_int <= 49999: category = "Digestive Surgery"
                elif 50010 <= c_int <= 58999: category = "Urinary/Genital Surgery"
                elif 60000 <= c_int <= 60699: category = "Endocrine Surgery"
                elif 61000 <= c_int <= 64999: category = "Nervous System Surgery"
                elif 65091 <= c_int <= 68899: category = "Eye/Ocular Surgery"
                elif 69000 <= c_int <= 69990: category = "Auditory Surgery"
                elif 70010 <= c_int <= 79999: category = "Radiology/Imaging"
                elif 80047 <= c_int <= 89398: category = "Pathology/Laboratory"
                elif 90281 <= c_int <= 99607: category = "Medicine/E&M"
            
            if any(m in description.lower() for m in ["repair", "reconstruction", "excision", "resection", "biopsy", "incision"]):
                intent = "Surgical Intervention"
            elif any(m in description.lower() for m in ["imaging", "ultrasound", "ct scan", "mri", "x-ray"]):
                intent = "Diagnostic Imaging"
                
            doc_text = f"CPT Procedure Code: {display_code} | Category: {category} | Intent: {intent} | Description: {description}"
            
        elif collection_type == "ICD":
            has_laterality = any(m in description.lower() for m in ["left", "right", "bilateral"])
            is_acute = any(m in description.lower() for m in ["acute", "chronic", "subacute"])
            doc_text = f"ICD-10 Diagnosis Code: {display_code} | Description: {description}"
            if has_laterality: doc_text += " | Specificity: Laterality Defined"
            if is_acute: doc_text += " | Specificity: Temporal Status Defined"

        ids.append(clean_code)
        documents.append(doc_text)
        metadatas.append({
            "code": display_code,
            "type": collection_type,
            "category": category,
            "intent": intent,
            "source": os.path.basename(csv_path)
        })
        texts_to_embed.append(doc_text)

    # INGEST (Hardened)
    batch_size = 25 # Ultimate safe batch size
    total_to_ingest = len(ids)
    start_time = time.time()
    
    for i in range(0, total_to_ingest, batch_size):
        s, e = i, min(i + batch_size, total_to_ingest)
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total_to_ingest - i) / rate if rate > 0 else 0
        
        logger.info(f"[{collection_type}] Batch {i//batch_size + 1} | Row {start_row + s} | Rate: {rate:.1f} rec/s | ETA: {eta/60:.1f}m")
        
        batch_texts = texts_to_embed[s:e]
        
        max_retries = 10 # More retries
        for attempt in range(max_retries):
            try:
                embeddings = await embedding_service.embed_texts(batch_texts)
                
                if collection_type == "ICD":
                    rag.upsert_icd(ids[s:e], embeddings, documents[s:e], metadatas[s:e])
                elif collection_type == "CPT":
                    rag.upsert_cpt(ids[s:e], embeddings, documents[s:e], metadatas[s:e])
                
                # Checkpoint save
                save_checkpoint(collection_type, start_row + e)
                break
            except Exception as ex:
                wait_time = 30 * (attempt + 1) # Longer wait
                logger.warning(f"ULTIMATE PRESSURE detected on {collection_type}. Waiting {wait_time}s... Error: {str(ex)[:100]}")
                await asyncio.sleep(wait_time)
                if attempt == max_retries - 1:
                    logger.error(f"CRITICAL: Ingestion failed at row {start_row + s}")
                    raise ex
        
        # Throttling to allow metadata sync
        await asyncio.sleep(0.5)

async def run_production_reindex():
    # 1. Initialize
    rag = get_rag_engine()
    embedding_service = get_embedding_service()
    
    checkpoint = load_checkpoint()
    
    # 2. Reset ONLY if no checkpoint exists (prevents accidental wipes on resume)
    if not checkpoint:
        logger.info("Fresh Ingestion Start: Resetting all collections...")
        rag.reset_icd()
        rag.reset_cpt()
        rag.reset_guidelines()
        rag.reset_symptoms()
    else:
        logger.info(f"Resuming Ingestion: Found checkpoints: {checkpoint}")

    # 3. Main Ingestion
    icd_csv = os.path.join(settings.data_dir, "icd10_codes.csv")
    cpt_csv = os.path.join(settings.data_dir, "cpt_codes.csv")
    
    await ingest_collection(rag, embedding_service, icd_csv, "ICD")
    await ingest_collection(rag, embedding_service, cpt_csv, "CPT")
    
    # 4. Guidelines & Symptoms
    try:
        from services.guideline_loader import GuidelineLoader
        loader = GuidelineLoader()
        if not checkpoint.get("Guidelines"):
            logger.info("Ingesting Guidelines/Symptoms...")
            counts = await loader.load_all()
            save_checkpoint("Guidelines", 1)
            logger.info(f"Guideline ingestion complete: {counts}")
        else:
            logger.info("Guidelines already ingested. Skipping.")
    except Exception as e:
        logger.error(f"Failed Guidelines: {e}")

    logger.info("STABILIZED INGESTION COMPLETE.")
    # Clear checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    asyncio.run(run_production_reindex())

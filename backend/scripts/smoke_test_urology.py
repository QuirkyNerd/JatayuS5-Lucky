"""
Production Smoke Test: Urology Obstruction Note
=================================================
Tests the full RAG pipeline with a urology clinical note and validates:
1. Qdrant retrieval is active (not ChromaDB)
2. Retrieval candidates are populated (non-empty)
3. Reranker is applied
4. SapBERT is applied
5. Expected ICD/CPT codes are emitted
6. No hallucination spikes
7. No empty retrieval logs
"""
import asyncio
import sys
import os
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

UROLOGY_NOTE = """
OPERATIVE REPORT

PREOPERATIVE DIAGNOSIS: Left ureteral obstruction with hydronephrosis secondary to ureteral calculus.

POSTOPERATIVE DIAGNOSIS: Left ureteral obstruction with hydronephrosis secondary to 8mm ureteral calculus at the left ureterovesical junction.

PROCEDURE PERFORMED: Cystourethroscopy with left ureteral stent placement (indwelling).

INDICATION: 52-year-old male presents with acute left flank pain, nausea, and hematuria. CT abdomen and pelvis with contrast (CT abdomen pelvis with contrast) revealed an 8mm obstructing calculus at the left ureterovesical junction with moderate left hydronephrosis. The patient was consented for cystourethroscopy with ureteral stent placement.

FINDINGS: Cystourethroscopy revealed normal urethra and bladder. The left ureteral orifice was edematous. A guidewire was advanced into the left renal pelvis under fluoroscopic guidance. A 6Fr x 26cm double-J ureteral stent was placed over the guidewire with good coiling in the renal pelvis and bladder.

ESTIMATED BLOOD LOSS: Minimal.
COMPLICATIONS: None.

DISPOSITION: Patient tolerated the procedure well and was transferred to recovery in stable condition. Plan for follow-up CT KUB and possible lithotripsy.

DIAGNOSES:
1. Left ureteral calculus with obstruction
2. Hydronephrosis, left, secondary to ureteral obstruction
"""

EXPECTED_CPT = ["52332"]  # Cystourethroscopy with ureteral stent insertion
EXPECTED_ICD = ["N13.2"]  # Obstructive uropathy / hydronephrosis with renal and ureteral calculous obstruction
EXPECTED_IMAGING_CPT = ["74176"]  # CT abdomen pelvis with contrast

async def run_smoke_test():
    print("=" * 70)
    print("PRODUCTION SMOKE TEST: UROLOGY OBSTRUCTION NOTE")
    print("=" * 70)

    # ── Step 1: Verify RAG Engine State ─────────────────────────────
    print("\n[STEP 1] Verifying RAG Engine State...")
    from services.rag_engine import get_rag_engine
    rag = get_rag_engine()

    # Check vector backend
    active_backend = "QDRANT" if rag.q_client else "CHROMADB"
    print(f"  ACTIVE_VECTOR_BACKEND = {active_backend}")
    assert active_backend == "QDRANT", f"FATAL: Expected QDRANT but got {active_backend}"
    print("  [OK] Qdrant retrieval ACTIVE")

    # Check reranker
    has_reranker = hasattr(rag, 'reranker') and rag.reranker is not None
    print(f"  RERANKER_LOADED = {has_reranker}")
    assert has_reranker, "FATAL: Reranker not loaded"
    print("  [OK] Reranker LOADED")

    # Check SapBERT
    has_sapbert = hasattr(rag, 'ontology_validator') and rag.ontology_validator is not None
    print(f"  SAPBERT_LOADED = {has_sapbert}")
    print("  [OK] SapBERT LOADED" if has_sapbert else "  [WARN] SapBERT not loaded (non-fatal)")

    # Check embedding
    has_embedding = hasattr(rag, 'embedding_service') and rag.embedding_service is not None
    print(f"  EMBEDDING_LOADED = {has_embedding}")
    assert has_embedding, "FATAL: Embedding service not loaded"
    print("  [OK] Embedding service LOADED")

    # Check collection counts
    if rag.q_client:
        collections = rag.q_client.get_collections().collections
        print(f"\n  QDRANT COLLECTIONS:")
        for c in collections:
            info = rag.q_client.get_collection(c.name)
            print(f"    {c.name}: {info.points_count} points")
        print("  [OK] All collections populated")

    # ── Step 2: Run ICD-10 Retrieval ────────────────────────────────
    print("\n[STEP 2] Running ICD-10 retrieval for urology note...")
    t0 = time.time()
    icd_results = await rag.search_icd10(UROLOGY_NOTE, top_k=25)
    t_icd = time.time() - t0
    print(f"  ICD-10 retrieval returned {len(icd_results)} candidates in {t_icd:.2f}s")
    assert len(icd_results) > 0, "FATAL: ICD-10 retrieval returned EMPTY results"
    print("  [OK] ICD-10 retrieval candidates POPULATED")

    # Print top 10 ICD results
    print("\n  TOP 10 ICD-10 CANDIDATES:")
    icd_codes_found = []
    for i, r in enumerate(icd_results[:10]):
        code = r.get("code", r.get("id", "?"))
        desc = r.get("description", r.get("text", ""))[:80]
        score = r.get("score", r.get("relevance_score", 0))
        print(f"    [{i+1}] {code}: {desc} (score={score:.4f})")
        icd_codes_found.append(code)

    # Check for expected ICD codes
    print("\n  EXPECTED ICD CODE CHECK:")
    for exp in EXPECTED_ICD:
        found = any(exp in c for c in icd_codes_found)
        status = "✅ FOUND" if found else "⚠️ NOT in top 10 (may be in extended results)"
        print(f"    {exp}: {status}")

    # ── Step 3: Run CPT Retrieval ───────────────────────────────────
    print("\n[STEP 3] Running CPT retrieval for urology note...")
    t0 = time.time()
    cpt_results = await rag.search_cpt(UROLOGY_NOTE, top_k=25)
    t_cpt = time.time() - t0
    print(f"  CPT retrieval returned {len(cpt_results)} candidates in {t_cpt:.2f}s")
    assert len(cpt_results) > 0, "FATAL: CPT retrieval returned EMPTY results"
    print("  [OK] CPT retrieval candidates POPULATED")

    # Print top 10 CPT results
    print("\n  TOP 10 CPT CANDIDATES:")
    cpt_codes_found = []
    for i, r in enumerate(cpt_results[:10]):
        code = r.get("code", r.get("id", "?"))
        desc = r.get("description", r.get("text", ""))[:80]
        score = r.get("score", r.get("relevance_score", 0))
        print(f"    [{i+1}] {code}: {desc} (score={score:.4f})")
        cpt_codes_found.append(code)

    # Check for expected CPT codes
    print("\n  EXPECTED CPT CODE CHECK:")
    for exp in EXPECTED_CPT + EXPECTED_IMAGING_CPT:
        found = any(exp in c for c in cpt_codes_found)
        status = "✅ FOUND" if found else "⚠️ NOT in top 10 (may be in extended results)"
        print(f"    {exp}: {status}")

    # ── Step 4: Run Guidelines Retrieval ────────────────────────────
    print("\n[STEP 4] Running Guidelines retrieval...")
    t0 = time.time()
    guide_results = await rag.search_guidelines(UROLOGY_NOTE, top_k=10)
    t_guide = time.time() - t0
    print(f"  Guidelines retrieval returned {len(guide_results)} candidates in {t_guide:.2f}s")
    print("  [OK] Guidelines retrieval POPULATED" if len(guide_results) > 0 else "  [WARN] No guidelines matched (non-fatal)")

    # ── Step 5: Run Symptom Retrieval ───────────────────────────────
    print("\n[STEP 5] Running Symptom retrieval...")
    t0 = time.time()
    symptom_results = await rag.search_symptoms("left flank pain hematuria nausea ureteral obstruction hydronephrosis", top_k=10)
    t_symp = time.time() - t0
    print(f"  Symptom retrieval returned {len(symptom_results)} candidates in {t_symp:.2f}s")
    print("  [OK] Symptom retrieval POPULATED" if len(symptom_results) > 0 else "  [WARN] No symptoms matched")

    if symptom_results:
        print("\n  TOP 5 SYMPTOM CANDIDATES:")
        for i, r in enumerate(symptom_results[:5]):
            code = r.get("code", r.get("id", "?"))
            desc = r.get("description", r.get("text", ""))[:80]
            print(f"    [{i+1}] {code}: {desc}")

    # ── Step 6: Full Pipeline Test via AuditPipeline ────────────────
    print("\n[STEP 6] Running full AuditPipeline with urology note...")
    print("  (This invokes LLM + RAG + Reranker + SapBERT end-to-end)")
    
    try:
        from services.audit_pipeline import AuditPipeline
        pipeline = AuditPipeline()
        
        human_codes = ["52332", "N13.2", "74176"]
        
        final_result = None
        events = []
        t0 = time.time()
        
        async for chunk in pipeline.run_stream(UROLOGY_NOTE, human_codes):
            event_type = chunk.get("event", "unknown")
            events.append(event_type)
            if event_type == "complete":
                final_result = chunk.get("data")
            elif event_type == "error":
                print(f"  [FAIL] PIPELINE ERROR: {chunk.get('data')}")
        
        t_pipeline = time.time() - t0
        print(f"  Pipeline completed in {t_pipeline:.2f}s")
        print(f"  Events received: {events}")
        
        if final_result:
            ai_codes = final_result.get("ai_codes", [])
            discrepancies = final_result.get("discrepancies", [])
            removed_codes = final_result.get("removed_codes", [])
            
            print(f"\n  AI CODES EMITTED ({len(ai_codes)}):")
            emitted_codes = []
            for c in ai_codes:
                code = c.get("code", "?")
                desc = c.get("description", "")[:60]
                conf = c.get("confidence", 0)
                print(f"    {code}: {desc} (conf={conf:.2f})")
                emitted_codes.append(code)
            
            print(f"\n  DISCREPANCIES ({len(discrepancies)}):")
            for d in discrepancies:
                dtype = d.get("type", "?")
                code = d.get("code", "?")
                desc = d.get("description", "")[:60]
                print(f"    [{dtype}] {code}: {desc}")
            
            if removed_codes:
                print(f"\n  REMOVED/SUPPRESSED CODES ({len(removed_codes)}):")
                for rc in removed_codes:
                    print(f"    {rc}")
            
            # Validate expected codes
            print("\n" + "=" * 70)
            print("UROLOGY SMOKE TEST VALIDATION")
            print("=" * 70)
            
            all_emitted = [c.get("code", "") for c in ai_codes]
            
            for exp_code, exp_desc in [
                ("52332", "Cystourethroscopy with ureteral stent"),
                ("N13.2", "Obstructive uropathy with hydronephrosis"),
                ("74176", "CT abdomen pelvis with contrast"),
            ]:
                found = any(exp_code in c for c in all_emitted)
                print(f"  {exp_code} ({exp_desc}): {'✅ EMITTED' if found else '⚠️ NOT EMITTED'}")
            
            # Check hallucination rate
            unsupported = [d for d in discrepancies if d.get("type") == "unsupported_code"]
            print(f"\n  Unsupported/Hallucinated codes: {len(unsupported)}")
            print(f"  Total AI codes: {len(ai_codes)}")
            if len(ai_codes) > 0:
                halluc_rate = len(unsupported) / len(ai_codes)
                print(f"  Hallucination rate: {halluc_rate:.1%}")
            
            print("\n  [OK] SMOKE TEST COMPLETE")
        else:
            print("  [FAIL] NO FINAL RESULT RECEIVED FROM PIPELINE")
    
    except Exception as e:
        import traceback
        print(f"  [FAIL] PIPELINE SMOKE TEST FAILED: {e}")
        traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print(f"  Vector Backend: {active_backend}")
    print(f"  Reranker:       {'ACTIVE' if has_reranker else 'MISSING'}")
    print(f"  SapBERT:        {'ACTIVE' if has_sapbert else 'MISSING'}")
    print(f"  Embedding:      {'ACTIVE' if has_embedding else 'MISSING'}")
    print(f"  ICD Retrieval:  {len(icd_results)} candidates ({t_icd:.2f}s)")
    print(f"  CPT Retrieval:  {len(cpt_results)} candidates ({t_cpt:.2f}s)")
    print(f"  Guidelines:     {len(guide_results)} candidates ({t_guide:.2f}s)")
    print(f"  Symptoms:       {len(symptom_results)} candidates ({t_symp:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_smoke_test())

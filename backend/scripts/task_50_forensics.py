import asyncio
import json
import os
import sys

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from services.audit_pipeline import AuditPipeline
from services.evaluation_engine import _calculate_partial_credit

# Categories for classification
CATEGORIES = [
    "anatomy drift",
    "semantic sibling",
    "vague NOS variant",
    "symptom leakage",
    "chronic incidental condition",
    "unrelated procedure",
    "section contamination",
    "history contamination",
    "semantic approximation",
    "retrieval pollution"
]

CHRONIC_PREFIXES = {"I10", "E11", "E78", "I50", "N18", "J44", "I25"}

def classify_fp(pred_code, gt_set, metadata):
    pred = pred_code.upper()
    sections = [s.lower() for s in metadata.get("sections", [])]
    
    # 1. unrelated procedure
    if metadata.get("type") == "CPT":
        return "unrelated procedure"
        
    # 2. vague NOS variant
    if pred.endswith(".9") or pred.endswith(".99") or pred.endswith("0"):
        return "vague NOS variant"
        
    # 3. symptom leakage
    if pred.startswith("R"):
        return "symptom leakage"
        
    # 4. history contamination
    if any("history" in s or "pmh" in s or "past" in s for s in sections):
        return "history contamination"
        
    # 5. section contamination
    if any("indication" in s or "suspected" in s or "rule out" in s for s in sections):
        return "section contamination"
        
    # 6. chronic incidental condition
    if any(pred.startswith(pfx) for pfx in CHRONIC_PREFIXES):
        return "chronic incidental condition"
        
    # 7. semantic sibling
    if any(pred[:3] == g[:3] for g in gt_set):
        return "semantic sibling"
        
    # 8. anatomy drift
    if any(pred[0] == g[0] for g in gt_set):
        return "anatomy drift"
        
    # 9. semantic approximation
    if float(metadata.get("rag_score") or 0.0) > 0.7:
        return "semantic approximation"
        
    return "retrieval pollution"

async def run_forensics():
    benchmark_path = os.path.join(backend_dir, "data", "benchmark_expanded.json")
    with open(benchmark_path, "r") as f:
        cases = json.load(f)

    # We only need at least 5 cases
    target_cases = cases[:10]
    
    family_counts = {cat: 0 for cat in CATEGORIES}
    
    print("\n" + "="*60)
    print("PHASE 1 — CASE OUTPUT TRACE")
    print("="*60)

    for case in target_cases:
        case_id = case.get("case_id")
        gt = set(case.get("ground_truth", []))
        note = case.get("note_snippet", "")
        
        pipeline = AuditPipeline()
        ai_codes = []
        removed_codes = []
        
        async for msg in pipeline.run_stream(note, human_codes=[], ground_truth=list(gt)):
            if msg["event"] == "complete":
                ai_codes = msg["data"]["ai_codes"]
                removed_codes = msg["data"]["removed_codes"]
        
        emitted_codes = [c.get("code") for c in ai_codes]
        
        print(f"\nCASE_ID: {case_id}")
        print(f"GROUND_TRUTH: {list(gt)}")
        print(f"FINAL_EMITTED_CODES: {emitted_codes}")
        
        # Phase 2 & 3: Classification
        for c_dict in ai_codes:
            code = c_dict.get("code")
            if code not in gt:
                cat = classify_fp(code, gt, c_dict)
                family_counts[cat] += 1
                print(f"  [FP] {code} -> {cat}")

        # Phase 4: Missed Gold Trace
        missed_gold = gt - set(emitted_codes)
        for gold in missed_gold:
            # Find in removed_codes
            removed_match = next((rc for rc in removed_codes if rc.get("code") == gold), None)
            
            # Find in retrieval (if available in traces)
            # For this forensic, we'll look at all candidates that entered SelectionEngine
            print(f"  [MISS] {gold}:")
            if removed_match:
                print(f"    Final Confidence: {removed_match.get('confidence')}")
                print(f"    Suppression Reason: {removed_match.get('rejection_reason', removed_match.get('rejection_stage', 'unknown'))}")
            else:
                print(f"    Final Confidence: N/A (Never reached emission)")
                print(f"    Suppression Reason: Filtered before SelectionEngine")

    print("\n" + "="*60)
    print("PHASE 3 — FAILURE FAMILY COUNTS")
    print("="*60)
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_families:
        print(f"{cat}: {count}")

    print("\n" + "="*60)
    print("PHASE 5 — TOP FALSE POSITIVE FAMILIES")
    print("="*60)
    top_3 = sorted_families[:3]
    for i, (cat, count) in enumerate(top_3):
        print(f"{i+1}. {cat} ({count})")

if __name__ == "__main__":
    asyncio.run(run_forensics())

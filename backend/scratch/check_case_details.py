import json
import os

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])

target_cases = ["ORTHO-001", "DIABETES-001", "POSTOP-001", "CARDIO-002", "POSTOP-002"]

for case in results:
    case_id = case.get("case_id")
    if case_id not in target_cases:
        continue
        
    print(f"\n==========================================")
    print(f"CASE DETAILS: {case_id} ({case.get('category')})")
    print(f"==========================================")
    print(f"Note: {case.get('note_snippet')}")
    print(f"Ground Truth: {case.get('ground_truth')}")
    print(f"Prediction: {[p.get('code') for p in case.get('prediction_enhanced', [])]}")
    
    forensic_trace = case.get("forensic_trace", {})
    candidate_pool = forensic_trace.get("candidate_pool", [])
    
    print("\nCandidate Pool Details (top 15 by score/rag_score):")
    sorted_candidates = sorted(candidate_pool, key=lambda x: x.get("rag_score") or 0.0, reverse=True)
    for c in sorted_candidates[:15]:
        code = c.get("code")
        desc = c.get("description")
        score = c.get("rag_score")
        source = c.get("source")
        grounding = c.get("grounding")
        print(f"  - {code} ({source}): rag_score={score} | desc={desc[:60]} | grounding={grounding}")
        if c.get("retrieval_trace"):
            print(f"    retrieval_trace: {c.get('retrieval_trace')}")
        if c.get("forensic_data"):
            print(f"    forensic_data: {c.get('forensic_data')}")
        if c.get("audit_traces"):
            print(f"    audit_traces: {c.get('audit_traces')}")
            
    print("\nPrediction Enhanced details:")
    for p in case.get("prediction_enhanced", []):
        print(f"  - {p.get('code')}: score={p.get('final_score')} | reason={p.get('reasoning')} | forensic={p.get('forensic')}")

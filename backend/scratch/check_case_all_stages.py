import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    case_id = case.get("case_id")
    if case_id not in ["COMBINATION-001", "DIABETES-001"]:
        continue
        
    print(f"\n==========================================")
    print(f"TRACING ALL CODES FOR: {case_id}")
    print(f"==========================================")
    
    forensic = case.get("forensic_trace", {})
    
    # Let's print candidate pool
    print("\n--- Candidate Pool (from forensic_trace) ---")
    for c in forensic.get("candidate_pool", []):
        print(f"Code: {c.get('code')} | source: {c.get('source')} | det_score: {c.get('det_score')} | rag_score: {c.get('rag_score')} | final_score: {c.get('final_score')}")
        
    print("\n--- Grounding Rejections ---")
    for r in forensic.get("grounding_rejected", []):
        print(f"Code: {r.get('code')} | reason: {r.get('reason')}")
        
    print("\n--- Selection Rejections ---")
    for r in forensic.get("selection_rejected", []):
        print(f"Code: {r.get('code')} | score: {r.get('score')} | stage: {r.get('stage')} | reason: {r.get('reason')}")
        
    print("\n--- Terminal Rejections ---")
    for r in forensic.get("terminal_rejections", []):
        print(f"Code: {r.get('code')} | reason: {r.get('reason')}")
        
    print("\n--- Predictions Enhanced ---")
    for p in case.get("prediction_enhanced", []):
        print(f"Code: {p.get('code')} | final_score: {p.get('final_score')} | reasoning: {p.get('reasoning')}")

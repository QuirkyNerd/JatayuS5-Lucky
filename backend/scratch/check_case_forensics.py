import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    case_id = case.get("case_id")
    if case_id not in ["COMBINATION-001", "DIABETES-001"]:
        continue
        
    print(f"\n==========================================")
    print(f"CASE FORENSICS: {case_id}")
    print(f"==========================================")
    print(f"Ground Truth: {case.get('ground_truth')}")
    print(f"Predictions: {[p.get('code') for p in case.get('prediction_enhanced', [])]}")
    
    forensic = case.get("forensic_trace", {})
    
    print("\n[Candidate Pool]")
    for c in forensic.get("candidate_pool", []):
        code = c.get("code")
        source = c.get("source")
        score = c.get("final_score") or c.get("rag_score")
        print(f"  - {code} ({source}) | score={score} | desc={c.get('description')[:50]}")
        
    print("\n[Grounding Rejections]")
    for r in forensic.get("grounding_rejected", []):
        print(f"  - {r.get('code')} | reason={r.get('reason')}")
        
    print("\n[Selection Rejections]")
    for r in forensic.get("selection_rejected", []):
        print(f"  - {r.get('code')} | score={r.get('score')} | stage={r.get('stage')} | reason={r.get('reason')}")
        
    print("\n[Terminal Rejections]")
    for r in forensic.get("terminal_rejections", []):
        print(f"  - {r.get('code')} | reason={r.get('reason')}")

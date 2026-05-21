import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])

for case in results:
    case_id = case.get("case_id")
    if case_id not in ["CARDIO-002", "COMBINATION-001"]:
        continue
        
    print(f"\n==========================================")
    print(f"CASE: {case_id}")
    print(f"==========================================")
    print(f"Note: {case.get('note_snippet')}")
    print(f"Ground Truth: {case.get('ground_truth')}")
    
    forensic_trace = case.get("forensic_trace", {})
    candidate_pool = forensic_trace.get("candidate_pool", [])
    
    # Let's find the candidate objects
    for code in ["I5021", "I5020", "E1151", "E119", "E11.51", "E11.9", "I50.21", "I50.20"]:
        cand = next((c for c in candidate_pool if c.get("code", "").upper().replace(".", "").strip() == code.replace(".", "")), None)
        if cand:
            print(f"\nCode: {cand.get('code')}")
            print(f"  description: {cand.get('description')}")
            print(f"  source: {cand.get('source')}")
            print(f"  rag_score: {cand.get('rag_score')}")
            print(f"  det_score: {cand.get('det_score')}")
            print(f"  llm_score: {cand.get('llm_score')}")
            print(f"  final_score (in pool): {cand.get('score')}")
            print(f"  forensic_data: {cand.get('forensic_data')}")
            print(f"  audit_traces: {cand.get('audit_traces')}")
            
    print("\nActual Predictions:")
    for p in case.get("prediction_enhanced", []):
        print(f"  - {p.get('code')}: final_score={p.get('final_score')} | reasoning={p.get('reasoning')}")

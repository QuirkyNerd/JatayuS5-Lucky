import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    if case.get("case_id") == "COMBINATION-001":
        print("=== COMBINATION-001 ===")
        forensic_trace = case.get("forensic_trace", {})
        
        print("\nAll candidates in pool:")
        for c in forensic_trace.get("candidate_pool", []):
            print(f"  Code: {c.get('code')} | source: {c.get('source')} | rag_score: {c.get('rag_score')} | det_score: {c.get('det_score')}")
            
        print("\nSelection Engine Rejections:")
        for r in forensic_trace.get("selection_rejected", []):
            print(f"  Code: {r.get('code')} | score: {r.get('score')} | stage: {r.get('stage')} | reason: {r.get('reason')}")
            
        print("\nGrounding Rejections:")
        for r in forensic_trace.get("grounding_rejected", []):
            print(f"  Code: {r.get('code')} | reason: {r.get('reason')}")

        print("\nTerminal Rejections:")
        for r in forensic_trace.get("terminal_rejections", []):
            print(f"  Code: {r.get('code')} | reason: {r.get('reason')}")

import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    case_id = case.get("case_id")
    if case_id not in ["CARDIO-002", "COMBINATION-001"]:
        continue
    print(f"\n==========================================")
    print(f"CASE: {case_id}")
    print(f"==========================================")
    print(f"Rejected Candidates:")
    rejected = case.get("rejected", [])
    for r in rejected:
        print(f"  - {r.get('code')}: score={r.get('score')} | stage={r.get('stage')} | reason={r.get('reason')}")

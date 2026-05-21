import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    if case.get("case_id") == "COMBINATION-001":
        print(json.dumps(case.get("forensic_trace"), indent=2))

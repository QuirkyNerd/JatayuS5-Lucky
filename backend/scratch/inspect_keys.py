import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])
if results:
    case = results[0]
    print("Case keys:", list(case.keys()))
    print("Forensic trace keys:", list(case.get("forensic_trace", {}).keys()))
    print("Prediction enhanced keys:", [list(p.keys()) for p in case.get("prediction_enhanced", [])])

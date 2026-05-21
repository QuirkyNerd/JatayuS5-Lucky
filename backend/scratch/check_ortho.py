import json
report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    if case.get("case_id") == "ORTHO-001":
        print("NoteSnippet:", case.get("note_snippet"))
        print("Ground Truth:", case.get("ground_truth"))
        print("Prediction:", [p.get("code") for p in case.get("prediction_enhanced", [])])
        print("Candidates in pool:")
        for c in case.get("forensic_trace", {}).get("candidate_pool", []):
            code = c.get("code")
            desc = c.get("description")
            source = c.get("source")
            score = c.get("rag_score")
            print(f"  - {code} ({source}): rag_score={score} | desc={desc[:60]}")

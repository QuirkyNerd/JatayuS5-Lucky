import json
import os

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])
total_gold_codes = 0
gold_predicted = 0
gold_retrieved = 0
gold_not_retrieved = 0
gold_retrieved_but_not_predicted = 0

for idx, case in enumerate(results):
    gt = [c.upper().replace(".", "").strip() for c in case.get("ground_truth", [])]
    pred_enhanced = [c.get("code").upper().replace(".", "").strip() for c in case.get("prediction_enhanced", [])]
    forensic_trace = case.get("forensic_trace", {})
    candidate_pool = forensic_trace.get("candidate_pool", [])
    candidate_codes = [c.get("code", "").upper().replace(".", "").strip() for c in candidate_pool]
    
    for g in gt:
        total_gold_codes += 1
        if g in pred_enhanced:
            gold_predicted += 1
        if g in candidate_codes:
            gold_retrieved += 1
            if g not in pred_enhanced:
                gold_retrieved_but_not_predicted += 1
        else:
            gold_not_retrieved += 1

print(f"Total Gold Codes: {total_gold_codes}")
print(f"Gold Codes Predicted (Fidelity Hits): {gold_predicted} ({gold_predicted/total_gold_codes*100:.1f}%)")
print(f"Gold Codes Retrieved in Pool: {gold_retrieved} ({gold_retrieved/total_gold_codes*100:.1f}%)")
print(f"Gold Codes NOT Retrieved (Retrieval Failures): {gold_not_retrieved} ({gold_not_retrieved/total_gold_codes*100:.1f}%)")
print(f"Gold Codes Retrieved BUT NOT Predicted (Ranking/Filter Failures): {gold_retrieved_but_not_predicted} ({gold_retrieved_but_not_predicted/total_gold_codes*100:.1f}%)")

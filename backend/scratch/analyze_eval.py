import json
import os

report_path = "backend/scratch/production_eval_report.json"
if not os.path.exists(report_path):
    print(f"Error: {report_path} not found.")
    exit(1)

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])

total_cases = len(results)
total_gold_codes = 0
gold_retrieved = 0
gold_not_retrieved = 0
gold_predicted = 0
gold_retrieved_but_not_predicted = 0

ranking_failures_details = []
retrieval_failures_details = []

for idx, case in enumerate(results):
    case_id = case.get("case_id", f"case_{idx}")
    category = case.get("category", "")
    gt = [c.upper().replace(".", "").strip() for c in case.get("ground_truth", [])]
    pred_enhanced_objs = case.get("prediction_enhanced", [])
    pred_enhanced = [c.get("code").upper().replace(".", "").strip() for c in pred_enhanced_objs]
    
    forensic_trace = case.get("forensic_trace", {})
    candidate_pool = forensic_trace.get("candidate_pool", [])
    candidate_codes = [c.get("code", "").upper().replace(".", "").strip() for c in candidate_pool]
    
    for g in gt:
        total_gold_codes += 1
        
        # Check prediction
        is_pred = g in pred_enhanced
        if is_pred:
            gold_predicted += 1
            
        # Check retrieval
        is_retrieved = g in candidate_codes
        if is_retrieved:
            gold_retrieved += 1
            if not is_pred:
                gold_retrieved_but_not_predicted += 1
                # Find the candidate object and prediction object
                cand_obj = next((c for c in candidate_pool if c.get("code", "").upper().replace(".", "").strip() == g), {})
                # Find predictions
                ranking_failures_details.append({
                    "case_id": case_id,
                    "category": category,
                    "gold_code": g,
                    "candidate": cand_obj,
                    "predictions": pred_enhanced_objs,
                    "note": case.get("note_snippet", "")
                })
        else:
            gold_not_retrieved += 1
            # Find similar candidates in pool
            similar_prefix = g[:3]
            similar = [c for c in candidate_pool if c.get("code", "").upper().replace(".", "").strip().startswith(similar_prefix)]
            retrieval_failures_details.append({
                "case_id": case_id,
                "category": category,
                "gold_code": g,
                "similar_in_pool": [s.get("code") for s in similar],
                "note": case.get("note_snippet", "")
            })

print(f"Total Gold Codes: {total_gold_codes}")
print(f"Gold Codes Predicted (Fidelity Hits): {gold_predicted} ({gold_predicted/total_gold_codes*100:.1f}%)")
print(f"Gold Codes Retrieved in Pool: {gold_retrieved} ({gold_retrieved/total_gold_codes*100:.1f}%)")
print(f"Gold Codes NOT Retrieved (Retrieval Failures): {gold_not_retrieved} ({gold_not_retrieved/total_gold_codes*100:.1f}%)")
print(f"Gold Codes Retrieved BUT NOT Predicted (Ranking/Filter Failures): {gold_retrieved_but_not_predicted} ({gold_retrieved_but_not_predicted/total_gold_codes*100:.1f}%)")

print("\n" + "=" * 80)
print("ANALYZING RANKING / FILTERING FAILURES (Retrieved but not predicted)")
print("=" * 80)
for rf in ranking_failures_details[:10]:
    print(f"\nCase: {rf['case_id']} ({rf['category']})")
    print(f"Gold Code: {rf['gold_code']}")
    cand = rf['candidate']
    print(f"  Candidate details: score={cand.get('score')} or rag_score={cand.get('rag_score')} | source={cand.get('source')} | grounding={cand.get('grounding')}")
    print(f"  Candidate trace/rationale: {cand.get('rationale') or cand.get('trace')}")
    print(f"  Final Predictions in this case:")
    for p in rf['predictions']:
        print(f"    - {p.get('code')} | score={p.get('final_score')} | reason={p.get('reasoning')}")

print("\n" + "=" * 80)
print("ANALYZING RETRIEVAL FAILURES (Not in candidate pool)")
print("=" * 80)
# Group by prefix of gold code to see common families
from collections import Counter
failed_prefixes = Counter([rf['gold_code'][:3] for rf in retrieval_failures_details])
print("Common failed gold code prefixes:")
for pref, count in failed_prefixes.most_common(15):
    print(f"  {pref}: {count} times")

print("\nExamples of Retrieval Failures:")
for rf in retrieval_failures_details[:10]:
    print(f"Case: {rf['case_id']} ({rf['category']}) | Gold Code: {rf['gold_code']}")
    print(f"  Note snippet: {rf['note'][:150]}...")
    print(f"  Similar codes retrieved in pool: {rf['similar_in_pool']}")

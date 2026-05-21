import json
import os
import re

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

results = report.get("results", [])

# Categories to track
categories = {
    "wrong_laterality": 0,
    "wrong_fracture_subtype": 0,
    "wrong_encounter_extension": 0,
    "specificity_nos_confusion": 0,
    "other_ranking_failure": 0
}

details = {k: [] for k in categories.keys()}

def get_laterality(desc):
    desc = desc.lower()
    if "left" in desc: return "left"
    if "right" in desc: return "right"
    if "bilateral" in desc: return "bilateral"
    return "unspecified"

def get_encounter(code):
    clean = code.replace(".", "").strip().upper()
    if not clean: return None
    if clean[-1] in ("A", "D", "S"): return clean[-1]
    return None

def get_fracture_subtypes(desc):
    desc = desc.lower()
    subtypes = []
    for s in ["displaced", "nondisplaced", "non-displaced", "pathological", "traumatic", "stress", "open", "closed"]:
        if s in desc:
            subtypes.append(s)
    return subtypes

for idx, case in enumerate(results):
    case_id = case.get("case_id", f"case_{idx}")
    gt_codes = [c.upper().replace(".", "").strip() for c in case.get("ground_truth", [])]
    pred_codes = [c.get("code").upper().replace(".", "").strip() for c in case.get("prediction_enhanced", [])]
    
    forensic_trace = case.get("forensic_trace", {})
    candidate_pool = forensic_trace.get("candidate_pool", [])
    
    for gt_code in gt_codes:
        # We only care about ranking/filtering failures: code is in candidate pool but NOT predicted
        in_pool = any(c.get("code", "").upper().replace(".", "").strip() == gt_code for c in candidate_pool)
        in_pred = gt_code in pred_codes
        
        if in_pool and not in_pred:
            # Let's find why it was not predicted by comparing with predictions in the same case
            cand_obj = next(c for c in candidate_pool if c.get("code", "").upper().replace(".", "").strip() == gt_code)
            gt_desc = cand_obj.get("description", "").lower()
            
            # Find candidate of the same type/family that WAS predicted
            prefix3 = gt_code[:3]
            competing_preds = [p for p in case.get("prediction_enhanced", []) if p.get("code").upper().replace(".", "").strip().startswith(prefix3)]
            
            if not competing_preds:
                # No predicted code from the same 3-character prefix family
                # Check if there is any predicted code of the same type (ICD vs CPT)
                # This might be a generic recall/filtering issue
                categories["other_ranking_failure"] += 1
                details["other_ranking_failure"].append({
                    "case_id": case_id, "gt_code": gt_code, "reason": "No competing prediction in same family"
                })
                continue
                
            # Compare with competing prediction
            comp_pred = competing_preds[0]
            comp_code = comp_pred.get("code").upper().replace(".", "").strip()
            # Let's find its description in the candidate pool
            comp_cand = next((c for c in candidate_pool if c.get("code", "").upper().replace(".", "").strip() == comp_code), {})
            comp_desc = comp_cand.get("description", "").lower()
            
            failed_cat = None
            
            # 1. Laterality Check
            gt_lat = get_laterality(gt_desc)
            comp_lat = get_laterality(comp_desc)
            if gt_lat != "unspecified" and comp_lat != "unspecified" and gt_lat != comp_lat:
                failed_cat = "wrong_laterality"
                reason = f"GT laterality is {gt_lat}, predicted {comp_code} with laterality {comp_lat}"
                
            # 2. Fracture Subtype Check
            if not failed_cat:
                gt_frac = get_fracture_subtypes(gt_desc)
                comp_frac = get_fracture_subtypes(comp_desc)
                if set(gt_frac) != set(comp_frac) and (gt_frac or comp_frac):
                    failed_cat = "wrong_fracture_subtype"
                    reason = f"GT fracture subtypes {gt_frac}, predicted {comp_code} with subtypes {comp_frac}"
            
            # 3. Encounter Check
            if not failed_cat:
                gt_enc = get_encounter(gt_code)
                comp_enc = get_encounter(comp_code)
                if gt_enc and comp_enc and gt_enc != comp_enc:
                    failed_cat = "wrong_encounter_extension"
                    reason = f"GT encounter is {gt_enc}, predicted {comp_code} with encounter {comp_enc}"
                    
            # 4. Specificity / NOS confusion
            if not failed_cat:
                # Check if predicted code is shorter or has "unspecified" or "NOS" in description
                is_comp_generic = len(comp_code) < len(gt_code) or "unspecified" in comp_desc or "nos" in comp_desc
                if is_comp_generic:
                    failed_cat = "specificity_nos_confusion"
                    reason = f"Predicted generic/shorter code {comp_code} ('{comp_desc[:40]}') instead of specific gold code {gt_code} ('{gt_desc[:40]}')"
                    
            if not failed_cat:
                failed_cat = "other_ranking_failure"
                reason = f"Competitor {comp_code} outranked gold {gt_code}"
                
            categories[failed_cat] += 1
            details[failed_cat].append({
                "case_id": case_id,
                "gt_code": gt_code,
                "comp_code": comp_code,
                "reason": reason
            })

print("=" * 80)
print("RANKING / FILTER FAILURE TAXONOMY")
print("=" * 80)
for cat, count in categories.items():
    print(f"{cat.upper():<30} : {count} occurrences ({count/sum(categories.values())*100:.1f}%)")

print("\n" + "=" * 80)
print("DETAILED EXAMPLES")
print("=" * 80)
for cat in categories.keys():
    print(f"\n--- {cat.upper()} ({len(details[cat])} cases) ---")
    for d in details[cat][:5]:
        print(f"Case {d['case_id']}: Gold {d['gt_code']} lost to {d.get('comp_code', 'None')} | Reason: {d['reason']}")

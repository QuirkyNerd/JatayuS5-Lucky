import json
import re
from collections import Counter

def run_analysis():
    report_path = "backend/scratch/production_eval_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    results = report.get("results", [])
    
    # 1. Hallucination families
    # A hallucination is a predicted code not in ground truth
    hallucinated_codes = []
    for res in results:
        gt = {c.upper().replace(".", "").strip() for c in res.get("ground_truth", [])}
        preds = {p.get("code").upper().replace(".", "").strip() for p in res.get("prediction_enhanced", [])}
        halls = preds - gt
        for h in halls:
            hallucinated_codes.append(h)
            
    hall_families = Counter([c[:3] for c in hallucinated_codes])
    
    # 2. Specificity Errors
    # Gold code is more specific (longer) than predicted code in same family
    specificity_errors = []
    # 3. Laterality conflicts
    laterality_mismatches = []
    # 4. Fracture Subtype Ranking
    fracture_subtype_cases = []
    
    def get_laterality(desc):
        desc = desc.lower()
        if "left" in desc: return "left"
        if "right" in desc: return "right"
        if "bilateral" in desc: return "bilateral"
        return "unspecified"
        
    def get_fracture_subtypes(desc):
        desc = desc.lower()
        subtypes = []
        for s in ["displaced", "nondisplaced", "non-displaced", "open", "closed"]:
            if s in desc:
                subtypes.append(s)
        return subtypes

    for res in results:
        case_id = res.get("case_id")
        gt = {c.upper().replace(".", "").strip() for c in res.get("ground_truth", [])}
        preds = {p.get("code").upper().replace(".", "").strip() for p in res.get("prediction_enhanced", [])}
        
        cand_pool = res.get("forensic_trace", {}).get("candidate_pool", [])
        cands_by_code = {c.get("code", "").upper().replace(".", "").strip(): c for c in cand_pool}
        
        # Look for specificity issues
        for gt_code in gt:
            if gt_code not in preds:
                # Find if a shorter prefix code was predicted
                prefix3 = gt_code[:3]
                competing_preds = [p for p in preds if p.startswith(prefix3)]
                if competing_preds:
                    comp_code = competing_preds[0]
                    gt_cand = cands_by_code.get(gt_code, {})
                    comp_cand = cands_by_code.get(comp_code, {})
                    
                    gt_desc = gt_cand.get("description", "Unknown")
                    comp_desc = comp_cand.get("description", "Unknown")
                    
                    if len(comp_code) < len(gt_code):
                        specificity_errors.append({
                            "case_id": case_id,
                            "gt_code": gt_code,
                            "gt_desc": gt_desc,
                            "pred_code": comp_code,
                            "pred_desc": comp_desc
                        })
                        
                    # Check for laterality mismatch
                    gt_lat = get_laterality(gt_desc)
                    comp_lat = get_laterality(comp_desc)
                    if gt_lat != "unspecified" and comp_lat != "unspecified" and gt_lat != comp_lat:
                        laterality_mismatches.append({
                            "case_id": case_id,
                            "gt_code": gt_code,
                            "gt_lat": gt_lat,
                            "pred_code": comp_code,
                            "pred_lat": comp_lat
                        })
                        
                    # Check for fracture subtype conflicts
                    gt_frac = get_fracture_subtypes(gt_desc)
                    comp_frac = get_fracture_subtypes(comp_desc)
                    if set(gt_frac) != set(comp_frac) and (gt_frac or comp_frac):
                        fracture_subtype_cases.append({
                            "case_id": case_id,
                            "gt_code": gt_code,
                            "gt_frac": gt_frac,
                            "pred_code": comp_code,
                            "pred_frac": comp_frac
                        })
                        
    # 5. Family penalty effects
    # Look for candidates that had family penalties applied and see how they ranked
    family_penalty_applications = []
    for res in results:
        case_id = res.get("case_id")
        cand_pool = res.get("forensic_trace", {}).get("candidate_pool", [])
        for c in cand_pool:
            trace = c.get("retrieval_trace") or {}
            fp = trace.get("family_penalty", 0.0)
            if fp < 0.0:
                family_penalty_applications.append({
                    "case_id": case_id,
                    "code": c.get("code"),
                    "penalty": fp,
                    "final_score": trace.get("final_score")
                })
                
    print("==================================================================")
    print("DETAILED VERIFICATION ANALYSIS")
    print("==================================================================")
    print(f"Total cases evaluated: {len(results)}")
    
    print("\n1. Top Hallucinated Code Families:")
    for fam, count in hall_families.most_common(10):
        print(f"  Family {fam}: {count}x")
        
    print(f"\n2. Specificity Errors found: {len(specificity_errors)}")
    for err in specificity_errors[:5]:
        print(f"  Case {err['case_id']}: Gold {err['gt_code']} ('{err['gt_desc']}') lost to predicted {err['pred_code']} ('{err['pred_desc']}')")
        
    print(f"\n3. Laterality Conflicts found: {len(laterality_mismatches)}")
    for err in laterality_mismatches[:5]:
        print(f"  Case {err['case_id']}: Gold {err['gt_code']} ({err['gt_lat']}) vs Predicted {err['pred_code']} ({err['pred_lat']})")
        
    print(f"\n4. Fracture Subtype Ranking conflicts: {len(fracture_subtype_cases)}")
    for err in fracture_subtype_cases[:5]:
        print(f"  Case {err['case_id']}: Gold {err['gt_code']} {err['gt_frac']} vs Predicted {err['pred_code']} {err['pred_frac']}")
        
    print(f"\n5. Family Penalty applications analyzed: {len(family_penalty_applications)}")
    print(f"  Average family penalty applied: {sum(x['penalty'] for x in family_penalty_applications)/max(1, len(family_penalty_applications)):.3f}")
    print(f"  Clamped family penalties (at -0.08 limit): {sum(1 for x in family_penalty_applications if x['penalty'] == -0.08)}")
    for app in family_penalty_applications[:5]:
        print(f"  Case {app['case_id']}: Code {app['code']} received penalty {app['penalty']} | final score = {app['final_score']}")

if __name__ == "__main__":
    run_analysis()

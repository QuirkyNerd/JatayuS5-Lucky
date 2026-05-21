import json
import os
import sys

def analyze():
    report_path = "backend/scratch/production_eval_report.json"
    if not os.path.exists(report_path):
        print(f"Error: report not found at {report_path}")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Total cases: {len(results)}")
    
    # Global counters
    total_gold_codes = 0
    total_missed_gold = 0
    missed_not_retrieved = 0
    missed_rejected_selection = 0
    missed_rejected_terminal = 0
    
    # Categorization of rejections
    selection_rejection_reasons = {}
    terminal_rejection_reasons = {}
    
    # List of detailed failures
    detailed_failures = []
    
    for case in results:
        case_id = case.get("case_id", "?")
        category = case.get("category", "unknown")
        note_snippet = case.get("note_snippet", "")
        
        # Ground truth
        gt = {c.replace(".", "").strip().upper() for c in case.get("ground_truth", [])}
        total_gold_codes += len(gt)
        
        # Emitted codes
        emitted = {c.get("code").replace(".", "").strip().upper() for c in case.get("prediction_enhanced", []) if c.get("code")}
        
        missed = gt - emitted
        total_missed_gold += len(missed)
        
        if not missed:
            continue
            
        # Analyze each missed code
        for gold in missed:
            # Let's check where it got lost
            # 1. Did it enter RAG candidates at all?
            ai_codes_full = case.get("_ai_codes_full", [])
            in_full = next((c for c in ai_codes_full if c.get("code", "").replace(".", "").strip().upper() == gold), None)
            
            # 2. Was it in selection_rejected?
            sel_rejected = case.get("forensic_trace", {}).get("selection_rejected", [])
            in_sel_rejected = next((c for c in sel_rejected if c.get("code", "").replace(".", "").strip().upper() == gold), None)
            
            # 3. Was it in terminal_rejections?
            term_rejections = case.get("forensic_trace", {}).get("terminal_rejections", [])
            in_term_rejections = next((c for c in term_rejections if c.get("code", "").replace(".", "").strip().upper() == gold), None)
            
            status = "unknown"
            reason = "not_retrieved"
            forensic = {}
            score = None
            
            if in_full:
                status = "retrieved_but_not_emitted"
                score = in_full.get("confidence") or in_full.get("final_score")
                forensic = in_full.get("forensic") or {}
                
            if in_sel_rejected:
                status = "selection_rejected"
                score = in_sel_rejected.get("score")
                reason = in_sel_rejected.get("reason", "unknown")
                stage = in_sel_rejected.get("stage", "unknown")
                selection_rejection_reasons[reason] = selection_rejection_reasons.get(reason, 0) + 1
                missed_rejected_selection += 1
                
            elif in_term_rejections:
                status = "terminal_validator_rejected"
                score = in_term_rejections.get("actual_score")
                reason = in_term_rejections.get("rejection_reason", "unknown")
                stage = in_term_rejections.get("rejection_stage", "unknown")
                terminal_rejection_reasons[reason] = terminal_rejection_reasons.get(reason, 0) + 1
                missed_rejected_terminal += 1
                # Try to grab forensic from candidate if available
                matching_full = next((c for c in ai_codes_full if c.get("code", "").replace(".", "").strip().upper() == gold), None)
                if matching_full:
                    forensic = matching_full.get("forensic") or {}
            else:
                if not in_full:
                    missed_not_retrieved += 1
                    status = "never_retrieved"
                    reason = "RAG retrieval missed it completely"
            
            detailed_failures.append({
                "case_id": case_id,
                "category": category,
                "gold_code": gold,
                "status": status,
                "reason": reason,
                "score": score,
                "forensic": forensic,
                "note_snippet": note_snippet
            })
            
    print(f"\n--- GLOBAL SUMMARY OF MISSED GOLD CODES ---")
    print(f"Total Gold Codes in dataset: {total_gold_codes}")
    print(f"Total Missed Gold Codes: {total_missed_gold} ({total_missed_gold/total_gold_codes:.1%})")
    print(f"  - Never Retrieved by RAG: {missed_not_retrieved} ({missed_not_retrieved/total_missed_gold:.1%})")
    print(f"  - Rejected by SelectionEngine: {missed_rejected_selection} ({missed_rejected_selection/total_missed_gold:.1%})")
    print(f"  - Rejected by Final Validator / Governance: {missed_rejected_terminal} ({missed_rejected_terminal/total_missed_gold:.1%})")
    
    print("\n--- SELECTION ENGINE REJECTION REASONS ---")
    for r, count in sorted(selection_rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:3d} x {r}")
        
    print("\n--- TERMINAL VALIDATOR REJECTION REASONS ---")
    for r, count in sorted(terminal_rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:3d} x {r}")
        
    # Write detailed failures to scratch file for further analysis
    out_path = "backend/scratch/detailed_failures_report.json"
    with open(out_path, "w") as f:
        json.dump(detailed_failures, f, indent=2)
    print(f"\nSaved detailed failures to: {out_path}")

if __name__ == "__main__":
    analyze()

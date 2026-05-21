import json

report_path = "backend/scratch/production_eval_report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for case in report.get("results", []):
    if case.get("case_id") == "COMBINATION-001":
        print(f"=== COMBINATION-001 ===")
        gt = case.get("ground_truth")
        print("Ground Truth:", gt)
        print("Prediction Enhanced:", [p.get("code") for p in case.get("prediction_enhanced", [])])
        
        forensic_trace = case.get("forensic_trace", {})
        
        # Check Candidate Pool
        cand_pool = forensic_trace.get("candidate_pool", [])
        print(f"In Candidate Pool: {any(c.get('code') in ['E1151', 'E11.51'] for c in cand_pool)}")
        for c in cand_pool:
            if c.get('code') in ['E1151', 'E11.51']:
                print("Candidate pool item:", c)
                
        # Check Grounding Rejected
        gr_rej = forensic_trace.get("grounding_rejected", [])
        print(f"In Grounding Rejected: {any(c.get('code') in ['E1151', 'E11.51'] for c in gr_rej)}")
        for r in gr_rej:
            if r.get('code') in ['E1151', 'E11.51']:
                print("Grounding rejected item:", r)
                
        # Check Selection Rejected
        sel_rej = forensic_trace.get("selection_rejected", [])
        print(f"In Selection Rejected: {any(r.get('code') in ['E1151', 'E11.51'] for r in sel_rej)}")
        for r in sel_rej:
            if r.get('code') in ['E1151', 'E11.51']:
                print("Selection rejected item:", r)
                
        # Check Terminal Rejections
        t_rej = forensic_trace.get("terminal_rejections", [])
        print(f"In Terminal Rejections: {any(r.get('code') in ['E1151', 'E11.51'] for r in t_rej)}")
        for r in t_rej:
            if r.get('code') in ['E1151', 'E11.51']:
                print("Terminal rejected item:", r)

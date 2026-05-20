import json
import logging
from pprint import pprint

def analyze_suppression_budget(ai_codes, rejected_codes):
    """
    Computes average penalty magnitude per stage.
    """
    stage_impacts = {}
    
    for c in ai_codes + rejected_codes:
        history = c.get("contribution_history", [])
        for step in history:
            stage = step.get("stage", "unknown")
            delta = step.get("delta", 0)
            if delta < 0:
                if stage not in stage_impacts:
                    stage_impacts[stage] = []
                stage_impacts[stage].append(delta)
                
    budget_report = {}
    for stage, deltas in stage_impacts.items():
        budget_report[stage] = {
            "avg_penalty": sum(deltas) / len(deltas),
            "total_hits": len(deltas)
        }
        
    return budget_report

def build_confidence_waterfall(code_dict):
    """
    Constructs the requested CONFIDENCE_WATERFALL trace.
    """
    history = code_dict.get("contribution_history", [])
    
    waterfall = {
        "INITIAL_SCORE": code_dict.get("base_evidence_strength", 0.5),
        "STAGES": [],
        "FINAL_CONFIDENCE": code_dict.get("confidence", 0)
    }
    
    current = waterfall["INITIAL_SCORE"]
    for step in history:
        stage = step.get("stage", "unknown")
        delta = step.get("delta", 0)
        current += delta
        waterfall["STAGES"].append({
            "stage": stage,
            "delta": delta,
            "running_score": round(current, 3)
        })
        
    code_dict["CONFIDENCE_WATERFALL"] = waterfall
    return waterfall

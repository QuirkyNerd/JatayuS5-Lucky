"""
services/debugging_engine.py – Reasoning Path Reconstruction and Failure Replay Engine.

RESPONSIBILITIES:
  1. Reconstructs the exact reasoning path of a failed prediction.
  2. Replays governance layer decisions for deterministic failure analysis.
  3. Visualizes confidence evolution and evidence arbitration timelines.
"""

import logging
import json

logger = logging.getLogger(__name__)

def replay_failed_reasoning_path(code_dict: dict, audit_log: list[str]) -> dict:
    """
    Step 3 — Trace-Based Failure Replay.
    Deterministic reconstruction of the failed reasoning path.
    """
    path = {
        "code": code_dict.get("code"),
        "final_confidence": float(code_dict.get("confidence") or 0),
        "layers_executed": [],
        "suppression_points": [],
        "arbitration_outcome": "RETAINED" if float(code_dict.get("evidence_strength") or 0) > 0.4 else "SUPPRESSED"
    }
    
    # Reconstruct from audit traces
    traces = code_dict.get("audit_traces", [])
    for t in traces:
        if "SUPPRESSED" in t or "FAILED" in t:
            path["suppression_points"].append(t)
        path["layers_executed"].append(t)
        
    return path

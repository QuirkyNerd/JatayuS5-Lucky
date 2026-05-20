import re
import logging
from .validation_utils import (
    detect_text_corruption_patterns,
    detect_copy_forward_artifacts,
    compute_abbreviation_disambiguation_confidence,
    resolve_negation_scope,
    compute_section_reliability_weight,
    stabilize_clinical_entity_boundaries
)

logger = logging.getLogger(__name__)

def apply_text_repair_heuristics(text: str) -> str:
    """
    Step 3 — OCR & Text Corruption Detection (Task 16).
    Repairs common recoverable corruption patterns.
    """
    if not text: return text
    
    repaired = text
    # 1. Split merged camelCase (often OCR error)
    repaired = re.sub(r'([a-z])([A-Z])', r'\1 \2', repaired)
    # 2. Fix broken medication dosages (e.g. "50mg" -> "50 mg")
    repaired = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', repaired)
    # 3. Normalize common OCR substitutions in context
    # (Only if corruption pattern was detected)
    if detect_text_corruption_patterns(text) > 0.3:
        repaired = repaired.replace("1LL", "ILL")
        repaired = repaired.replace("0F", "OF")
        
    return repaired


def apply_input_reliability_reconciliation(
    raw_note: str, 
    sections: dict[str, str], 
    history: list[str] = None
) -> dict:
    """
    Step 7 — Input Reliability Reconciliation (Task 16).
    Orchestrates the normalization pipeline before reasoning execution.
    """
    history = history or []
    reconciled_sections = {}
    
    for sec_name, content in sections.items():
        # 1. Detect Corruption & Repair
        repaired_content = apply_text_repair_heuristics(content)
        
        # 2. Section Reliability Weight
        weight = compute_section_reliability_weight(sec_name)
        
        # 3. Detect Copy-Forward Artifacts
        artifact_score = detect_copy_forward_artifacts(repaired_content, history)
        
        reconciled_sections[sec_name] = {
            "content": repaired_content,
            "reliability_weight": weight,
            "artifact_score": artifact_score,
            "is_stale": artifact_score > 0.8
        }
        
    logger.info("INPUT_RELIABILITY_RECONCILIATION_APPLIED")
    
    return {
        "reconciled_sections": reconciled_sections,
        "normalization_traces": ["INPUT_RELIABILITY_RECONCILIATION_APPLIED", "NORMALIZATION_PIPELINE_FINALIZED"]
    }

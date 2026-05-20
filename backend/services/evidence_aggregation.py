"""
services/evidence_aggregation.py – Evidence Aggregation Layer for Clinical Signal Fusion.

PURPOSE:
  Merge evidence across the entire note for each candidate.
  Candidates accumulate mentions, anatomy overlap, procedure linkage, 
  symptom support, and retrieval recurrence.
"""

import re
from typing import List, Dict, Any
from utils.logging import get_logger
from services.validation_utils import (
    extract_anatomy_regions,
    get_code_anatomy,
    clean_rag_description,
    ENCOUNTER_DOMAINS
)

logger = get_logger(__name__)

class EvidenceAggregationEngine:
    """
    🚨 TASK 47 — HIGH-FIDELITY EVIDENCE FEATURE EXTRACTION.
    Produces high-resolution clinical signals (0.6-0.95 for strong, 0.05-0.25 for weak).
    """
    def __init__(self):
        self.medical_priority_tokens = {
            "fracture", "intertrochanteric", "arthroplasty", "dialysis", "cabg",
            "stemi", "sepsis", "osteomyelitis", "femur", "lumbar", "coronary",
            "infarction", "failure", "stroke", "pneumonia", "colectomy", "bypass",
            "aneurysm", "hemorrhage", "stenosis", "clogged", "blocked", "acute"
        }
        self.abbreviations = {
            "stemi": "st-elevation myocardial infarction",
            "cabg": "coronary artery bypass graft",
            "esrd": "end stage renal disease",
            "cad": "coronary artery disease",
            "htn": "hypertension",
            "hld": "hyperlipidemia",
            "dm": "diabetes mellitus",
            "aki": "acute kidney injury",
            "ckd": "chronic kidney disease",
            "hf": "heart failure",
            "tia": "transient ischemic attack"
        }

    def aggregate(self, candidates: List[Dict[str, Any]], note_text: str) -> List[Dict[str, Any]]:
        if not candidates: return []

        note_lower = note_text.lower()
        note_anatomy = extract_anatomy_regions(note_lower)
        active_codes = {c["code"].upper() for c in candidates}

        aggregated: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            code = cand["code"].upper()
            if code not in aggregated:
                aggregated[code] = cand.copy()
                agg = aggregated[code]
                agg["recurrence_count"] = 1
                agg["source_queries"] = {cand.get("entity", "unknown")}
                agg["sections"] = {cand.get("section", "unspecified")}
                agg["support_signals"] = []
            else:
                agg = aggregated[code]
                agg["recurrence_count"] += 1
                agg["source_queries"].add(cand.get("entity", "unknown"))
                agg["sections"].add(cand.get("section", "unspecified"))
                agg["confidence"] = max(agg["confidence"], cand.get("confidence", 0.0))
                agg["rag_score"] = max(agg["rag_score"], cand.get("rag_score", 0.0))

        for code, agg in aggregated.items():
            desc = agg.get("description", "").lower()
            code_type = agg.get("type", "ICD-10")

            # Phase 1 & 2: Terminology Feature Extraction
            agg["terminology_overlap"] = self._compute_terminology_score(desc, note_lower)

            # Phase 3: Anatomy Feature Extraction
            agg["anatomy_overlap"] = self._compute_anatomy_score(code, desc, note_anatomy, note_lower)

            # Phase 4: Procedure Feature Extraction
            agg["procedure_linkage"] = self._compute_procedure_score(code, desc, active_codes, note_lower)

            # Phase 5: Section Feature Extraction
            agg["section_authority"] = self._compute_section_score(agg["sections"])

            # Recurrence Feature
            agg["recurrence_score"] = min(1.0, (len(agg["source_queries"]) - 1) * 0.35)

            # Phase 6: Normalization Trace (Internal aggregate score for legacy compatibility)
            # Note: Final score is handled by SelectionEngine reconstruction formula (Task 46)
            agg["aggregation_score"] = (
                agg["terminology_overlap"] * 0.35 +
                agg["anatomy_overlap"] * 0.25 +
                agg["procedure_linkage"] * 0.20 +
                agg["section_authority"] * 0.15 +
                agg["recurrence_score"] * 0.05
            )

        result = list(aggregated.values())
        for agg in result:
            agg["source_queries"] = list(agg["source_queries"])
            agg["sections"] = list(agg["sections"])
        return result

    def _compute_terminology_score(self, desc: str, note_text: str) -> float:
        """
        🚨 TASK 47 — PHASE 1: TERMINOLOGY RECONSTRUCTION.
        High-fidelity phrase and token matching.
        """
        desc_clean = re.sub(r"[^a-z0-9\s]", " ", desc.lower())
        
        # 1. Exact Phrase Matching (High Resolution)
        # Extract meaningful multi-token phrases from description
        phrases = [p.strip() for p in desc_clean.split(",") if len(p.strip().split()) > 1]
        if any(p in note_text for p in phrases):
            return 0.95 # Dominant phrase match

        # 2. Medical Token Priority Matching
        words = [w for w in desc_clean.split() if len(w) > 3]
        if not words: return 0.0

        score = 0.0
        match_count = 0
        for w in words:
            weight = 2.5 if w in self.medical_priority_tokens else 1.0
            if w in note_text:
                score += weight
                match_count += 1
            # Check abbreviations
            elif any(abbr == w and full in note_text for abbr, full in self.abbreviations.items()):
                score += weight
                match_count += 1

        max_possible = sum(2.5 if w in self.medical_priority_tokens else 1.0 for w in words)
        raw_overlap = score / max_possible

        # Normalization (Phase 6)
        if raw_overlap > 0.7: return 0.85 + (raw_overlap - 0.7) * 0.33
        if raw_overlap > 0.4: return 0.60 + (raw_overlap - 0.4) * 0.83
        return raw_overlap * 0.5

    def _compute_anatomy_score(self, code: str, desc: str, note_anatomy: set, note_text: str) -> float:
        """
        🚨 TASK 47 — PHASE 3: ANATOMY EXTRACTION IMPROVEMENT.
        """
        code_anatomy = get_code_anatomy(code, desc)
        if not code_anatomy: return 0.10 # Weak incidental
        if not note_anatomy: return 0.0

        # Body Region Match
        overlap = code_anatomy & note_anatomy
        if not overlap: return 0.05

        # Side Match (Laterality)
        side_score = 0.0
        desc_lower = desc.lower()
        if ("left" in desc_lower and "left" in note_text) or ("right" in desc_lower and "right" in note_text):
            side_score = 0.40
        elif ("left" in desc_lower and "right" in note_text) or ("right" in desc_lower and "left" in note_text):
            return 0.10 # Mismatch penalty through low resolution

        base_score = 0.55 if len(overlap) > 0 else 0.0
        return min(0.95, base_score + side_score)

    def _compute_procedure_score(self, icd_code: str, icd_desc: str, active_codes: set, note_text: str) -> float:
        """
        🚨 TASK 47 — PHASE 4: PROCEDURE LINK EXTRACTION.
        """
        linkage_rules = {
            "27130": ["M16", "S72", "hip", "femur", "arthroplasty"],
            "27447": ["M17", "S82", "knee", "arthroplasty"],
            "47562": ["K80", "K81", "cholecyst"],
            "33533": ["I25", "cad", "bypass", "cabg"],
            "92928": ["I21", "stemi", "pci", "stent", "coronary"],
            "90935": ["N18.6", "esrd", "dialysis"],
            "58150": ["D25", "N85", "uterus", "fibroid", "hysterectomy"],
            "44140": ["K57", "K56", "colon", "diverticulitis", "colectomy"],
            "27244": ["S72.1", "intertrochanteric", "orif"]
        }

        for proc_code, triggers in linkage_rules.items():
            if proc_code in active_codes:
                if any(t in icd_code or t in icd_desc.lower() for t in triggers):
                    return 0.90 # Strong linkage
            # Check if procedure keyword is in note even if code is missing (Phase 4 recovery)
            if any(t in note_text for t in triggers if len(t) > 3) and any(t in icd_desc.lower() for t in triggers):
                return 0.65 # Moderate linkage via keyword
        
        return 0.05 # Weak incidental

    def _compute_section_score(self, sections: set) -> float:
        """
        🚨 TASK 47 — PHASE 5: SECTION PHRASE DETECTION.
        """
        high_auth = {"operative diagnosis", "postop diagnosis", "postoperative diagnosis", "final diagnosis", "principal diagnosis", "discharge diagnosis", "impression", "assessment", "plan"}
        mod_auth = {"findings", "indication", "chief complaint", "reason for visit"}
        
        sections_lower = {s.lower() for s in sections}
        if sections_lower & high_auth: return 0.95
        if sections_lower & mod_auth: return 0.65
        if "history" in str(sections_lower): return 0.25
        return 0.10

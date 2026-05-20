"""
services/validation_utils.py – Shared Clinical Evidence and Negation Utilities.

RESPONSIBILITIES:
  1. Centralized negation and prophylaxis context detection.
  2. Clinical evidence strength scoring and tier-based calibration.
  3. Anatomical region extraction and consistency verification.
  4. ICD-10/CPT specificity scoring and hierarchy resolution.
"""

from __future__ import annotations

import re
import logging
import time
import copy
import traceback

logger = logging.getLogger(__name__)

def clean_rag_description(raw: str) -> str:
    """Strip 'Code: XX | Description: ' noise from ChromaDB document text."""
    # Remove patterns like "Code: E11.9 | Description: " or "Code: E11.9 |"
    cleaned = re.sub(r"(?i)code:\s*[A-Z0-9.]+\s*\|?\s*(?:description:)?\s*", "", raw)
    # Remove leading/trailing whitespace and pipes
    cleaned = cleaned.strip().strip("|").strip()
    return cleaned if cleaned else raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL INDICATORS  (Generalized)
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_INDICATORS = frozenset([
    "prior", "history of", "hx of", "previous", "resolved", "remote",
    "years ago", "status post", "s/p", "old", "former", "childhood",
    "historical", "pre-existing", "baseline", "at baseline"
])

ACTIVE_INDICATORS = frozenset([
    "acute", "active", "current", "ongoing", "admitted for", "diagnosed with",
    "confirmed", "undergoing treatment", "presents with", "complains of",
    "new onset", "exacerbation", "worsening", "evidence of"
])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION AUTHORITY  (Generalized)
# ─────────────────────────────────────────────────────────────────────────────
HIGH_AUTHORITY_SECTIONS = frozenset([
    "assessment", "impression", "plan", "final diagnosis",
    "discharge diagnosis", "operative diagnosis", "procedure diagnosis",
    "chief complaint", "reason for visit", "preoperative diagnosis",
    "postoperative diagnosis", "preop diagnosis", "postop diagnosis",
    "operative diagnosis", "pre-operative diagnosis", "post-operative diagnosis",
    "findings", "procedure note"
])

HISTORY_SECTIONS = frozenset([
    "pmh", "past medical history", "history", "family history",
    "medication history", "social history", "surgical history",
    "prior history"
])

# ─────────────────────────────────────────────────────────────────────────────
# CHRONIC PROTECTION SET (ICD-10 Prefix Families)
# ─────────────────────────────────────────────────────────────────────────────
CHRONIC_MANAGED_PREFIXES = frozenset([
    "E11", "E10", "E13",         # Diabetes
    "I10", "I11", "I12", "I13",  # Hypertension
    "N18",                       # CKD
    "I50",                       # Heart Failure
    "J44", "J45",                # COPD / Asthma
    "E78",                       # Dyslipidemia
    "G40",                       # Epilepsy
])

def detect_temporal_status(text: str, section: str = "") -> str:
    """
    Generalized temporal status detection based on phrases and section.
    Returns: 'ACTIVE', 'HISTORICAL', 'RESOLVED', 'CHRONIC_MANAGED'
    """
    text_lower = text.lower()
    section_lower = section.lower()

    # Priority 1: Direct indicator phrases using word boundaries to prevent substring matches
    is_historical = any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in HISTORICAL_INDICATORS)
    is_active = any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in ACTIVE_INDICATORS)
    is_resolved = "resolved" in text_lower or "hx of" in text_lower

    # Priority 2: Section Authority
    in_history_section = any(s in section_lower for s in HISTORY_SECTIONS)
    in_active_section = any(s in section_lower for s in HIGH_AUTHORITY_SECTIONS)

    if is_resolved:
        return "RESOLVED"
    
    if in_active_section:
        # High authority section promotes to active unless explicitly historical
        return "HISTORICAL" if is_historical and not is_active else "ACTIVE"

    if in_history_section:
        # History section is historical unless explicitly active (e.g. "active htn")
        return "ACTIVE" if is_active else "HISTORICAL"

    # Fallback
    if is_active: return "ACTIVE"
    if is_historical: return "HISTORICAL"
    return "ACTIVE"


def calculate_soft_fusion_confidence(code_dict: dict) -> float:
    """
    Task 10.4: Soft Evidence Fusion logic.
    Aggregates multiple confidence signals into a unified trust value.
    This replaces hard cliffs with weighted accumulation.
    """
    conf         = float(code_dict.get("confidence") or code_dict.get("final_score") or 0.5)
    is_converged = code_dict.get("converged", False)
    sec_priority = code_dict.get("section_priority", 3)
    is_protected = bool(
        code_dict.get("protected") 
        or code_dict.get("source") == "deterministic"
        or code_dict.get("grounding") == "deterministic"
    )
    
    if is_protected:
        return 1.0

    # Base trust starts with raw confidence
    trust = conf
    
    # Signal 1: Convergence (The "Second Opinion" signal)
    if is_converged:
        trust += 0.12  # Soft boost for multi-mention evidence
        
    # Signal 2: Section Authority (The "Direct Intent" signal)
    if sec_priority >= 8:
        trust += 0.10  # Assessment/Plan/Operative sections
    elif sec_priority >= 5:
        trust += 0.05  # Impression/Findings sections
        
    # Signal 3: Chronic Protection (The "Contextual Continuity" signal)
    code = (code_dict.get("code") or "").upper()
    is_chronic = any(code.startswith(pfx) for pfx in CHRONIC_MANAGED_PREFIXES)
    if is_chronic:
        trust += 0.08  # Prefer historical chronic context survival
        
    # Signal 4: Entity Grounding (The "Objectivity" signal)
    entity_conf = float(code_dict.get("entity_confidence") or 0)
    if entity_conf > 0.80:
        trust += 0.05

    # Signal 5: Retrieval Authority (🚨 TASK 28)
    # If the candidate was a Top 1-3 retrieval match, trust it more.
    retrieval_trace = code_dict.get("retrieval_trace") or {}
    if not isinstance(retrieval_trace, dict): retrieval_trace = {}
    
    # Check if retrieval_rank is in metadata or trace
    r_rank = code_dict.get("metadata", {}).get("retrieval_rank") or retrieval_trace.get("retrieval_rank") or 99
    if r_rank == 1:
        trust += 0.15 # Strong anchor boost
    elif r_rank <= 3:
        trust += 0.08 # Moderate anchor boost
        
    return clamp_score(trust)


def calculate_composite_boost(boosts: list[float], cap: float = 0.25) -> float:
    """
    Step 1 & 3: Generalized diminishing returns for boost stacking (Task 7).
    
    Formula: total = sum(b_i * (0.4 ^ i))
    Ensures that the first boost matters most, and cumulative weak signals
    contribution is strictly capped (default 0.25).
    """
    if not boosts:
        return 0.0
    
    # Sort boosts descending to prioritize the strongest signal
    valid_boosts = sorted([float(b) for b in boosts if b > 0], reverse=True)
    if not valid_boosts:
        return 0.0
        
    total = 0.0
    for i, b in enumerate(valid_boosts):
        # Aggressive diminishing factor: 1.0, 0.4, 0.16, 0.06...
        factor = (0.4 ** i)
        total += b * factor
        
    return min(total, cap)


# ─────────────────────────────────────────────────────────────────────────────
# NEGATION TOKENS  (Step 3 in spec)
# These words, when appearing near a clinical term, negate active disease.
# ─────────────────────────────────────────────────────────────────────────────
NEGATION_TOKENS: tuple[str, ...] = (
    "no ",
    "not ",
    "without ",
    "denies ",
    "denied ",
    "negative for ",
    "no evidence of ",
    "no sign of ",
    "no signs of ",
    "no history of ",
    "no indication of ",
    "ruled out ",
    "rule out ",
    "rules out ",
    "r/o ",
    "exclude ",
    "excluded ",
    "absence of ",
    "absent ",
    "unremarkable for ",
)


# ─────────────────────────────────────────────────────────────────────────────
# PROPHYLAXIS / EXCLUSION CONTEXT TOKENS  (Step 2 in spec)
# These modify a term to indicate PREVENTION, not active disease.
# CRITICAL: "DVT prophylaxis" must NOT generate I82.401 (acute DVT)
# ─────────────────────────────────────────────────────────────────────────────
PROPHYLAXIS_TOKENS: tuple[str, ...] = (
    "prophylaxis",
    "prophylactic",
    "prevention",
    "preventive",
    "history of",
    "hx of",
    "h/o ",
    "risk of",
    "risk for",
    "at risk for",
    "suspected",
    "concern for",
    "possible ",
    "probable ",
    "query ",
    "r/o ",
    "rule out ",
    "ruled out ",
    "to rule out",
    "screening for",
    "evaluate for",
    "workup for",
)


# Section → confidence weight (higher = more clinically significant for active billing)
SECTION_WEIGHTS: dict[str, float] = {
    "postop_diagnosis": 1.0,    # Definitive post-procedure diagnosis
    "procedure":        0.95,   # Confirms procedure performed
    "preop_diagnosis":  0.85,   # Planned / admitting diagnosis
    "assessment":       0.85,   # Clinician's current assessment
    "impression":       0.75,   # Radiologist / specialist impression
    "findings":         0.70,   # Intraoperative / exam findings
    "plan":             0.65,   # Treatment plan (implies diagnosis)
    "history":          0.25,   # HPI context only
    "pmh":              0.20,   # Past history — not currently active
    "medications":      0.15,   # Medication context only
    "family_history":   0.05,   # Family history — not patient's active diagnosis
}

# Sections where a diagnosis should NOT directly become an active billing code
LOW_PRIORITY_SECTIONS  = frozenset(["pmh", "family_history", "medications"])

# Sections where a diagnosis IS actively confirmed
HIGH_PRIORITY_SECTIONS = frozenset([
    "postop_diagnosis", "procedure", "preop_diagnosis", "assessment"
])

# Clinical domains for specialty-constrained retrieval
ENCOUNTER_DOMAINS: dict[str, dict] = {
    "cardiology": {
        "keywords": ["heart", "cardiac", "atrial", "valve", "coronary", "chf", "mi", "stemi", "angina", "stroke", "infarct", "mca", "afib", "arrhythmia", "stenosis", "claudication"],
        "prefixes": ["I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"]
    },
    "orthopedic": {
        "keywords": ["fracture", "joint", "bone", "hip", "knee", "spine", "ortho", "operative", "fixation", "arthroplasty", "orif", "supraspinatus", "tendon", "rotator cuff", "discectomy", "laminectomy"],
        "prefixes": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "S4", "S5", "S6", "S7", "S8", "S9"]
    },
    "nephrology": {
        "keywords": ["renal", "kidney", "ckd", "nephro", "dialysis", "creatinine", "gfr", "aki", "pyelonephritis"],
        "prefixes": ["N0", "N1", "N2", "N3"]
    },
    "infectious_disease": {
        "keywords": ["infection", "sepsis", "bacterial", "viral", "culture", "antibiotic", "fever", "uti", "coli", "streptococcus", "pneumoniae", "vancomycin", "cellulitis"],
        "prefixes": ["A0", "B0", "B9", "L0"]
    }
}

PROCEDURE_COHERENCE_FAMILIES: dict[str, list[str]] = {
    "orthopedic_procedures": ["27130", "27447", "27245", "27506", "27759", "25600", "63030", "63047", "29827"],
    "cardiology_procedures": ["36.10", "36.00", "92928", "93452"],
    "general_surgery": ["47562", "44950", "49505"]
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPLICIT DIAGNOSIS TOKENS  (Step 1 in spec)
# When these appear near a term, it IS an active confirmed diagnosis.
# ─────────────────────────────────────────────────────────────────────────────
EXPLICIT_DIAGNOSIS_TOKENS: tuple[str, ...] = (
    "diagnosed with",
    "diagnosis of",
    "confirmed ",
    "positive for ",
    "presents with ",
    "admitted for ",
    "admitted with ",
    "acute ",
    "chronic ",
    "active ",
    "documented ",
    "known ",
    "established ",
    "primary diagnosis",
    "principal diagnosis",
    "secondary diagnosis",
    "assessment:",
    "impression:",
    "preoperative diagnosis",
    "postoperative diagnosis",
    "preop diagnosis",
    "postop diagnosis",
    "operative diagnosis",
    "pre-operative diagnosis",
    "post-operative diagnosis",
)


# Indicators of active clinical management (Step 3/5 Task 6)
MANAGEMENT_INDICATORS: dict[str, list[str]] = {
    "treatment": ["treatment", "therapy", "management", "plan", "rx", "care", "addressed"],
    "medication": ["prescribed", "started", "dose", "infusion", "medication", "ordered"],
    "intervention": ["procedure", "performed", "surgical", "operation", "placement", "removal"],
    "monitoring": ["monitoring", "labs", "imaging", "repeat", "serial", "follow-up", "f/u"],
}


def is_negated(term: str, note_text: str, window: int = 80) -> bool:
    """
    Return True if `term` appears in a negated context within `note_text`.

    Algorithm:
      1. Find all occurrences of `term` in note_text (case-insensitive).
      2. For each occurrence, extract up to `window` characters BEFORE the term.
      3. If a NEGATION_TOKEN appears in that pre-window, the term is negated.
      4. If an EXPLICIT_DIAGNOSIS_TOKEN appears in the full window, override
         the negation — explicit confirmation wins.

    Returns True  → term is negated (do NOT code as active disease)
    Returns False → term is not negated (safe to code)
    """
    text_lower = note_text.lower()
    term_lower = term.lower()

    positions: list[int] = []
    start = 0
    while True:
        idx = text_lower.find(term_lower, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        return False

    any_positive = False
    for pos in positions:
        pre_start   = max(0, pos - window)
        pre_window  = text_lower[pre_start:pos]
        post_end    = min(len(text_lower), pos + len(term_lower) + window)
        full_window = text_lower[pre_start:post_end]

        # Check for explicit confirmation first
        confirmed = any(tok in full_window for tok in EXPLICIT_DIAGNOSIS_TOKENS)
        if confirmed:
            return False # Positive confirmation wins immediately

        # Check if this specific instance is negated
        instance_negated = any(tok in pre_window for tok in NEGATION_TOKENS)
        
        # v15: Fracture protection — never negate a fracture if it's in a positive-leaning context
        is_fracture = (term_lower == "fracture")
        if is_fracture and not instance_negated:
             any_positive = True
             break

        if not instance_negated:
            any_positive = True
            break # Found a positive instance, so the term is NOT negated

    return not any_positive


def has_prophylaxis_context(term: str, note_text: str, window: int = 100) -> bool:
    """
    Return True if `term` appears in a prophylaxis/prevention/exclusion context.

    Critical examples:
      "DVT prophylaxis"              → True  (do NOT code I82.401 acute DVT)
      "heparin for DVT prophylaxis"  → True
      "history of DVT"               → True  (use Z86.718, not I82.401)
      "DVT confirmed on duplex"      → False (active diagnosis)
      "acute DVT"                    → False (active diagnosis)

    If an EXPLICIT_DIAGNOSIS_TOKEN appears in the same window, returns False
    because explicit confirmation overrides prophylaxis context.
    """
    text_lower = note_text.lower()
    term_lower = term.lower()

    # Find all occurrences of the term
    positions = [m.start() for m in re.finditer(re.escape(term_lower), text_lower)]
    if not positions:
        return False

    any_prophylaxis = False
    for pos in positions:
        start       = max(0, pos - window)
        end         = min(len(text_lower), pos + len(term_lower) + window)
        window_text = text_lower[start:end]

        if any(tok in window_text for tok in EXPLICIT_DIAGNOSIS_TOKENS):
            return False # Confirmation wins

        if any(tok in window_text for tok in PROPHYLAXIS_TOKENS):
            any_prophylaxis = True

    return any_prophylaxis

    # Explicit diagnosis in the same window overrides prophylaxis flag
    is_explicit = any(tok in window_text for tok in EXPLICIT_DIAGNOSIS_TOKENS)
    if is_explicit:
        logger.debug(
            "PROPHYLAXIS_OVERRIDE: '%s' has prophylaxis token but explicit diagnosis also present",
            term,
        )
        return False

    logger.debug("PROPHYLAXIS_CONTEXT: '%s' is in prophylaxis/exclusion context", term)
    return True


def compute_evidence_strength(
    code: str,
    description: str,
    note_text: str,
    entity_confidence: float = 0.0,
    is_rag_only: bool = False,
) -> tuple[float, str]:
    """
    Compute a 0.0–1.0 evidence strength score for a proposed ICD code.

    Tiers (Step 1 in spec):
      1.0  — Explicit diagnosis token + term present in note       STRONG
      0.80 — Term directly present in note (synonym match)         MODERATE
      0.65 — Entity confidence > 0.85                              MODERATE
      0.45 — Entity confidence 0.60–0.85                           WEAK
      0.20 — RAG-only (no entity or note support)                  VERY WEAK
      0.0  — Negated or in prophylaxis context                     REJECT

    Returns: (strength: float, reason: str)
    """
    text_lower = note_text.lower()
    desc_lower = description.lower()

    # Extract meaningful clinical terms from ICD description
    stop_words = {
        "unspecified", "other", "type", "nos", "due", "with", "without",
        "acute", "chronic", "bilateral", "right", "left", "initial", "subsequent",
        "encounter", "specified", "site", "code", "also", "first", "and",
    }
    desc_words = [
        w for w in re.sub(r"[^a-z\s]", "", desc_lower).split()
        if len(w) > 4 and w not in stop_words
    ]

    if not desc_words:
        return (0.20 if is_rag_only else 0.45), "no meaningful description terms to match"

    # --- Reject: negation on key terms ---
    for word in desc_words[:3]:
        if is_negated(word, note_text, window=80):
            logger.info(
                "EVIDENCE_REJECT[%s]: key term '%s' is negated → strength=0.0", code, word
            )
            return 0.0, f"key term '{word}' negated in note"

    # --- Reject: prophylaxis/exclusion context ---
    for word in desc_words[:3]:
        if has_prophylaxis_context(word, note_text, window=100):
            logger.info(
                "EVIDENCE_REJECT[%s]: key term '%s' in prophylaxis context → strength=0.0",
                code, word,
            )
            return 0.0, f"key term '{word}' in prophylaxis/exclusion context"

    # --- Signal Fusion (Task 7 & 15: continuous scoring) ---
    has_term_in_note = any(w in text_lower for w in desc_words[:3])
    has_explicit_token = any(tok in text_lower for tok in EXPLICIT_DIAGNOSIS_TOKENS)
    
    # Start with entity confidence (RAG + SapBERT + Cross-Encoder)
    score = entity_confidence
    
    # 1. Boost for explicit Note Support
    if has_term_in_note:
        if has_explicit_token:
            score += 0.25 # Explicit diagnosis boost
            reason = "entity support + explicit diagnosis token in note"
        else:
            score += 0.15 # Clinical term boost
            reason = "entity support + clinical term present in note"
    else:
        # Penalize if term is missing from note
        score -= 0.20
        reason = "entity support but term missing from note context"

    # 2. RAG-only adjustment
    if is_rag_only:
        score -= 0.10
        reason += " | RAG-only caution applied"

    # Task 15: Fully continuous linear scoring
    return clamp_score(score), reason




# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SYSTEM  (Steps 1-4, 6-7, 9-10 in calibration spec)
# ─────────────────────────────────────────────────────────────────────────────

# Step 2: Single authoritative clamp — ALL intermediate scores pass through this.
def clamp_score(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]. Use everywhere to prevent boost/penalty overflow."""
    try:
        return float(max(lo, min(hi, value)))
    except (TypeError, ValueError):
        return float(lo)

def sanitize_numpy(obj: Any) -> Any:
    """Recursively converts numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_numpy(obj.tolist())
    return obj


# Step 4: Differentiated rejection thresholds per confidence tier.
# EMPIRICAL TUNING v2 — benchmark failure analysis:
#   Cluster 1 (hallucinations): rag_only raised to suppress unsupported acute codes
#   Cluster 2 (over-suppression): high/medium_entity lowered to recover valid diagnoses
#   Cluster 3 (procedure instability): cpt/procedure lowered for stable CPT acceptance
CALIBRATION_THRESHOLDS: dict[str, float] = {
    # --- CONFIDENCE TIERS (Differentiated Thresholds) ---
    "postop_diagnosis":   0.28,
    "procedure":          0.28,
    "preop_diagnosis":    0.26,  # Lowered from 0.36
    "assessment":         0.26,  # Lowered from 0.36
    "impression":         0.45,
    "findings":           0.50,
    "deterministic":      0.00,
    "cpt":                0.30,  # Lowered from 0.32
    "high_entity":        0.32,  # Lowered from 0.42
    "medium_entity":      0.40,  # Lowered from 0.48
    "rag_only":           0.65,  # Lowered from 0.76
    "high_risk":          0.86,
    "default":            0.40,  # Lowered from 0.55

    # --- SYSTEM PARAMETERS (Internal Tuning) ---
    "evidence_minimum":              0.35,
    "cpt_protection_minimum":        0.50,
    "tier4_survival_floor":          0.20,
    "overgeneralization_suppress":   0.60,
    "semantic_drift_limit":          0.55,
    "false_positive_ev_ceiling":     0.45,
    "severe_diagnosis_floor":        0.45,
    "anatomy_coherence_threshold":   0.60,
    "relationship_certainty_floor":  0.45,
    "terminal_negation_threshold":   0.85,
    "terminal_tier4_ev_max":         0.15,
    "terminal_overgen_threshold":    0.80,
    "cpt_operative_boost":           0.18,
    "cpt_workflow_boost":            0.12,
    "fn_recovery_assertion_min":     0.75,
    "fn_recovery_anatomy_min":       0.60,
    "fn_recovery_ev_target":         0.55,
}

# Backwards-compatible alias used by legacy code
EVIDENCE_STRENGTH_THRESHOLD = CALIBRATION_THRESHOLDS["default"]

# Empirical tuning: extended prophylaxis window for medication-heavy notes.
# Standard window (100 chars) missed: "... enoxaparin ... DVT prophylaxis ..."
# in long medication lists where terms are 150+ chars apart.
PROPHYLAXIS_WINDOW_STD  = 100   # Standard window for focused clinical mentions
PROPHYLAXIS_WINDOW_LONG = 220   # Extended window for medication-section contexts


def get_differentiated_threshold(
    code: str,
    code_type: str,
    source: str,
    entity_confidence: float,
    section_dominant: str | None = None,
) -> tuple[float, str]:
    """
    Step 4: Return the appropriate rejection threshold for a code, plus the tier name.

    Priority (highest→lowest):
      1. deterministic/protected  → 0.00
      2. CPT                      → 0.35
      3. post-op/procedure section→ 0.30
      4. pre-op/assessment section→ 0.38
      5. high entity confidence   → 0.45
      6. medium entity confidence → 0.50
      7. RAG-only                 → 0.62
      8. default                  → 0.50
    """
    if source in ("deterministic", "protected"):
        return CALIBRATION_THRESHOLDS["deterministic"], "deterministic"

    if code_type == "CPT":
        return CALIBRATION_THRESHOLDS["cpt"], "cpt"

    sec = (section_dominant or "").lower()
    if sec in ("postop_diagnosis",):
        return CALIBRATION_THRESHOLDS["postop_diagnosis"], "postop_diagnosis"
    if sec in ("procedure",):
        return CALIBRATION_THRESHOLDS["procedure"], "procedure"
    if sec in ("preop_diagnosis",):
        return CALIBRATION_THRESHOLDS["preop_diagnosis"], "preop_diagnosis"
    if sec in ("assessment",):
        return CALIBRATION_THRESHOLDS["assessment"], "assessment"
    if sec in ("impression",):
        return CALIBRATION_THRESHOLDS["impression"], "impression"
    if sec in ("findings",):
        return CALIBRATION_THRESHOLDS["findings"], "findings"

    if entity_confidence >= 0.85:
        return CALIBRATION_THRESHOLDS["high_entity"], "high_entity"
    if entity_confidence >= 0.60:
        return CALIBRATION_THRESHOLDS["medium_entity"], "medium_entity"

    if source == "rag" and entity_confidence < 0.60:
        return CALIBRATION_THRESHOLDS["rag_only"], "rag_only"

    # Task 10.3: Proper high-risk prefix check (added J96)
    HIGH_RISK_PREFIXES = ("I21", "I22", "I63", "A41", "J96")
    if any(code.startswith(pfx) for pfx in HIGH_RISK_PREFIXES):
        return CALIBRATION_THRESHOLDS["high_risk"], "high_risk"

    return CALIBRATION_THRESHOLDS["default"], "default"


def build_scoring_breakdown(
    evidence_score: float,
    anatomy_score: float,
    specificity_score: float,
    section_score: float,
    relationship_score: float,
    penalty_score: float,
) -> dict:
    """
    Step 1: Build and return a structured scoring breakdown dict.

    All scores clamped to [0.0, 1.0].
    final_score = weighted composite for ranking stability (Step 3).

    Weights (tuned for clinical billing priority):
      evidence     0.35  — primary clinical grounding
      section      0.25  — section context (post-op > history)
      relationship 0.15  — modifier/causality/procedure links
      specificity  0.10  — ICD specificity (prefer specific over generic)
      anatomy      0.10  — anatomy consistency
      penalty      0.05  — negation/hedge/generic penalty (subtracted)

    The final_score sits in [0.0, 1.0] and is used for stable ranking.
    """
    ev  = clamp_score(evidence_score)
    an  = clamp_score(anatomy_score)
    sp  = clamp_score(specificity_score)
    sec = clamp_score(section_score)
    rel = clamp_score(relationship_score)
    pen = clamp_score(penalty_score)

    final = clamp_score(
        ev  * 0.38   # Tuned↑ from 0.35: evidence must be dominant signal
        + sec * 0.22   # Tuned↓ from 0.25: section alone without evidence was too permissive
        + rel * 0.10   # Tuned↓ from 0.15: relationship over-boost caused FP survival
        + sp  * 0.10   # Unchanged: specificity is a tiebreaker, not primary signal
        + an  * 0.15   # Tuned↑ from 0.10: anatomy match is critical safety gate
        - pen * 0.08   # Tuned↑ from 0.05: stronger penalty for negation/hedge codes
    )

    return {
        "evidence_score":      round(ev,    3),
        "anatomy_score":       round(an,    3),
        "specificity_score":   round(sp,    3),
        "section_score":       round(sec,   3),
        "relationship_score":  round(rel,   3),
        "penalty_score":       round(pen,   3),
        "final_score":         round(final, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANATOMY SYSTEM  (Steps 1-3 in spec)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical region → list of synonym terms found in notes / ICD descriptions
ANATOMY_REGION_MAP: dict[str, list[str]] = {
    "hip":        ["hip", "femoral neck", "femoral head", "femur", "femoral",
                   "acetabulum", "acetabular", "pelvis", "pelvic", "trochanter",
                   "intertrochanteric", "subtrochanteric"],
    "knee":       ["knee", "tibia", "tibial", "patella", "patellar",
                   "fibula", "fibular", "distal femur"],
    "forearm":    ["forearm", "radius", "radial", "ulna", "ulnar",
                   "wrist", "distal radius"],
    "shoulder":   ["shoulder", "humerus", "humeral", "clavicle", "clavicular",
                   "scapula", "glenoid", "rotator cuff"],
    "spine":      ["spine", "spinal", "vertebra", "vertebral", "lumbar",
                   "cervical", "thoracic", "sacral", "sacrum", "disc", "disk"],
    "ankle":      ["ankle", "malleolus", "fibula", "calcaneus", "calcaneal",
                   "talus", "talar"],
    "foot":       ["foot", "metatarsal", "phalanx", "phalange", "tarsal"],
    "hand":       ["hand", "metacarpal", "finger", "thumb", "carpal"],
    "elbow":      ["elbow", "olecranon", "radial head", "distal humerus"],
    "chest":      ["rib", "sternum", "sternal", "chest wall"],
    "lower_leg":  ["tibia", "tibial shaft", "fibula", "lower leg"],
    "upper_arm":  ["humerus", "humeral shaft", "upper arm"],
}

# ICD-10 code PREFIX → canonical anatomy region(s)
_ICD_PREFIX_TO_ANATOMY: dict[str, list[str]] = {
    # Fractures (S7x = femur/hip, S5x = forearm, S8x = lower leg, etc.)
    "S72": ["hip"],          # Fracture of femur
    "S73": ["hip"],          # Dislocation of hip
    "S79": ["hip"],          # Other hip/thigh injuries
    "S52": ["forearm"],      # Fracture of forearm (radius/ulna)
    "S53": ["forearm", "elbow"],
    "S59": ["forearm", "elbow"],
    "S82": ["lower_leg", "knee"],  # Fracture of lower leg
    "S83": ["knee"],
    "S89": ["lower_leg", "knee"],
    "S42": ["shoulder"],     # Fracture of shoulder / upper arm
    "S43": ["shoulder"],
    "S49": ["shoulder", "upper_arm"],
    "S92": ["foot"],
    "S93": ["ankle", "foot"],
    "S99": ["ankle", "foot"],
    "S62": ["hand"],
    "S63": ["hand", "wrist"],
    "S32": ["spine", "hip"],  # Fracture lumbar spine / pelvis
    "S22": ["chest", "spine"],
    "S12": ["spine"],
    # Arthroplasty / procedures
    "Z96": ["hip", "knee", "shoulder"],  # Presence of implants
    "M16": ["hip"],   # Osteoarthritis hip
    "M17": ["knee"],  # Osteoarthritis knee
    "M75": ["shoulder"],
}

# Procedure CPT prefix → anatomy
_CPT_PREFIX_TO_ANATOMY: dict[str, list[str]] = {
    "271": ["hip"],    # 271xx = hip arthroplasty
    "272": ["knee"],   # 272xx = knee arthroplasty
    "234": ["shoulder"],
    "235": ["shoulder"],
    "236": ["shoulder"],
    "259": ["forearm", "elbow"],
    "260": ["forearm"],
    "279": ["lower_leg", "ankle"],
    "280": ["foot"],
}


def extract_anatomy_regions(note_text: str) -> set[str]:
    """
    Step 1: Extract all anatomical regions mentioned in the note.

    Scans note_text for each synonym in ANATOMY_REGION_MAP.
    Returns a set of canonical region names, e.g. {"hip", "forearm"}.

    Examples:
      "displaced femoral neck fracture" → {"hip"}
      "fracture of the radius"          → {"forearm"}
    """
    text_lower = note_text.lower()
    found: set[str] = set()
    for region, synonyms in ANATOMY_REGION_MAP.items():
        for syn in synonyms:
            if syn in text_lower:
                found.add(region)
                break
    return found


def get_code_anatomy(code: str, description: str) -> set[str]:
    """
    Step 2: Infer the anatomical region(s) for an ICD or CPT code.

    Uses:
      a) ICD prefix table (_ICD_PREFIX_TO_ANATOMY)
      b) CPT prefix table (_CPT_PREFIX_TO_ANATOMY)
      c) Keyword scan of the code description against ANATOMY_REGION_MAP

    Returns a set of canonical region names, e.g. {"hip"}, {"forearm"}.
    Empty set means the code has no clear anatomy association (pass-through).
    """
    code_upper = code.strip().upper()
    desc_lower = description.lower()
    regions: set[str] = set()

    # (a) ICD prefix table — check 3-char and 4-char prefixes
    for pfx, rgns in _ICD_PREFIX_TO_ANATOMY.items():
        if code_upper.startswith(pfx):
            regions.update(rgns)
            break

    # (b) CPT prefix table
    for pfx, rgns in _CPT_PREFIX_TO_ANATOMY.items():
        if code_upper.startswith(pfx):
            regions.update(rgns)
            break

    # (c) Description keyword scan
    for region, synonyms in ANATOMY_REGION_MAP.items():
        for syn in synonyms:
            if syn in desc_lower:
                regions.add(region)
                break

    return regions


def check_anatomy_consistency(
    code: str,
    description: str,
    note_anatomy: set[str],
) -> tuple[bool, str]:
    """
    Step 3: Hard anatomical consistency check.

    Returns (is_consistent: bool, reason: str).

    A code is ANATOMICALLY CONSISTENT if:
      - The code has no specific anatomy (empty set) — pass-through.
      - OR any of the code's anatomy regions overlaps with note_anatomy.

    If note_anatomy is empty (no anatomy detected), we skip the check
    (conservative: don't reject when we cannot extract anatomy).

    Examples:
      code=S52.90XA (forearm), note_anatomy={"hip"}  → INCONSISTENT → REJECT
      code=S72.011A (femur),   note_anatomy={"hip"}  → CONSISTENT  → KEEP
    """
    if not note_anatomy:
        return True, "no anatomy detected in note — skip check"

    code_anatomy = get_code_anatomy(code, description)
    if not code_anatomy:
        return True, "code has no specific anatomy association"

    overlap = code_anatomy & note_anatomy
    if overlap:
        return True, f"anatomy consistent: {overlap}"
    
    # v15: Loosen anatomy for pathological fractures (M80) — they often involve regional crossovers
    if code.startswith("M80") or code.startswith("M81"):
        flexible_regions = {"hip", "femur", "shoulder", "humerus", "pelvis", "spine"}
        if (code_anatomy & flexible_regions) and (note_anatomy & flexible_regions):
            return True, f"anatomy flexible match for pathological fracture: {code_anatomy & flexible_regions}"

    return (
        False,
        f"anatomy mismatch: code anatomy={code_anatomy}, note anatomy={note_anatomy}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROCEDURE VALIDATION  (Steps 4-5 in spec)
# ─────────────────────────────────────────────────────────────────────────────

# CPT code → list of phrase patterns that CONFIRM the procedure in a note.
# Matching ANY phrase → evidence_strength boosted to 1.0.
PROCEDURE_EVIDENCE_MAP: dict[str, list[str]] = {
    # Hip arthroplasty
    "27130": ["total hip arthroplasty", "total hip replacement", "tha",
              "hip replacement", "hip arthroplasty", "thr"],
    "27132": ["revision hip", "revision total hip", "revision tha"],
    "27125": ["hemi hip", "hemiarthroplasty hip", "femoral head replacement"],
    # Knee arthroplasty
    "27447": ["total knee arthroplasty", "total knee replacement", "tka",
              "knee replacement", "knee arthroplasty", "tkr"],
    "27487": ["revision knee", "revision total knee", "revision tka"],
    # Shoulder
    "23472": ["total shoulder", "shoulder arthroplasty", "shoulder replacement",
              "tsa", "reverse shoulder", "reverse total shoulder"],
    # ORIF hip / femur
    "27236": ["orif femoral neck", "open reduction femoral", "orif hip",
              "femoral nail", "intramedullary nail", "im nail femur"],
    "27244": ["orif intertrochanteric", "orif trochanteric", "sliding hip screw",
              "dynamic hip screw", "compression hip screw"],
    # ORIF forearm
    "25600": ["orif radius", "closed reduction radius", "wrist fracture repair"],
    "25607": ["orif distal radius", "open reduction radius", "volar plate"],
    # Spinal fusion
    "22612": ["posterior lumbar fusion", "plif", "tlif", "lumbar fusion"],
    "22630": ["posterior lumbar interbody", "interbody fusion lumbar"],
    # Neuro-intervention
    "61645": ["mechanical thrombectomy", "thrombectomy", "endovascular therapy", "evt"],
    "61624": ["cerebral aneurysm coiling", "coiling", "aneurysm embolization"],
    # Imaging
    "70450": ["ct head without contrast", "ct head w/o contrast", "ct head", "head ct", "non-contrast ct head"],
    "70460": ["ct head with contrast", "ct head w/ contrast"],
    "70470": ["ct head without and with", "ct head w/wo contrast"],
    "70496": ["cta head", "ct angiography head", "computed tomography angiography head", "cta of the head"],
    "70498": ["cta neck", "ct angiography neck", "computed tomography angiography neck"],
    "70551": ["mri brain without contrast", "mri brain w/o contrast", "mri brain", "brain mri", "non-contrast mri brain"],
    "70553": ["mri brain with and without", "mri brain w/wo"],
    "71250": ["ct chest without contrast", "ct chest", "chest ct"],
    "71275": ["cta chest", "ct angiography chest", "pulmonary embolism protocol ct", "pe protocol ct"],
    "93306": ["echocardiogram", "tte", "transthoracic echocardiogram", "echo"],
    # Cardiovascular
    "92920": ["pci", "percutaneous coronary intervention", "angioplasty", "stent placement"],
    "33510": ["cabg", "coronary artery bypass graft", "bypass surgery"],
    "93458": ["left heart catheterization", "coronary angiogram", "lhc", "cardiac cath"],
}


def validate_procedure_evidence(
    cpt_code: str,
    note_text: str,
    code_description: str = "",
) -> tuple[float, str]:
    """
    Steps 4-5: Validate a CPT procedure code against note text (Task 8A).

    Returns (evidence_strength: float, matched_text: str).

    Algorithm:
      1. Explicit phrase match in PROCEDURE_EVIDENCE_MAP (1.0)
      2. Workflow-level alignment (Step 5): Multiple indicators (e.g. Guidance + Modality)
      3. Proximity check (Step 2): Qualifiers near procedure terms
      4. Fallback to generic context (0.75) or weak grounding (0.55)
    """
    # Strip modifier (e.g. "27130-LT" → "27130")
    base_code = cpt_code.split("-")[0].strip().upper()
    text_lower = note_text.lower()
    desc_lower = code_description.lower()

    # (1) Explicit phrase match
    phrases = PROCEDURE_EVIDENCE_MAP.get(base_code, [])
    for phrase in phrases:
        if phrase in text_lower:
            logger.info("PROCEDURE_VALIDATED: code=%s | matched_text='%s'", cpt_code, phrase)
            return 1.0, f"PROCEDURE_WORKFLOW_MATCH: explicit phrase '{phrase}'"

    # (2) Workflow-level alignment (Step 5)
    # Check for guidance, contrast, and modality indicators
    workflow_hits = []
    for category, indicators in PROCEDURE_WORKFLOW_INDICATORS.items():
        if any(ind in text_lower for ind in indicators):
            # Proximity check (Step 2)
            # If the indicator is in the code description AND in the note, it's a strong match
            if any(ind in desc_lower for ind in indicators):
                workflow_hits.append(f"{category}_grounded")
            else:
                workflow_hits.append(category)

    if len(workflow_hits) >= 2:
        return 0.92, f"PROCEDURE_WORKFLOW_MATCH: aligned signals {workflow_hits} (PROCEDURE_UNSUPPORTED_RECALIBRATED)"

    # (3) Generic procedure context (Step 4 recalibration)
    generic_proc = any(w in text_lower for w in [
        "arthroplasty", "replacement", "orif", "open reduction",
        "arthroscopy", "fusion", "fixation", "repair", "reconstruction",
        "thoracentesis", "biopsy", "angiography", "thrombectomy", "infusion"
    ])
    if generic_proc:
        return 0.75, "PROCEDURE_UNSUPPORTED_RECALIBRATED: generic procedure context found"

    return 0.55, "no explicit procedure phrase found (weak grounding)"


# ─────────────────────────────────────────────────────────────────────────────
# SPECIFICITY SYSTEM  (Steps 1-9 in spec)
# ─────────────────────────────────────────────────────────────────────────────

# Tokens in an ICD description that indicate HIGHER specificity
_SPECIFICITY_MARKERS: dict[str, int] = {
    # Laterality (+3 each)
    "left":          3,
    "right":         3,
    "bilateral":     2,
    # Vascular Territory (+3) - Task 5 Step 2
    "mca":           3,
    "middle cerebral": 3,
    "pca":           3,
    "posterior cerebral": 3,
    "aca":           3,
    "anterior cerebral": 3,
    "carotid":       2,
    "vertebral":     2,
    "basilar":       2,
    "coronary":      2,
    "lad":           3,
    "left anterior descending": 3,
    "rca":           3,
    "right coronary": 3,
    "circumflex":    3,
    # Chronic Complication / Combination (+4) - Task 5 Step 2 & 5
    "with retinopathy": 4,
    "with nephropathy": 4,
    "with neuropathy": 4,
    "with ulcer":    4,
    "with gangrene": 4,
    "with chronic kidney": 4,
    "with ckd":      4,
    # Staging / Acuity (+2)
    "stage":         2,
    "end stage":     3,
    "esrd":          3,
    "persistent":    2,
    "paroxysmal":    2,
    "recurrent":     2,
    "transformed":   2,
    # Encounter type (+2)
    "initial":       2,
    "subsequent":    2,
    "sequela":       2,
    # Displacement (+3 — clinically critical)
    "displaced":     3,
    "nondisplaced":  3,
    # Pathological / fracture type (+3)
    "pathological":  4,
    "fragility":     4,
    "osteoporotic":  3,
    "compression":   2,
    "stress":        2,
    # Anatomical precision (+2)
    "neck":          2,
    "head":          2,
    "shaft":         2,
    "base":          2,
    "condyle":       2,
    "malleolus":     2,
    "trochanter":    2,
    "intertrochanteric": 3,
    "subtrochanteric":   3,
    "acetabular":    2,
    # Acuity (+1)
    "acute":         1,
    "chronic":       1,
    # Respiratory Specificity (Task 8B Step 1)
    "hypoxemic":     5,   # J96.01 over J96.00 — clinically critical qualifier
    "hypercapnic":   5,   # hypercapnic respiratory failure
    "with hypoxia":  4,   # respiratory condition with documented hypoxia
    "exacerbation":  4,   # COPD/asthma exacerbation — key specificity driver
    "with exacerbation": 4,
    "acute-on-chronic": 4,
    "acute on chronic": 4,
    "oxygen dependent": 3,
    "ventilator dependent": 4,
    "mechanical ventilation": 4,
    "non-invasive ventilation": 3,
    "bipap":         3,
    "cpap":          3,
    "high flow":     3,
    "transudate":    4,   # pleural effusion subtype
    "exudate":       4,   # pleural effusion subtype
    "malignant":     3,
    "parapneumonic": 4,
    # Combination code marker (+2)
    "with":          2,
    "due to":        2,
    # Generic/NOS penalties (negative)
    "unspecified":  -4, # Calibrated Tier 3 penalty
    "nos":          -4,
    "other":        -3,
}

# CPT-specific markers for Task 8A Step 1
_CPT_SPECIFICITY_MARKERS: dict[str, int] = {
    # Imaging Guidance (+5)
    "guidance": 5,
    "guided": 5,
    "ultrasound": 4,
    "fluoroscopy": 4,
    "ct guided": 6,
    "mri guided": 6,
    "radiological supervision": 5,
    "stereotactic": 4,
    
    # Contrast (+4)
    "with contrast": 4,
    "with and without contrast": 5,
    
    # Administration/Workflow specificity (+4)
    "infusion": 4,
    "injection": 2,
    "intravenous": 2,
    "initial": 2,
    "each additional": 3,
    "up to 1 hour": 3,
    "additional hour": 3,
    "concurrent": 3,
    "sequential": 3,
    
    # Modality Detail (+3)
    "thoracentesis": 4,
    "thrombectomy": 4,
    "biopsy": 3,
    "angiography": 3,
    "therapeutic": 3,
    "diagnostic": 2,
}

# Indicators of procedural workflow (Step 2/5 Task 8A)
PROCEDURE_WORKFLOW_INDICATORS: dict[str, list[str]] = {
    "guidance": ["ultrasound", "u/s", "fluoroscopy", "guidance", "imaging control", "radiologist"],
    "contrast": ["contrast", "omnipaque", "visipaque", "gadolinium", "dye", "enhanced"],
    "infusion": ["infusion", "pump", "bolus", "duration", "hours", "minutes", "rate"],
    "specimen": ["specimen", "fluid", "analysis", "cytology", "culture", "biopsy"],
}

# Generic ICD suffixes that indicate a parent/unspecified code
_GENERIC_SUFFIXES = (".9", ".90", ".900", "XA", "9XA")
_GENERIC_DESC_WORDS = frozenset(["unspecified", "nos", "other specified", "not elsewhere classified"])


def clinical_specificity_score(code: str, description: str) -> int:
    """
    Step 1: Compute a specificity score for a code (ICD-10 or CPT).
    Unified scoring system for Task 8A.
    """
    code_upper = code.strip().upper()
    desc_lower = (description or "").lower()
    
    # Check if CPT (5 digits)
    is_cpt = code_upper.isdigit() and len(code_upper) == 5
    
    if is_cpt:
        # (a) CPT Specificity (Step 1 Task 8A)
        score = 10 # Base score for CPT
        for marker, weight in _CPT_SPECIFICITY_MARKERS.items():
            if marker in desc_lower:
                score += weight
        return score
    else:
        # (b) ICD Specificity (Original logic)
        score = len(code_upper.replace(".", ""))
        for marker, weight in _SPECIFICITY_MARKERS.items():
            if marker in desc_lower:
                score += weight
        if any(code_upper.endswith(suf) for suf in _GENERIC_SUFFIXES):
            score -= 4 # Tier 3 penalty
        return score


def compute_procedural_survival_score(
    code_dict: dict,
    note_text: str = "",
) -> float:
    """
    Step 1: Compute a survival score for a procedure (CPT) (Task 9A).
    Higher score = stronger grounding = higher resistance to pruning/suppression.
    """
    code_type = (code_dict.get("type") or "ICD-10").upper()
    if code_type != "CPT":
        return 0.0

    score = 0.0
    
    # 1. Evidence Strength (Primary Grounding) - 40% weight
    ev_strength = float(code_dict.get("evidence_strength") or 0)
    score += ev_strength * 0.40
    
    # 2. Section Authority (30% weight)
    sec = (code_dict.get("section_dominant") or "full_note").lower()
    if sec in ["operative_report", "procedure", "postop_diagnosis", "findings"]:
        score += 0.30
    elif sec in ["assessment", "plan"]:
        score += 0.15
        
    # 3. Workflow & Modifier Stability (30% weight)
    reason = (code_dict.get("evidence_reason") or "").upper()
    if "PROCEDURE_WORKFLOW_MATCH" in reason:
        score += 0.15
    
    code = (code_dict.get("code") or "").upper()
    if "-" in code:
        score += 0.15
        
    return min(1.0, score)


def has_specificity_markers(description: str) -> list[str]:
    """
    Step 2: Return which specificity markers are present in the description.
    """
    desc_lower = (description or "").lower()
    return [m for m in _SPECIFICITY_MARKERS if _SPECIFICITY_MARKERS[m] > 0 and m in desc_lower]


def compute_specificity_dominance(code_a: str, code_b: str, desc_a: str, desc_b: str) -> float:
    """
    Part 1 — Specificity Survival Lock.
    Compares two codes (A and B) and returns a score indicating how much A dominates B
    in terms of specificity granularity.
    Positive value = A is more specific than B.
    """
    score_a = clinical_specificity_score(code_a, desc_a)
    score_b = clinical_specificity_score(code_b, desc_b)
    
    gap = float(score_a - score_b)
    
    # Bonus for specific modifiers (laterality, subtype)
    desc_a_l = (desc_a or "").lower()
    desc_b_l = (desc_b or "").lower()
    
    specificity_dims = [
        "left", "right", "bilateral",
        "persistent", "paroxysmal", "permanent",
        "hypoxemic", "hypercapnic",
        "guided", "with contrast", "ultrasound", "fluoroscopy",
        "pathological", "displaced",
        "acute on chronic", "acute-on-chronic",
        "mca", "pca", "aca", "lad", "rca", "circumflex"
    ]
    
    for dim in specificity_dims:
        if dim in desc_a_l and dim not in desc_b_l:
            gap += 2.0
            
    return gap


def is_less_specific_variant(code_a: str, code_b: str, desc_a: str, desc_b: str) -> bool:
    """
    Returns True if A is a less specific variant of B.
    Used to trigger aggressive downranking of A if B is strongly grounded.
    """
    # 1. Check direct prefix hierarchy
    if is_parent_of(code_a, code_b):
        return True
        
    # 2. Check semantic specificity gap
    gap = compute_specificity_dominance(code_b, code_a, desc_b, desc_a)
    if gap >= 3.0:
        # If they share the same 3-char prefix but B is much more specific
        prefix_a = code_a[:3]
        prefix_b = code_b[:3]
        if prefix_a == prefix_b:
            return True
            
    return False


def compute_specificity_gap(code_a: str, code_b: str) -> int:
    """Simple integer gap for quick reconciliation decisions."""
    return len(code_a.replace(".", "")) - len(code_b.replace(".", ""))


def compute_procedure_grounding_strength(code_dict: dict, note_text: str) -> float:
    """
    Part 2 — Procedure Survival Stabilization.
    Aggregates match quality, workflow indicators, and section authority.
    """
    return compute_procedural_survival_score(code_dict, note_text)


def is_symptom_integral_to_diagnosis(symptom_code: str, diagnosis_code: str, diagnosis_desc: str) -> bool:
    """
    Part 4 — Principal Condition Dominance.
    Returns True if the symptom is likely integral to the diagnosis.
    """
    s_code = (symptom_code or "").upper()
    d_code = (diagnosis_code or "").upper()
    desc   = (diagnosis_desc or "").lower()
    
    # R-codes are usually symptoms
    if not s_code.startswith("R"):
        # Exception: Back pain (M54.5) etc.
        if not s_code.startswith("M54"):
            return False
            
    # Integral pairs (Generalized)
    integral_logic = [
        ("I50", ["edema", "dyspnea", "shortness of breath", "chf"]),
        ("J44", ["dyspnea", "shortness of breath", "cough"]),
        ("J96", ["dyspnea", "shortness of breath", "hypoxia"]),
        ("M80", ["pain", "back pain", "ache"]),
        ("S",   ["pain", "swelling", "bruising"]),
        ("A41", ["fever", "tachycardia", "hypotension"]),
        ("N39.0", ["frequency", "urgency", "dysuria"]),
        ("I21", ["chest pain", "pain"]),
        ("I48", ["palpitations", "chest pain"]),
    ]
    
    for prefix, symptoms in integral_logic:
        if d_code.startswith(prefix):
             if any(s in desc for s in symptoms):
                  return True
                  
    return False


def is_symptom_independently_managed(symptom_code: str, note_text: str) -> bool:
    """
    Returns True if the symptom is explicitly evaluated or treated independently.
    """
    text = (note_text or "").lower()
    management_signals = [
        "treated with", "started on", "managed with", "evaluated by",
        "consulted for", "plan for", "workup for"
    ]
    
    # This is a heuristic check for independent management signals
    if any(sig in text for sig in management_signals):
        # Could be more precise with proximity, but this is a start
        return True
                
    return False


def compute_management_activity_score(code_dict: dict, note_text: str) -> float:
    """
    Part 1 — Encounter Narrative Dominance.
    Evaluates treatment linkage, medication changes, and monitoring intensity.
    """
    text = (note_text or "").lower()
    score = 0.0
    
    # 1. Treatment/Medication Signals
    treatment_keywords = [
        "started on", "treated with", "dose increased", "dose decreased",
        "medication changed", "titrated", "bolus", "drip", "infusion",
        "prescribed", "refilled", "discontinued"
    ]
    for kw in treatment_keywords:
        if kw in text:
            score += 0.15
            break
            
    # 2. Monitoring Intensity
    monitoring_keywords = [
        "monitored for", "frequent checks", "serial", "q4h", "q2h", "continuous",
        "telemetry", "observation", "stable on", "improved on"
    ]
    for kw in monitoring_keywords:
        if kw in text:
            score += 0.15
            break
            
    return min(0.5, score)


def compute_encounter_driver_score(code_dict: dict, note_text: str) -> float:
    """
    Evaluates assessment/plan dominance and diagnostic workup.
    """
    score = 0.0
    sec = (code_dict.get("section_dominant") or "full_note").lower()
    
    # 1. Section Authority
    if sec in ["assessment", "plan", "impression", "final_diagnosis", "discharge_diagnosis"]:
        score += 0.40
    elif sec in ["procedure", "postop_diagnosis", "preop_diagnosis", "findings"]:
        score += 0.60  # Increased for operative driver
        
    # 2. Workup signals
    workup_keywords = [
        "workup for", "evaluation of", "ordered", "diagnostic", "labs show",
        "imaging reveals", "biopsy confirmed", "culture pending"
    ]
    text = (note_text or "").lower()
    if any(kw in text for kw in workup_keywords):
        score += 0.20
        
    return min(1.0, score)


def compute_encounter_narrative_strength(code_dict: dict, note_text: str) -> float:
    """
    Aggregates management activity and encounter driver scores.
    """
    m_score = compute_management_activity_score(code_dict, note_text)
    d_score = compute_encounter_driver_score(code_dict, note_text)
    return min(1.0, m_score + d_score)


def compute_principal_diagnosis_strength(code_dict: dict, note_text: str) -> float:
    """
    Part 3 — Principal Diagnosis Centralization.
    Combines narrative strength with procedure linkage and discharge significance.
    """
    narrative_strength = compute_encounter_narrative_strength(code_dict, note_text)
    
    score = narrative_strength * 0.60
    
    # 1. Procedure Linkage
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.30
        
    # 2. Discharge/Final Authority
    sec = (code_dict.get("section_dominant") or "").lower()
    if "final" in sec or "discharge" in sec:
        score += 0.20
        
    return min(1.0, score)


# --- Task 9E: Cross-Pass Stability & Priority Hierarchy ---

REASONING_PRIORITY = {
    "TEMPORAL": 10,
    "ENCOUNTER": 9,
    "PROCEDURE": 8,
    "PRINCIPAL": 7,
    "SPECIFICITY": 6,
    "RELATIONSHIP": 5,
    "SEMANTIC": 4,
    "COMPACTION": 3,
    "BASELINE": 0
}


def apply_priority_safe_adjustment(current_val: float, delta: float, pass_type: str, code_dict: dict) -> float:
    """
    Part 1 — Reasoning Priority Hierarchy.
    Ensures lower-priority passes cannot override higher-priority decisions.
    """
    pass_key = pass_type.upper()
    priority = REASONING_PRIORITY.get(pass_key, 0)
    highest_lock = code_dict.get("highest_priority_lock", 0)
    
    if priority < highest_lock:
        if delta < 0: # Only block suppression if priority is lower
            code_dict.setdefault("audit_traces", []).append("OVERRIDE_BLOCKED")
            return current_val
        
    # Apply lock if this pass is high priority (Temporal, Encounter, Procedure)
    if priority >= 8:
        code_dict["highest_priority_lock"] = max(highest_lock, priority)
        if "PRIORITY_LOCK_APPLIED" not in (code_dict.get("audit_traces") or []):
             code_dict.setdefault("audit_traces", []).append("PRIORITY_LOCK_APPLIED")
        
    return clamp_score(current_val + delta)


def can_override_reasoning_state(new_priority: int, current_lock: int) -> bool:
    return new_priority >= current_lock


def compute_stability_resistance(code_dict: dict) -> float:
    """
    Part 2 — Stable Confidence Accumulation.
    Strongly grounded codes gain resistance to downstream volatility.
    """
    base_strength = float(code_dict.get("base_evidence_strength") or 0.5)
    return min(0.5, base_strength * 0.5)


def compute_confidence_momentum(current_conf: float, delta: float, resistance: float) -> float:
    """
    Part 2 — Gradually evolves confidence based on stability resistance.
    """
    return current_conf + (delta * (1.0 - resistance))


def compute_procedural_immunity(code_dict: dict) -> float:
    """
    Part 4 — Procedural Immunity Calibration (Task 9E).
    Proportional immunity based on grounding quality.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    score = 0.0
    if code_dict.get("PROCEDURE_SURVIVAL_PRIORITY"):
        score += 0.4
    if code_dict.get("PROCEDURE_GROUNDED_BY_WORKFLOW"):
        score += 0.2
    if code_dict.get("PROCEDURE_RECONCILIATION_PROTECTED"):
        score += 0.2
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.2
        
    return min(1.0, score)


# --- Task 9F: Clinical Grounding Fidelity Hardening ---

def compute_exact_context_overlap(description: str, note_text: str) -> float:
    """
    Part 1 — Phrase-Level Grounding Dominance.
    Evaluates exact phrase presence in the note.
    """
    desc = (description or "").lower()
    text = (note_text or "").lower()
    
    if not desc or not text:
        return 0.0

    # 1. Exact match
    if desc in text:
        return 1.0
        
    # 2. Token overlap ratio for key terms
    desc_tokens = [t for t in desc.split() if len(t) > 3]
    if not desc_tokens:
        return 0.0
        
    matches = sum(1 for t in desc_tokens if t in text)
    return matches / len(desc_tokens)


def compute_local_phrase_density(description: str, note_text: str) -> float:
    """
    Evaluates local phrase concentration and modifier proximity.
    """
    desc = (description or "").lower()
    text = (note_text or "").lower()
    
    if not desc or not text:
        return 0.0

    sentences = text.split('.')
    max_density = 0.0
    
    desc_tokens = set(t for t in desc.split() if len(t) > 3)
    if not desc_tokens:
        return 0.0
        
    for s in sentences:
        s_tokens = set(s.split())
        overlap = desc_tokens.intersection(s_tokens)
        density = len(overlap) / len(desc_tokens)
        max_density = max(max_density, density)
        
    return max_density


def compute_phrase_grounding_strength(description: str, note_text: str) -> float:
    """
    Aggregates exact overlap and local density.
    """
    overlap = compute_exact_context_overlap(description, note_text)
    density = compute_local_phrase_density(description, note_text)
    return min(1.0, (overlap * 0.7) + (density * 0.3))


def compute_ontology_dependence_ratio(code_dict: dict) -> float:
    """
    Part 2 — Ontology Drift Suppression.
    Estimates how much a code depends on semantic relationships versus actual textual grounding.
    """
    rel_score = float(code_dict.get("scoring_breakdown", {}).get("relationship_score") or 0.0)
    ev_score = float(code_dict.get("scoring_breakdown", {}).get("evidence_score") or 0.5)
    
    if ev_score == 0: return 1.0
    return rel_score / (rel_score + ev_score + 0.1)


def compute_procedure_subtype_grounding(code_dict: dict, note_text: str) -> float:
    """
    Part 3 — Procedural Subtype Grounding.
    Ensures procedure specificity is backed by workflow and modality evidence.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 1.0
        
    desc = (code_dict.get("description") or "").lower()
    text = (note_text or "").lower()
    
    score = 0.0
    
    # 1. Modality anchoring
    modalities = ["ultrasound", "fluoroscopy", "ct", "mri", "guidance", "contrast", "radiological"]
    for m in modalities:
        if m in desc and m in text:
            score += 0.3
            
    # 2. Qualifier confirmation
    qualifiers = ["bilateral", "unilateral", "initial", "subsequent", "open", "percutaneous", "diagnostic", "interventional"]
    for q in qualifiers:
        if q in desc and q in text:
            score += 0.3
            
    # 3. Workflow indicators
    if code_dict.get("PROCEDURE_GROUNDED_BY_WORKFLOW"):
        score += 0.4
        
    return min(1.0, score)


def compute_local_context_coherence(code_dict: dict, note_text: str) -> float:
    """
    Part 4 — Local Context Isolation.
    Evaluates same-section proximity and nearby modifier relationships.
    """
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "plan", "operative_report", "procedure", "postop_diagnosis"]:
        return 1.0
    if sec in ["history_of_present_illness", "chief_complaint"]:
        return 0.8
    if sec in ["physical_exam", "review_of_systems"]:
        return 0.6
    return 0.4


# --- Task 9G: Final Calibration & Representation Stability ---

def normalize_confidence_scale(val: float) -> float:
    """
    Part 1 — Confidence Scale Normalization.
    Ensures all confidence scores are normalized into stable bounded ranges.
    """
    return clamp_score(round(val, 3))


def bounded_confidence_delta(current_val: float, delta: float, max_change: float = 0.15) -> float:
    """
    Ensures no single reasoning pass causes massive spikes or collapses.
    """
    clamped_delta = max(-max_change, min(max_change, delta))
    return normalize_confidence_scale(current_val + clamped_delta)


def compute_confidence_band(val: float) -> str:
    """
    Categorizes confidence into calibrated bands for transparency and stability.
    """
    if val >= 0.90: return "DEFINITIVE"
    if val >= 0.75: return "STRONGLY_GROUNDED"
    if val >= 0.60: return "MODERATELY_GROUNDED"
    if val >= 0.45: return "WEAKLY_GROUNDED"
    return "UNSTABLE"


def compute_specificity_survival_weight(code_dict: dict, note_text: str) -> float:
    """
    Part 2 — Specificity Survival Calibration.
    Highly specific grounded concepts gain resistance to generic replacement.
    """
    score = 0.0
    phrase_quality = compute_phrase_grounding_strength(code_dict.get("description", ""), note_text)
    
    score += phrase_quality * 0.40
    
    # Specificity markers (localized qualifiers)
    if has_specificity_markers(code_dict.get("description", "")):
        score += 0.20
        
    # Anatomical precision
    if code_dict.get("ANATOMY_REGION_MATCH"):
        score += 0.20
        
    # Section authority
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "plan", "operative_report", "postop_diagnosis"]:
        score += 0.20
        
    return min(1.0, score)


def compute_procedural_stability_weight(code_dict: dict, note_text: str) -> float:
    """
    Part 3 — Procedural Stability Calibration.
    Ensures grounded procedures remain stable across reconciliation.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    score = 0.0
    score += compute_procedure_grounding_strength(code_dict, note_text) * 0.50
    
    if code_dict.get("PROCEDURAL_SUBTYPE_CONFIRMED"):
        score += 0.20
        
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.30
        
    return min(1.0, score)


    return min(1.0, score)


def compute_chronic_relevance_weight(code_dict: dict, note_text: str) -> float:
    """
    Part 4 — Chronic Condition Relevance Calibration (Task 9G).
    Stabilizes survival of actively managed chronic conditions.
    """
    score = compute_management_activity_score(code_dict, note_text)
    
    # Assessment linkage
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "plan", "final_diagnosis"]:
        score += 0.30
        
    # Procedure/Indication relevance
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.20
        
    return min(1.0, score)


# --- Task 9H: Domain Calibration & False Positive Control ---

def compute_false_positive_risk(code_dict: dict, note_text: str) -> float:
    """
    Part 1 — False Positive Survival Control.
    Calibrates survivability pressure based on aggregate risk factors.
    """
    risk = 0.0
    
    # 1. Ontology dependence vs direct grounding
    dependence = compute_ontology_dependence_ratio(code_dict)
    if dependence > 0.70: risk += 0.30
    
    # 2. Grounding density
    density = compute_local_phrase_density(code_dict.get("description") or "", note_text)
    if density < 0.30: risk += 0.30
    
    # 3. Section authority
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["history", "past_medical_history", "review_of_systems", "social_history"]:
        risk += 0.20
        
    # 4. Encounter relevance
    relevance = float(code_dict.get("encounter_relevance") or 0.5)
    if relevance < 0.40: risk += 0.20
    
    return min(1.0, risk)


def compute_domain_calibration_weight(code_dict: dict, note_text: str) -> float:
    """
    Part 2 — Domain-Specific Calibration.
    Strengthens survivability of grounded domain-specific concepts.
    """
    weight = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # Phrase authority
    phrase_strength = compute_phrase_grounding_strength(desc, note_text)
    weight += phrase_strength * 0.40
    
    # Subtype grounding
    if code_dict.get("PROCEDURAL_SUBTYPE_CONFIRMED") or code_dict.get("SPECIFICITY_RESISTANCE_GRANTED"):
        weight += 0.30
        
    # Domain proximity (modifiers/terms)
    if any(m in desc for m in ["acute", "severe", "persistent", "recurrent", "bilateral", "unstable"]):
        weight += 0.30
        
    return min(1.0, weight)


def compute_sibling_grounding_advantage(code_a: dict, code_b: dict, note_text: str) -> float:
    """
    Part 3 — Sibling Replacement Stabilization.
    Compares two siblings and returns the grounding advantage of A over B.
    """
    phrase_a = compute_phrase_grounding_strength(code_a.get("description") or "", note_text)
    phrase_b = compute_phrase_grounding_strength(code_b.get("description") or "", note_text)
    
    advantage = phrase_a - phrase_b
    
    # Specificity resistance
    if code_a.get("SPECIFICITY_RESISTANCE_GRANTED") and not code_b.get("SPECIFICITY_RESISTANCE_GRANTED"):
        advantage += 0.20
    elif code_b.get("SPECIFICITY_RESISTANCE_GRANTED") and not code_a.get("SPECIFICITY_RESISTANCE_GRANTED"):
        advantage -= 0.20
        
    # Temporal status
    if code_a.get("temporal_status") == "ACTIVE" and code_b.get("temporal_status") != "ACTIVE":
        advantage += 0.15
    elif code_b.get("temporal_status") == "ACTIVE" and code_a.get("temporal_status") != "ACTIVE":
        advantage -= 0.15
        
    return advantage


def compute_procedural_domain_strength(code_dict: dict, note_text: str) -> float:
    """
    Part 4 — Procedural Domain Hardening (Task 9H).
    Aggregates workflow, modality, and indication evidence for CPT stability.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    strength = 0.0
    strength += compute_procedure_subtype_grounding(code_dict, note_text) * 0.40
    
    if code_dict.get("PROCEDURE_GROUNDED_BY_WORKFLOW"):
        strength += 0.30
        
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        strength += 0.30
        
    return min(1.0, strength)


# --- Task 9I: Pipeline Lockdown & Regression-Stability ---

def compute_regression_resistance(code_dict: dict, note_text: str) -> float:
    """
    Part 1 — Regression Resistance.
    Stability reinforcement based on grounding quality and domain coherence.
    """
    score = 0.0
    
    # Direct grounding authority
    if code_dict.get("PHRASE_GROUNDING_CONFIRMED"):
        score += 0.30
        
    # Specificity survival
    if float(code_dict.get("SPECIFICITY_SURVIVAL_WEIGHT") or 0) > 0.80:
        score += 0.20
        
    # Procedural stability
    if float(code_dict.get("PROCEDURAL_STABILITY_WEIGHT") or 0) > 0.80:
        score += 0.20
        
    # Temporal validity
    if code_dict.get("temporal_status") == "ACTIVE":
        score += 0.15
        
    # Encounter relevance
    if float(code_dict.get("encounter_relevance") or 0.5) > 0.80:
        score += 0.15
        
    return min(1.0, score)


def compute_procedural_immunity_lock(code_dict: dict, note_text: str) -> float:
    """
    Part 3 — Procedural Immunity Lock.
    Ensures grounded procedures resist collapse and preserve fidelity.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    score = 0.0
    # Subtype grounding quality
    score += compute_procedure_subtype_grounding(code_dict, note_text) * 0.40
    
    # Workflow and modality
    if code_dict.get("PROCEDURE_GROUNDED_BY_WORKFLOW"):
        score += 0.30
        
    # Indication coherence
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.30
        
    return min(1.0, score)


def compute_specificity_immunity_lock(code_dict: dict, note_text: str) -> float:
    """
    Part 4 — Specificity Immunity Lock.
    Protects strongly grounded specific variants from generic replacement.
    """
    score = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # Subtype phrase grounding
    if code_dict.get("PHRASE_GROUNDING_CONFIRMED"):
        score += 0.40
        
    # Modifier density
    if has_specificity_markers(desc):
        score += 0.20
        
    # Anatomical specificity
    if code_dict.get("ANATOMY_REGION_MATCH"):
        score += 0.20
        
    # Procedural coherence
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.20
        
    return min(1.0, score)


# --- Task 10A: Principal Encounter Dominance Hardening ---

def compute_principal_encounter_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Principal Encounter Driver Scoring.
    Calculates a generalized encounter-driving score based on clinical authority.
    """
    score = 0.0
    
    # 1. SECTION AUTHORITY
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "impression", "final_diagnosis", "operative_diagnosis", "discharge_diagnosis"]:
        score += 0.40
    elif sec in ["plan", "procedure_findings", "postop_diagnosis"]:
        score += 0.30
    elif sec in ["chief_complaint", "history_of_present_illness"]:
        score += 0.15
        
    # 2. PROCEDURAL LINKAGE
    if code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.30
    if code_dict.get("PROCEDURE_SURVIVAL_PRIORITY"):
        score += 0.20
        
    # 3. ACTIVE MANAGEMENT
    # Uses management_activity_score (0-0.5) if already calculated
    management = float(code_dict.get("management_activity_score") or 0)
    score += management * 0.40
    
    # 4. ACUITY / SEVERITY
    desc = (code_dict.get("description") or "").lower()
    acuity_terms = [
        "hemorrhage", "acute", "failure", "pathological", "sepsis", 
        "nephritis", "ketoacidosis", "exacerbation", "rupture", "unstable"
    ]
    if any(term in desc for term in acuity_terms):
        score += 0.15
        
    # 5. LOCAL CONTEXT DENSITY
    density = compute_local_phrase_density(desc, note_text)
    score += density * 0.15
    
    return min(1.0, score)


def compute_generalization_penalty(code_dict: dict) -> float:
    """
    Step 6 — Generalization Penalty (Task 10A).
    Penalizes unspecified, NOS, and broad fallback concepts.
    """
    desc = (code_dict.get("description") or "").lower()
    penalty = 0.0
    
    if "unspecified" in desc or "nos" in desc or "not otherwise specified" in desc:
        penalty += 0.35
    if "other" in desc and "specified" not in desc:
        penalty += 0.20
    if "symptom" in desc or "sign" in desc:
        penalty += 0.15
        
    return penalty


# --- Task 10B: Clinical Evidence Synthesis Hardening ---

def compute_distributed_evidence_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Distributed Evidence Synthesis.
    Aggregates signals from labs, procedures, treatments, imaging, and physiology.
    Generalized evidence aggregation, NOT disease-specific templates.
    """
    score = 0.0
    text = note_text.lower()
    
    # 1. Laboratory Signals
    lab_terms = ["gap", "ketones", "lactate", "creatinine", "hemoglobin", "wbc", "troponin", "abg", "proteinuria", "aki", "hyperglycemia", "electrolytes"]
    for term in lab_terms:
        if term in text: score += 0.05
        
    # 2. Procedural Signals
    proc_terms = ["thrombectomy", "hemostasis", "clipping", "biopsy", "drainage", "stenting", "catheter", "dialysis", "thoracentesis", "resection"]
    for term in proc_terms:
        if term in text: score += 0.10
        
    # 3. Treatment Signals
    rx_terms = ["infusion", "transfusion", "antibiotics", "chemotherapy", "oxygen", "anticoagulation", "steroids", "insulin", "pressors"]
    for term in rx_terms:
        if term in text: score += 0.08
        
    # 4. Imaging Signals
    img_terms = ["obstruction", "hemorrhage", "infarction", "effusion", "fracture", "mass", "lesion", "consolidation"]
    for term in img_terms:
        if term in text: score += 0.08
        
    # 5. Physiologic Signals
    phys_terms = ["hypoxia", "hypotension", "tachycardia", "fever", "altered mental status", "shock"]
    for term in phys_terms:
        if term in text: score += 0.05
        
    return min(0.60, score)


def compute_supporting_evidence_diversity(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Supporting Evidence Weighting.
    Measures diversity of independent evidence streams.
    """
    streams = 0
    text = note_text.lower()
    
    # Text (phrase match)
    if code_dict.get("PHRASE_GROUNDING_CONFIRMED"): streams += 1
    
    # Labs
    if any(t in text for t in ["gap", "ketones", "lactate", "creatinine", "hemoglobin", "wbc"]): streams += 1
    
    # Meds/Treatment
    if any(t in text for t in ["insulin", "transfusion", "antibiotics", "chemo", "steroids", "oxygen"]): streams += 1
    
    # Procedures
    if (code_dict.get("type") or "").upper() == "CPT" or code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"): streams += 1
    
    # Imaging
    if any(t in text for t in ["ct", "mri", "x-ray", "ultrasound", "imaging", "finding"]): streams += 1
    
    return min(1.0, streams * 0.25)


def compute_severity_preservation_strength(code_dict: dict) -> float:
    """
    Step 6 — Clinical Severity Preservation.
    Protects severity-rich diagnoses from generic abstraction.
    """
    desc = (code_dict.get("description") or "").lower()
    score = 0.0
    
    severity_markers = [
        "hemorrhage", "failure", "ketoacidosis", "pathological", 
        "acute on chronic", "organism", "complication", "obstruction",
        "severe", "unstable", "exacerbation"
    ]
    for marker in severity_markers:
        if marker in desc: score += 0.25
        
    return min(1.0, score)


# --- Task 10C: Clinical Uncertainty & Diagnostic Certainty Hardening ---

def compute_diagnostic_certainty(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Diagnostic Certainty Scoring.
    Evaluates assertion strength, multi-signal confirmation, and specialty authority.
    Conservative baseline for clinical defensibility.
    """
    score = 0.50 # Baseline
    text = note_text.lower()
    
    # 1. Direct Assertion Strength
    strong_assertions = ["diagnosed with", "confirmed", "final diagnosis", "consistent with", "evident", "manifested by"]
    weak_assertions = ["possible", "may represent", "likely", "suggestive", "rule out", "suspected", "potential", "questionable"]
    
    for term in strong_assertions:
        if term in text: score += 0.15
    for term in weak_assertions:
        if term in text: score -= 0.20
        
    # 2. Multi-Signal Confirmation
    diversity = compute_supporting_evidence_diversity(code_dict, note_text)
    score += diversity * 0.25
    
    # 3. Procedural Confirmation
    if code_dict.get("PROCEDURAL_INTENT_CONFIRMED") or code_dict.get("PROCEDURAL_INDICATION_CONFIRMED"):
        score += 0.15
        
    # 4. Specialty Authority
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "final_diagnosis", "discharge_diagnosis", "operative_diagnosis", "impression"]:
        score += 0.20
        
    # 5. Negative Indicators
    if code_dict.get("HARD_TEMPORAL_LOCK") or "resolved" in text:
        score -= 0.30
        
    return min(1.0, max(0.0, score))


def compute_procedural_subtype_certainty(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Procedural Subtype Uncertainty.
    Measures grounding quality for highly specific procedural variants.
    Prevents hallucinated specificity in CPT selection.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 1.0 # Not a procedure
        
    score = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # Subtype phrase grounding quality
    if code_dict.get("PHRASE_GROUNDING_CONFIRMED"): score += 0.40
    
    # Context coherence
    if float(code_dict.get("PROCEDURAL_DOMAIN_STRENGTH") or 0) > 0.8: score += 0.30
    
    # Subtype-specific keywords in description
    subtype_keywords = ["guided", "catheter", "infusion", "interventional", "percutaneous", "approach", "modality"]
    for kw in subtype_keywords:
        if kw in desc:
            # Check if this keyword is actually in the note (phrase level)
            if kw in note_text.lower(): score += 0.15
            
    return min(1.0, score)


def compute_relationship_confidence(code_a: dict, code_b: dict, note_text: str) -> float:
    """
    Step 5 — Relationship Confidence Model.
    Measures the strength of the link between two related codes.
    Prevents forced causality and weak complication linkage.
    """
    confidence = 0.0
    text = note_text.lower()
    
    # 1. Section Coherence
    if code_a.get("section_dominant") == code_b.get("section_dominant"):
        confidence += 0.35
        
    # 2. Causal Terminology
    causal_terms = ["due to", "secondary to", "caused by", "resulting in", "complicated by", "with manifestation", "manifested by"]
    for term in causal_terms:
        if term in text: confidence += 0.25
        
    # 3. Intervention Alignment
    if code_a.get("PROCEDURAL_INTENT_CONFIRMED") or code_b.get("PROCEDURAL_INTENT_CONFIRMED"):
        confidence += 0.20
        
    # 4. Evidence Convergence
    if code_a.get("EVIDENCE_CONVERGENCE_DETECTED") and code_b.get("EVIDENCE_CONVERGENCE_DETECTED"):
        confidence += 0.20
        
    return min(1.0, confidence)


# --- Task 10D: Representation Consistency & Conflict Governance ---

def compute_representation_family(code_dict: dict) -> str:
    """
    Step 1 — Representation Family Governance.
    Identifies the semantic family prefix for grouping related concepts.
    """
    code = (code_dict.get("code") or "").strip().upper()
    if not code: return "UNKNOWN"
    # Family prefix (3-char for ICD, 3-char for CPT)
    return code[:3]


def compute_semantic_overlap_strength(code_a: dict, code_b: dict) -> float:
    """
    Step 4 — Duplicate Semantic Collapse.
    Measures clinical redundancy between two codes.
    """
    desc_a = (code_a.get("description") or "").lower()
    desc_b = (code_b.get("description") or "").lower()
    
    if not desc_a or not desc_b: return 0.0
    if desc_a == desc_b: return 1.0
    
    # Word-level jaccard
    words_a = set(re.findall(r'\w+', desc_a)) - {"unspecified", "nos", "with", "and", "other"}
    words_b = set(re.findall(r'\w+', desc_b)) - {"unspecified", "nos", "with", "and", "other"}
    if not words_a or not words_b: return 0.0
    
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    
    # Hierarchical containment boost
    if is_parent_of(code_a.get("code", ""), code_b.get("code", "")) or is_parent_of(code_b.get("code", ""), code_a.get("code", "")):
        jaccard += 0.40
        
    return min(1.0, jaccard)


def compute_consistency_priority(code_dict: dict) -> float:
    """
    Step 7 — Consistency Priority Model.
    Prioritizes coherent integrated representations over raw ontology breadth.
    """
    priority = 0.0
    
    # 1. Coherent Integrated/Combination
    if code_dict.get("COMBINATION_DOMINANCE_ACTIVE"): 
        priority += 0.50
    
    # 2. Principal Encounter Structure
    if code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"): 
        priority += 0.40
    
    # 3. Grounded Procedures
    if float(code_dict.get("PROCEDURAL_DOMAIN_STRENGTH") or 0) > 0.8: 
        priority += 0.30
    
    # 4. Actively Managed
    if float(code_dict.get("management_activity_score") or 0) > 0.3: 
        priority += 0.20
    
    # 5. Severity (preservation)
    if compute_severity_preservation_strength(code_dict) > 0.5: 
        priority += 0.15
        
    return priority


# --- Task 10E: Coding Policy & Billing Realism Hardening ---

def compute_reportability_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Reportability Scoring.
    Evaluates active management, encounter relevance, and clinical significance.
    Conservative billing-governance scoring.
    """
    score = 0.0
    
    # 1. Active Management
    mgmt = float(code_dict.get("management_activity_score") or 0)
    score += mgmt * 0.50
    
    # 2. Encounter Relevance
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "final_diagnosis", "discharge_diagnosis", "postop_diagnosis", "preop_diagnosis", "impression", "findings"]:
        score += 0.40
    elif sec in ["plan", "operative_findings", "procedure"]:
        score += 0.20
        
    # 3. Principal/Combination status
    if code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"): score += 0.15
    if code_dict.get("COMBINATION_DOMINANCE_ACTIVE"): score += 0.10
    
    # 4. Multi-signal support
    diversity = compute_supporting_evidence_diversity(code_dict, note_text)
    score += diversity * 0.15
    
    return min(1.0, score)


def compute_independent_management_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Independent Management Detection.
    Detects evidence of standalone treatment or specialist oversight.
    """
    strength = 0.0
    text = note_text.lower()
    
    # 1. Specialist keywords
    specialists = ["consult", "specialist", "referred to", "managed by", "nephrology", "cardiology", "oncology", "neurology"]
    for s in specialists:
        if s in text: strength += 0.10
        
    # 2. Targeted medications/management
    mgmt = float(code_dict.get("management_activity_score") or 0)
    strength += mgmt * 0.60
    
    # 3. Dedicated workup/monitoring
    workup = ["ordered", "monitored", "titrated", "adjusted", "stable on", "refractory"]
    for w in workup:
        if w in text: strength += 0.05
        
    return min(1.0, strength)


def compute_clinical_significance_priority(code_dict: dict) -> float:
    """
    Step 6 — Clinical Significance Priority.
    Prioritizes billable impactful conditions over incidental findings.
    """
    priority = 0.0
    
    # Acute/Severe
    if compute_severity_preservation_strength(code_dict) > 0.5: priority += 0.40
    
    # Principal/Intervention
    if code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"): priority += 0.30
    if float(code_dict.get("PROCEDURAL_DOMAIN_STRENGTH") or 0) > 0.8: priority += 0.20
    
    # Managed Chronic
    if float(code_dict.get("management_activity_score") or 0) > 0.3: priority += 0.10
    
    return priority


# --- Task: Hierarchical Coding Compliance & Encounter Attribution Hardening ---

def compute_encounter_attribution_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Encounter Attribution Governance.
    Determines if a condition is responsible for the current encounter burden.
    """
    score = 0.0
    sec = (code_dict.get("section_dominant") or "").lower()
    
    # 1. Assessment/Plan prominence
    if sec in ["assessment", "plan", "final_diagnosis", "discharge_diagnosis"]:
        score += 0.50
    elif sec in ["impression", "operative_diagnosis"]:
        score += 0.35
        
    # 2. Treatment/Management linkage
    mgmt = float(code_dict.get("management_activity_score") or 0)
    score += mgmt * 0.40
    
    # 3. Principal diagnosis lock
    if code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"):
        score += 0.15
        
    return min(1.0, score)


def resolve_condition_temporal_state(code_dict: dict, note_text: str) -> str:
    """
    Step 2 — Temporal Clinical State Resolution.
    Distinguishes active disease from historical/resolved mentions.
    """
    # v15: Pass the note text and section to detect_temporal_status
    status = detect_temporal_status(note_text, code_dict.get("section_dominant", ""))
    
    if status == "HISTORICAL": return "HISTORICAL"
    if status == "RESOLVED": return "RESOLVED"
    
    desc = (code_dict.get("description") or "").lower()
    if any(u in desc for u in ["possible", "rule out", "suspected", "probable"]):
        return "SUSPECTED"
        
    if any(c in desc for c in ["chronic", "stable on", "maintained", "long term"]):
        return "CHRONIC_ACTIVE"
        
    return "ACTIVE"


def compute_management_intensity_score(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Management Intensity Scoring.
    Weighted scoring of diagnostic and therapeutic burden.
    """
    score = 0.0
    text = note_text.lower()
    
    # 1. Medication intensity
    mgmt = float(code_dict.get("management_activity_score") or 0)
    score += mgmt * 0.50
    
    # 2. Specialist consultation
    if any(s in text for s in ["consult", "referred", "specialist"]):
        score += 0.15
        
    # 3. Diagnostic escalation
    if any(d in text for d in ["ordered", "imaging", "labs", "biopsy"]):
        score += 0.15
        
    # 4. Monitoring frequency
    if any(m in text for m in ["monitored", "reassessed", "serial", "daily"]):
        score += 0.20
        
    return min(1.0, score)


def detect_mutually_exclusive_conditions(code_a: dict, code_b: dict) -> bool:
    """
    Step 5 — Mutually Exclusive Diagnosis Arbitration.
    Identifies clinically incompatible diagnoses.
    """
    desc_a = (code_a.get("description") or "").lower()
    desc_b = (code_b.get("description") or "").lower()
    
    # Case: Sepsis vs Rule-out infection
    if "sepsis" in desc_a and "infection" in desc_b and ("ruled out" in desc_b or "possible" in desc_b):
        return True
    if "sepsis" in desc_b and "infection" in desc_a and ("ruled out" in desc_a or "possible" in desc_a):
        return True
            
    return False


def compute_complication_hierarchy_strength(code_dict: dict) -> float:
    """
    Step 6 — Hierarchical Complication Governance.
    Measures audit-defensibility of CC/MCC conditions.
    """
    score = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    complications = ["failure", "sepsis", "encephalopathy", "ketoacidosis", "crisis", "shock"]
    if any(c in desc for c in complications):
        score += 0.40
        
    if code_dict.get("EVIDENCE_CONVERGENCE_DETECTED"):
        score += 0.30
        
    mgmt = float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0)
    score += mgmt * 0.30
    
    return min(1.0, score)


# --- Task: Clinical Evidence Integrity & Documentation Defensibility Hardening ---

def build_evidence_provenance_graph(code_dict: dict, note_text: str) -> dict:
    """
    Step 1 — Evidence Provenance Graph.
    Tracks lineage of documentation support for audit defensibility.
    """
    return {
        "section": code_dict.get("section_dominant", "UNKNOWN"),
        "management": float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0) > 0.4,
        "procedure": code_dict.get("PROCEDURAL_INTENT_CONFIRMED", False),
        "convergence": code_dict.get("EVIDENCE_CONVERGENCE_DETECTED", False),
        "temporal": code_dict.get("TEMPORAL_STATE", "ACTIVE")
    }


def compute_documentation_confidence(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Documentation Confidence Scoring.
    Evaluates definitive vs speculative wording and explicit commitment.
    """
    score = 0.50 # Baseline
    desc = (code_dict.get("description") or "").lower()
    
    # Penalize speculative wording
    speculative = ["possible", "suspected", "rule out", "probable", "differential", "likely"]
    if any(s in desc for s in speculative):
        score -= 0.35
        
    # Boost definitive Assessment/Plan prominence
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["assessment", "plan", "final_diagnosis", "discharge_diagnosis"]:
        score += 0.25
        
    # Repetition stability (multi-signal)
    if code_dict.get("EVIDENCE_CONVERGENCE_DETECTED"):
        score += 0.15
        
    return min(1.0, max(0.0, score))


def compute_objective_evidence_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Objective Evidence Corroboration.
    Correlates objective signals with clinical diagnoses.
    """
    strength = 0.0
    
    # 1. Management intensity (proxy for corroboration)
    intensity = float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0)
    strength += intensity * 0.60
    
    # 2. Section priority (where objective findings are reported)
    sec = (code_dict.get("section_dominant") or "").lower()
    if sec in ["operative_findings", "procedure", "plan", "assessment"]:
        strength += 0.25
        
    # 3. Grounding confirmation
    if code_dict.get("PHRASE_GROUNDING_CONFIRMED"):
        strength += 0.15
        
    return min(1.0, strength)


def resolve_provider_intent_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 5 — Provider Intent Resolution.
    Distinguishes active commitment from speculative consideration.
    """
    intent = 0.0
    sec = (code_dict.get("section_dominant") or "").lower()
    
    if sec in ["assessment", "plan", "final_diagnosis", "discharge_diagnosis"]:
        intent += 0.60
    elif sec in ["hpi", "history", "subjective"]:
        intent += 0.25
        
    desc = (code_dict.get("description") or "").lower()
    if "differential" in desc or "working" in desc:
        intent -= 0.20
        
    return min(1.0, max(0.0, intent))


def compute_cross_document_consistency(code_dict: dict, note_text: str) -> float:
    """
    Step 6 — Cross-Document Consistency Governance.
    Measures documentation stability across the encounter.
    """
    consistency = 0.50
    if code_dict.get("EVIDENCE_CONVERGENCE_DETECTED"):
        consistency += 0.35
    if code_dict.get("CROSS_SECTION_SUPPORT"):
        consistency += 0.15
    return min(1.0, consistency)


# --- Task: Probabilistic Clinical Reasoning & Longitudinal Encounter Evolution Hardening ---

def compute_evidence_temporal_decay(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Temporal Evidence Decay Governance.
    Models the decay of early provisional evidence in evolving encounters.
    """
    decay = 1.0
    sec = (code_dict.get("section_dominant") or "").lower()
    
    # Early sections decay if not supported later (proxy: low convergence)
    if sec in ["hpi", "subjective", "triage", "emergency_department"]:
        if not code_dict.get("EVIDENCE_CONVERGENCE_DETECTED"):
            decay *= 0.65
            
    # Resolve decay
    if code_dict.get("TEMPORAL_STATE") == "RESOLVED":
        decay *= 0.45
        
    return decay


def compute_provider_authority_weight(code_dict: dict) -> float:
    """
    Step 2 — Provider Authority Weighting.
    Prioritizes higher-authority documentation in reconciliation.
    """
    weight = 0.55 # Baseline
    sec = (code_dict.get("section_dominant") or "").lower()
    
    if any(k in sec for k in ["attending", "final", "discharge"]):
        weight = 1.0
    elif any(k in sec for k in ["consult", "specialist", "operative"]):
        weight = 0.85
    elif any(k in sec for k in ["nursing", "observation", "triage"]):
        weight = 0.35
        
    return weight


def compute_discharge_finality_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Discharge Finality Governance.
    Evaluates alignment with discharge reconciliation.
    """
    strength = 0.0
    sec = (code_dict.get("section_dominant") or "").lower()
    
    if sec in ["discharge_diagnosis", "discharge_summary", "final_diagnosis"]:
        strength += 0.85
        
    if code_dict.get("PRINCIPAL_DIAGNOSIS_CONFIRMED"):
        strength += 0.15
        
    return strength


def compute_probabilistic_diagnostic_confidence(code_dict: dict) -> float:
    """
    Step 4 — Probabilistic Diagnostic Confidence Fusion.
    Fuses multiple dimensions into a unified probabilistic score.
    """
    factors = {
        "DOCUMENTATION_CONFIDENCE_VAL": 0.25,
        "OBJECTIVE_EVIDENCE_VAL": 0.25,
        "ENCOUNTER_ATTRIBUTION_VAL": 0.20,
        "MANAGEMENT_INTENSITY_VAL": 0.15,
        "CROSS_DOCUMENT_CONSISTENCY_VAL": 0.15
    }
    
    fusion = 0.0
    for key, weight in factors.items():
        fusion += float(code_dict.get(key) or 0.5) * weight
        
    decay = float(code_dict.get("TEMPORAL_DECAY_VAL") or 1.0)
    return min(1.0, fusion * decay)


def track_encounter_state_evolution(code_dict: dict) -> str:
    """
    Step 5 — Longitudinal Encounter State Evolution.
    Tracks diagnosis transitions.
    """
    state = code_dict.get("TEMPORAL_STATE", "ACTIVE")
    attribution = float(code_dict.get("ENCOUNTER_ATTRIBUTION_VAL") or 0)
    
    if state == "ACTIVE" and attribution > 0.85:
        return "STABILIZED_DRIVER"
    if state == "ACTIVE" and attribution < 0.45:
        return "EMERGING_CONCEPT"
    if state in ["RESOLVED", "HISTORICAL"]:
        return "LEGACY_FINDING"
        
    return "TRANSITIONAL_STATE"


def resolve_multi_provider_conflict(code_a: dict, code_b: dict) -> int:
    """
    Step 6 — Multi-Provider Conflict Arbitration.
    Returns 1 if code_a wins, -1 if code_b wins, 0 if tie.
    """
    auth_a = compute_provider_authority_weight(code_a)
    auth_b = compute_provider_authority_weight(code_b)
    
    if auth_a > auth_b + 0.15: return 1
    if auth_b > auth_a + 0.15: return -1
    
    conf_a = float(code_a.get("PROBABILISTIC_CONFIDENCE_VAL") or 0.5)
    conf_b = float(code_b.get("PROBABILISTIC_CONFIDENCE_VAL") or 0.5)
    
    if conf_a > conf_b + 0.10: return 1
    if conf_b > conf_a + 0.10: return -1
    
    return 0


# --- Task: Regulatory Coding Compliance & Guideline-Aware Semantic Governance ---

def build_guideline_reference_map(code_dict: dict) -> list[str]:
    """
    Step 1 — Guideline Reference Mapping.
    Tracks linkage to official coding guidelines and conventions.
    """
    guidelines = []
    code = (code_dict.get("code") or "").upper()
    
    if code_dict.get("ETIOLOGY_MANIFESTATION_LINKED"):
        guidelines.append("ICD-10-CM Guideline I.B.7 (Etiology/Manifestation)")
        
    desc = (code_dict.get("description") or "").lower()
    if any(u in desc for u in ["possible", "suspected", "rule out"]):
        guidelines.append("ICD-10-CM Guideline II.H (Uncertain Diagnosis)")
        
    if code_dict.get("PRINCIPAL_DIAGNOSIS_CONFIRMED"):
        guidelines.append("ICD-10-CM Guideline II (Principal Diagnosis Selection)")
        
    return guidelines


def resolve_etiology_manifestation_relationship(code_a: dict, code_b: dict) -> bool:
    """
    Step 2 — Etiology-Manifestation Relationship Governance.
    Identifies semantic etiology/manifestation linkage.
    """
    c1 = (code_a.get("code") or "").upper()
    c2 = (code_b.get("code") or "").upper()
    
    for eti, mani in MANIFESTATION_OVERLAP_PAIRS:
        if c1.startswith(eti) and c2.startswith(mani): return True
        if c2.startswith(eti) and c1.startswith(mani): return True
            
    return False


def compute_sequencing_confidence(code_dict: dict, position: int) -> float:
    """
    Step 3 — Sequencing Governance.
    Validates ordering conventions.
    """
    score = 0.50
    if position == 0 and code_dict.get("PRINCIPAL_DIAGNOSIS_CONFIRMED"):
        score += 0.45
    return score


def resolve_encounter_setting_policy(note_text: str) -> str:
    """
    Step 4 — Inpatient vs Outpatient Policy Arbitration.
    Resolves encounter setting to apply appropriate coding rules.
    """
    text = note_text.lower()
    if any(k in text for k in ["discharge summary", "hospitalist", "admission", "inpatient", "floor"]):
        return "INPATIENT"
    return "OUTPATIENT"


def resolve_uncertain_diagnosis_policy(code_dict: dict, setting: str) -> str:
    """
    Step 5 — Uncertain Diagnosis Compliance Governance.
    Differentiates inpatient vs outpatient uncertainty handling.
    """
    desc = (code_dict.get("description") or "").lower()
    is_uncertain = any(u in desc for u in ["possible", "suspected", "rule out", "probable", "likely"])
    
    if is_uncertain:
        if setting == "INPATIENT":
            return "REPORTABLE_PER_GUIDELINE_II_H"
        else:
            return "SUPPRESS_PER_OUTPATIENT_GUIDELINE"
            
    return "DEFINITIVE"


def compute_risk_adjustment_significance(code_dict: dict) -> float:
    """
    Step 6 — HCC/RAF & Risk Adjustment Governance.
    Measures chronic burden and MEAT reinforcement.
    """
    score = 0.0
    mgmt = float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0)
    score += mgmt * 0.55
    
    if code_dict.get("TEMPORAL_STATE") == "CHRONIC_ACTIVE":
        score += 0.35
        
    return score


def classify_prediction_failure(pred: dict, gt_codes: list[str]) -> str:
    """
    Step 1 — Failure Taxonomy Framework (Task 15).
    Categorizes incorrect predictions for empirical precision hardening.
    """
    code = (pred.get("code") or "").upper()
    desc = (pred.get("description") or "").lower()
    traces = pred.get("audit_traces", [])
    
    # 1. Principal Diagnosis Miss
    if pred.get("PRINCIPAL_DIAGNOSIS_CONFIRMED") and code not in gt_codes:
        return "principal_diagnosis_miss"
        
    # 2. Symptom Overcoding
    if "symptom" in desc or "pain" in desc:
        return "symptom_overcoding"
        
    # 3. Temporal Leakage
    if pred.get("TEMPORAL_STATE") in ["HISTORICAL", "RESOLVED"]:
        return "temporal_leakage"
        
    # 4. Negation Failure
    if "no " in desc or "negative" in desc or "denies" in desc:
        return "negation_failure"
        
    # 5. Uncertain Diagnosis Policy Violation
    if any(u in desc for u in ["possible", "suspected", "rule out", "likely", "probable"]):
        return "uncertain_diagnosis_policy_violation"
        
    # 6. Etiology Linkage Failure
    if pred.get("ETIOLOGY_ROLE") or pred.get("MANIFESTATION_ROLE"):
        return "etiology_linkage_failure"

    # 7. Abbreviation Ambiguity
    if "ABBREVIATION_AMBIGUITY_DETECTED" in traces:
        return "abbreviation_ambiguity"

    # 8. Sequencing Violation
    if "SEQUENCING_AUDIT_RISK_DETECTED" in traces:
        return "sequencing_violation"

    # 9. Duplicate Representation Conflict
    if "FAILURE_PATTERN_DETECTED: duplicate_conflict" in traces:
        return "duplicate_representation_conflict"

    # 10. Section Attribution Failure
    if float(pred.get("ENCOUNTER_ATTRIBUTION_VAL") or 0) < 0.3:
        return "section_attribution_failure"

    # 11. Medication Inference Error
    if "medication" in desc or any(m in desc for m in ["tab", "mg", "dose"]):
         return "medication_inference_error"

    # 12. Discharge Reconciliation Failure
    if "DISCHARGE_RECONCILIATION_APPLIED" not in traces and pred.get("TEMPORAL_STATE") == "ACTIVE":
         return "discharge_reconciliation_failure"
        
    return "unsupported_diagnosis_retention"


# --- Task: Clinical NLP Reliability & Note Normalization Hardening ---

def compute_document_reliability(section_name: str, text: str) -> float:
    """
    Step 1 — Document Reliability Model (Task 11F).
    Scores the authoritative weight of a note segment.
    """
    score = 0.50 # Baseline
    sec = section_name.lower()
    
    # 1. Section Authority
    if any(k in sec for k in ["assessment", "plan", "operative", "discharge", "attending"]):
        score += 0.35
    elif any(k in sec for k in ["pmh", "problem_list", "history", "nursing", "imported"]):
        score -= 0.20
        
    # 2. Authoritative Language (proxy: presence of definitive verbs)
    if re.search(r'\b(confirmed|diagnosed|revealed|demonstrated|indicated|performed)\b', text.lower()):
        score += 0.15
        
    # 3. Contradiction Risk (proxy: mixed certainty)
    if any(k in text.lower() for k in ["possible", "suspected", "rule out", "unlikely"]):
        score -= 0.15
        
    return min(1.0, max(0.0, score))


def compute_copy_forward_probability(text: str, history: list[str]) -> float:
    """
    Step 3 — Template / Copy-Forward Suppression (Task 11F).
    Detects repeated blocks across documentation history.
    """
    if not text or not history: return 0.0
    
    max_overlap = 0.0
    for prev in history:
        # Check for large unchanged blocks (proxy: substring or Jaccard if we had it)
        if len(text) > 100 and text in prev:
            max_overlap = 0.85
            break
            
    return max_overlap


def compute_noise_tolerance_strength(text: str) -> float:
    """
    Step 7 — Real-World Noise Tolerance (Task 11F).
    Measures resilience against OCR, dictation, and formatting artifacts.
    """
    strength = 1.0
    if not text: return 0.0
    
    # 1. OCR Merged Tokens
    if re.search(r'[a-z][A-Z]', text): strength -= 0.15
    # 2. Punctuation/Spacing Corruption
    if re.search(r'[^a-zA-Z0-9\s]{3,}', text): strength -= 0.20
    # 3. Capitalization Inconsistency
    if text.isupper() and len(text) > 50: strength -= 0.10
    
    return min(1.0, max(0.0, strength))


def compute_abbreviation_disambiguation_confidence(abbr: str, context_dict: dict) -> float:
    """
    Step 5 — Strengthened Abbreviation Governance (Task 11F).
    Avoids unsafe expansion using multi-signal alignment.
    """
    confidence = 0.40 # Reduced baseline for safety
    abbr = abbr.upper().strip()
    note_text = context_dict.get("NOTE_TEXT", "").upper()
    specialty = context_dict.get("PROVIDER_SPECIALTY", "").upper()
    
    # Rule: Expansion requires local terminology reinforcement OR procedural alignment
    
    if abbr == "MS":
        if "MULTIPLE SCLEROSIS" in note_text: confidence += 0.55
        elif "MITRAL STENOSIS" in note_text: confidence += 0.50
        elif "NEUROLOGY" in specialty: confidence += 0.25
        elif "CARDIOLOGY" in specialty: confidence += 0.25
        
    if abbr == "PE":
        if "PULMONARY EMBOLISM" in note_text: confidence += 0.55
        elif "PHYSICAL EXAM" in note_text: confidence += 0.30 # Common but lower billing impact
        elif "CTPA" in note_text or "HEPARIN" in note_text: confidence += 0.40 # Supportive treatment
        
    if abbr == "SOB":
        if "SHORTNESS OF BREATH" in note_text: confidence += 0.55
        elif "DYSPNEA" in note_text: confidence += 0.45
        
    return min(1.0, confidence)


def compute_section_reliability_weight(section_name: str) -> float:
    """
    Step 2 — Section Reliability Governance.
    Prioritizes reliable documentation sources.
    """
    sec = section_name.lower()
    weights = {
        "discharge_summary": 1.0,
        "assessment_plan": 0.95,
        "final_diagnosis": 0.95,
        "consult_note": 0.90,
        "hpi": 0.85,
        "nursing_note": 0.60,
        "imported_problem_list": 0.40,
        "pmh": 0.45
    }
    return weights.get(sec, 0.75)


def detect_text_corruption_patterns(text: str) -> float:
    """
    Step 3 — OCR & Text Corruption Detection.
    Measures character-level corruption and token merging.
    """
    if not text: return 0.0
    corruption = 0.0
    if re.search(r'[a-z][A-Z]', text): corruption += 0.25
    if re.search(r'[0-9][a-zA-Z]', text): corruption += 0.20
    if len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) > 0.2: corruption += 0.35
    return min(1.0, corruption)


def resolve_negation_scope(mention_text: str, context_window: str) -> bool:
    """
    Step 4 — Clinical Negation & Scope Resolution.
    Determines if a concept is negated within its contextual window.
    """
    full = f"{context_window} {mention_text}".lower()
    markers = ["no ", "negative", "denies", "without", "rule out", "absent", "resolved", "not present"]
    return any(m in full for m in markers)


def detect_copy_forward_artifacts(text: str, history: list[str]) -> float:
    """
    Step 5 — Copy-Forward & Template Artifact Suppression.
    Detects unchanged repeated blocks across documentation.
    """
    if not text or not history: return 0.0
    score = 0.0
    for prev in history:
        if text in prev: 
            score += 0.70
            break
    return min(1.0, score)


def stabilize_clinical_entity_boundaries(entity: str) -> str:
    """
    Step 6 — Clinical Entity Boundary Stabilization.
    Normalizes fragmented or merged terminology spans.
    """
    if not entity: return entity
    norm = entity.lower().strip()
    norm = re.sub(r'([a-z])([A-Z])', r'\1 \2', norm)
    norm = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', norm)
    return norm


def compute_reportability_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Reportability Scoring (Task 10E).
    Measures if a condition meets professional coding reportability standards.
    Factors: Active Management, Encounter Relevance, Clinical Significance.
    """
    score = 0.40 # Baseline
    
    # 1. Active Management (MEAT criteria)
    mgmt = float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0)
    score += mgmt * 0.40
    
    # 2. Encounter Relevance
    relevance = float(code_dict.get("encounter_relevance") or 0.5)
    score += (relevance - 0.5) * 0.30
    
    # 3. Independent Significance
    ind_mgmt = compute_independent_management_strength(code_dict)
    score += ind_mgmt * 0.20
    
    # 4. Billing Relevance Boosts
    if code_dict.get("PRINCIPAL_DIAGNOSIS_CONFIRMED") or code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"):
        score += 0.30
    if code_dict.get("COMPLICATION_HIERARCHY_VAL", 0) > 0.7:
        score += 0.15
        
    # Penalize incidental background
    if relevance < 0.3 and mgmt < 0.3:
        score -= 0.25
        
    return min(1.0, max(0.0, score))


def compute_independent_management_strength(code_dict: dict) -> float:
    """
    Step 3 — Independent Management Detection (Task 10E).
    Determines if a condition required dedicated clinical attention.
    """
    strength = 0.0
    traces = code_dict.get("audit_traces", [])
    
    if "SPECIALIST_CONSULT_DETECTED" in traces: strength += 0.40
    if "DEDICATED_MEDICATION_LINKED" in traces: strength += 0.35
    if "PROCEDURAL_INTERVENTION_LINKED" in traces: strength += 0.30
    if "MONITORING_ESCALATION_DETECTED" in traces: strength += 0.25
    
    # Management Intensity proxy
    mgmt = float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0)
    strength += mgmt * 0.20
    
    return min(1.0, strength)


def compute_clinical_significance_priority(code_dict: dict) -> float:
    """
    Step 6 — Clinical Significance Priority (Task 10E).
    Ranks conditions by impact on encounter burden and billing priority.
    """
    priority = 0.30 # Baseline
    
    if code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED"): priority += 0.60
    if float(code_dict.get("PROCEDURAL_STABILITY_WEIGHT") or 0) > 0.8: priority += 0.40
    if code_dict.get("TEMPORAL_STATE") == "ACUTE_SEVERE": priority += 0.35
    if code_dict.get("COMPLICATION_HIERARCHY_VAL", 0) > 0.7: priority += 0.30
    
    ind_mgmt = compute_independent_management_strength(code_dict)
    priority += ind_mgmt * 0.25
    
    # Downrank incidental tails
    if float(code_dict.get("encounter_relevance") or 0.5) < 0.25:
        priority -= 0.20
        
    return min(1.0, max(0.0, priority))


# Step 3: Manifestation Overlap Suppression Pairs (Task 9B Step 3)
# (Broader Code Prefix, Manifestation Prefix)
# If both exist, the broader code already "includes" the manifestation.
MANIFESTATION_OVERLAP_PAIRS: list[tuple[str, str]] = [
    ("E11.4", "G62"), # Diabetes with neuropathy includes polyneuropathy
    ("E11.4", "G56"), # Diabetes with neuropathy includes mononeuropathy
    ("E11.2", "N18"), # Diabetes with nephropathy includes CKD
    ("E11.3", "H35"), # Diabetes with retinopathy includes retinopathy
    ("M80", "S22"),   # Pathological fracture includes generic fracture (vertebral)
    ("M80", "S32"),   # Pathological fracture includes generic fracture (lumbar)
    ("M80", "S72"),   # Pathological fracture includes generic fracture (hip)
    ("I50.2", "I50.9"), # Systolic HF includes generic HF
    ("I50.3", "I50.9"), # Diastolic HF includes generic HF
    ("I11.0", "I50"),   # Hypertensive Heart Disease with HF includes HF
    ("I12", "N18"),     # Hypertensive CKD includes CKD
    ("I13", "I50"),     # Hypertensive Heart & CKD includes HF
    ("I13", "N18"),     # Hypertensive Heart & CKD includes CKD
]


def is_parent_of(parent_code: str, child_code: str) -> bool:
    """
    Step 3: Return True if parent_code is an ICD-10 prefix ancestor of child_code.

    Strict prefix ancestry only (no cross-family matching):
      is_parent_of("S72", "S72.011A")   → True
      is_parent_of("S72.9", "S72.90XA") → True
      is_parent_of("M81", "M80.012A")   → False (different family)
      is_parent_of("S72", "S73.001A")   → False

    Rules:
      1. parent must be shorter than child
      2. child must start with parent (after normalising dots)
      3. The character after the parent prefix in the child must be
         a dot, digit, or letter (not a different branch)
    """
    p = parent_code.strip().upper()
    c = child_code.strip().upper()
    if p == c or len(p) >= len(c):
        return False
    # Child must start with parent prefix
    if not c.startswith(p):
        return False
    # The character immediately after the parent in the child must indicate hierarchy
    next_char = c[len(p)] if len(p) < len(c) else ""
    return next_char in (".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                         "A", "B", "C", "D", "E", "F", "X")


def apply_specificity_hierarchy(
    codes: list[dict],
    note_text: str = "",
) -> list[dict]:
    """
    Steps 3 + 8: Parent-child suppression + final hierarchy pass.

    For each pair of codes in the list:
      If code A is a strict ICD ancestor of code B:
        → Remove A (keep B — the more specific one)
        → Log SPECIFICITY_WINNER: winner=B removed=A

    Also applies generic code penalty:
      If a code has "unspecified"/"NOS" in description AND a more specific
      code exists in the same 3-char prefix family → mark for removal.

    Returns a NEW list with generic parents removed.
    Safe: CPT codes are always passed through.
    Protected codes (source=deterministic) are immune from removal AS PARENTS,
    but a MORE-SPECIFIC deterministic code still wins over a LESS-SPECIFIC one.
    """
    if len(codes) <= 1:
        return codes

    code_set = {(d.get("code") or "").strip().upper() for d in codes}
    to_remove: set[str] = set()

    # Build specificity scores once
    scores: dict[str, int] = {}
    for d in codes:
        c = (d.get("code") or "").strip().upper()
        scores[c] = clinical_specificity_score(c, d.get("description") or "")

    for i, d_parent in enumerate(codes):
        p_code = (d_parent.get("code") or "").strip().upper()
        if not p_code or p_code in to_remove:
            continue
        p_type = (d_parent.get("type") or "ICD-10").upper()

        # Step 3: Procedure Family Collapse Prevention (Task 8A Step 3 + Task 9A Step 5)
        if p_type == "CPT":
            p_fam = p_code[:3]
            p_survival = compute_procedural_survival_score(d_parent, note_text)
            
            for d_other in codes:
                o_code = (d_other.get("code") or "").strip().upper()
                if o_code == p_code or o_code in to_remove:
                    continue
                if (d_other.get("type") or "").upper() == "CPT":
                    o_fam = o_code[:3]
                    if o_fam == p_fam:
                        o_survival = compute_procedural_survival_score(d_other, note_text)
                        
                        # Task 9A Step 3: Modifier Survival Stabilization
                        has_p_mod = "-" in p_code
                        has_o_mod = "-" in o_code
                        
                        # Same family - highest specificity wins, but protect survival
                        # 1. If 'o' has a modifier and 'p' does not, 'o' wins (provided 'o' is grounded)
                        if has_o_mod and not has_p_mod and o_survival > 0.40:
                            to_remove.add(p_code)
                            logger.info(
                                "PROCEDURE_MODIFIER_PRESERVED: winner=%s removed=%s [Task 9A]",
                                o_code, p_code
                            )
                            break

                        # 2. Traditional specificity win
                        if scores.get(o_code, 0) > scores.get(p_code, 0):
                            if p_survival > 0.85 and o_survival < 0.60:
                                # p is strongly grounded, o is weak/generic - keep p
                                continue
                            
                            to_remove.add(p_code)
                            logger.info(
                                "PROCEDURE_RECONCILIATION_STABILIZED: winner=%s (score=%d) "
                                "removed=%s (score=%d) [CPT family collapse]",
                                o_code, scores.get(o_code, 0),
                                p_code, scores.get(p_code, 0),
                            )
                            break
            continue

        # --- ICD-10 Family & Manifestation Reconciliation (Task 9B) ---
        p_prefix_3 = p_code[:3]
        
        for d_other in codes:
            o_code = (d_other.get("code") or "").strip().upper()
            if o_code == p_code or o_code in to_remove:
                continue
            o_type = (d_other.get("type") or "ICD-10").upper()
            if o_type != "ICD-10":
                continue
                
            # 1. Manifestation Overlap Suppression (Step 3)
            for broad_pfx, manifestation_pfx in MANIFESTATION_OVERLAP_PAIRS:
                if p_code.startswith(broad_pfx) and o_code.startswith(manifestation_pfx):
                    # p is the broader combo, o is the manifestation - remove o
                    # (only if p is sufficiently grounded)
                    if float(d_parent.get("evidence_strength") or 0) > 0.45:
                        to_remove.add(o_code)
                        logger.info(
                            "MANIFESTATION_COLLAPSED: winner=%s (broad) removed=%s (manifestation)",
                            p_code, o_code
                        )

            # 2. Sibling Reconciliation (Step 1)
            # If they share the same 3-char prefix, the more specific one wins
            o_prefix_3 = o_code[:3]
            if p_prefix_3 == o_prefix_3:
                p_spec = scores.get(p_code, 0)
                o_spec = scores.get(o_code, 0)
                if o_spec > p_spec:
                    to_remove.add(p_code)
                    logger.info(
                        "REPRESENTATION_FAMILY_REFINED: winner=%s removed=%s [sibling reconciliation]",
                        o_code, p_code
                    )
                    break

        # 3. Traditional Parent-Child Suppression
        if p_code in to_remove:
            continue

        for d_child in codes:
            c_code = (d_child.get("code") or "").strip().upper()
            if c_code == p_code or c_code in to_remove:
                continue
            c_type = (d_child.get("type") or "ICD-10").upper()
            if c_type == "CPT":
                continue

            if is_parent_of(p_code, c_code):
                # Child is more specific → remove parent
                to_remove.add(p_code)
                logger.info(
                    "SPECIFICITY_WINNER: winner=%s (score=%d) removed=%s (score=%d) "
                    "[parent-child suppression]",
                    c_code, scores.get(c_code, 0),
                    p_code, scores.get(p_code, 0),
                )
                break

        # Step 7: Generic code penalty — same 3-char prefix family
        if p_code not in to_remove:
            p_prefix = p_code.split(".")[0][:3] if "." in p_code else p_code[:3]
            p_is_generic = any(w in (d_parent.get("description") or "").lower()
                               for w in _GENERIC_DESC_WORDS)
            if p_is_generic:
                # Look for a more specific sibling in same prefix family
                for d_sibling in codes:
                    s_code = (d_sibling.get("code") or "").strip().upper()
                    if s_code == p_code:
                        continue
                    s_prefix = s_code.split(".")[0][:3] if "." in s_code else s_code[:3]
                    if s_prefix == p_prefix and scores.get(s_code, 0) > scores.get(p_code, 0):
                        to_remove.add(p_code)
                        logger.info(
                            "SPECIFICITY_WINNER: winner=%s (score=%d) removed=%s (score=%d) "
                            "[generic penalty — same prefix family]",
                            s_code, scores.get(s_code, 0),
                            p_code, scores.get(p_code, 0),
                        )
                        break

    if to_remove:
        logger.info("apply_specificity_hierarchy: suppressed %d generic/parent codes: %s",
                    len(to_remove), to_remove)

    return [d for d in codes if (d.get("code") or "").strip().upper() not in to_remove]


def pathological_fracture_protection(
    codes: list[dict],
    note_text: str,
) -> list[dict]:
    """
    Step 4: Pathological fracture protection.

    CRITICAL: Never replace M80.x (with pathological fracture)
    with M81.x (without fracture) unless fracture evidence is absent.

    Rules:
      1. If note contains pathological fracture signals AND M80.x is in codes:
         → boost M80.x evidence_strength to 1.0
         → log PATHOLOGY_PROTECTED
      2. If M80.x AND M81.x BOTH present:
         → remove M81.x (M80 is more specific — it INCLUDES the fracture)
      3. If ONLY M81.x present but note has pathological fracture signals:
         → do NOT remove M81.x (may be the only osteoporosis code available)
         → but log a warning that M80.x should be preferred

    Pathological fracture signals:
      "pathological fracture", "fragility fracture", "osteoporotic fracture",
      "pathologic fracture", "insufficiency fracture"
    """
    PATHOLOGICAL_SIGNALS = [
        "pathological fracture", "pathologic fracture",
        "fragility fracture", "osteoporotic fracture",
        "insufficiency fracture", "low-trauma fracture",
        "low trauma fracture",
    ]
    text_lower = note_text.lower()
    has_path_fracture = any(sig in text_lower for sig in PATHOLOGICAL_SIGNALS)

    m80_codes = [(i, d) for i, d in enumerate(codes)
                 if (d.get("code") or "").strip().upper().startswith("M80")]
    m81_codes = [(i, d) for i, d in enumerate(codes)
                 if (d.get("code") or "").strip().upper().startswith("M81")]

    to_remove: set[str] = set()

    if has_path_fracture and m80_codes:
        # Boost M80 codes to maximum evidence
        for _, d in m80_codes:
            c = (d.get("code") or "").strip().upper()
            d["evidence_strength"] = 1.0
            d["evidence_reason"]   = "pathological fracture explicitly documented in note"
            logger.info(
                "PATHOLOGY_PROTECTED: code=%s | reason='pathological fracture in note' | "
                "evidence boosted to 1.0",
                c,
            )

        # Remove M81 codes — M80 (with fracture) is more specific
        for _, d in m81_codes:
            c = (d.get("code") or "").strip().upper()
            to_remove.add(c)
            logger.info(
                "SPECIFICITY_WINNER: winner=%s (M80 with fracture) removed=%s (M81 without fracture) "
                "[pathological fracture protection]",
                [d2.get("code") for _, d2 in m80_codes][0] if m80_codes else "M80.x",
                c,
            )

    elif m80_codes and m81_codes:
        # Even without explicit signal: M80 is ALWAYS more specific than M81
        # (M80 = with fracture, M81 = without) → remove M81
        for _, d in m81_codes:
            c = (d.get("code") or "").strip().upper()
            to_remove.add(c)
            logger.info(
                "SPECIFICITY_WINNER: winner=M80.x removed=%s "
                "[M80 (with fracture) always supersedes M81 (without)]", c,
            )

    elif has_path_fracture and not m80_codes and m81_codes:
        logger.warning(
            "PATHOLOGY_PROTECTED: note has pathological fracture signals but only M81.x found. "
            "M80.x should be preferred — check RAG/deterministic code generation."
        )

    if to_remove:
        return [d for d in codes if (d.get("code") or "").strip().upper() not in to_remove]
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION-AWARE SYSTEM  (Steps 1-10 in spec)
# ─────────────────────────────────────────────────────────────────────────────

# Section header patterns → canonical name (ordered: most-specific first)
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("postop_diagnosis",  [
        r"post[\s\-]?op(?:erative)?\s+diag(?:nosis)?",
        r"postoperative\s+diag(?:nosis)?",
        r"post\s*op\s+dx",
    ]),
    ("preop_diagnosis",   [
        r"pre[\s\-]?op(?:erative)?\s+diag(?:nosis)?",
        r"preoperative\s+diag(?:nosis)?",
        r"pre\s*op\s+dx",
        r"admitting\s+diag(?:nosis)?",
    ]),
    ("procedure",         [
        r"procedure(?:\s+performed)?",
        r"operation\s+performed",
        r"operative\s+procedure",
        r"surgery\s+performed",
        r"operation:",
    ]),
    ("assessment",        [
        r"assessment(?:\s+and\s+plan)?",
        r"clinical\s+impression",
        r"final\s+diagnosis",
        r"discharge\s+diagnosis",
    ]),
    ("impression",        [
        r"impression",
        r"radiologic(?:al)?\s+impression",
    ]),
    ("findings",          [
        r"intraoperative\s+findings",
        r"operative\s+findings",
        r"findings",
    ]),
    ("plan",              [
        r"plan(?:\s+of\s+care)?",
        r"treatment\s+plan",
    ]),
    ("medications",       [
        r"medications?(?:\s+list)?",
        r"current\s+medications?",
        r"home\s+medications?",
        r"medication\s+reconciliation",
    ]),
    ("pmh",               [
        r"past\s+(?:medical\s+)?history",
        r"(?:past\s+)?medical\s+history",
        r"\bpmh\b",
        r"prior\s+medical\s+history",
    ]),
    ("family_history",    [
        r"family\s+history",
        r"\bfh\b:",
    ]),
    ("history",           [
        r"history\s+(?:of\s+)?(?:present(?:ing)?\s+)?illness",
        r"\bhpi\b",
        r"chief\s+complaint",
        r"social\s+history",
    ]),
]

# Section classification logic (Step 1)


def parse_note_sections(note_text: str) -> dict[str, str]:
    """
    Step 1: Extract named sections from a clinical / operative note.

    Returns a dict: canonical_name → section body text.
    Always includes 'full_note' key with the entire note text.

    Algorithm:
      1. Scan for header patterns (case-insensitive).
      2. Each section body runs until the next detected header or EOF.
      3. First matching canonical name per position wins.
      4. Duplicate section headers are concatenated.
    """
    sections: dict[str, str] = {"full_note": note_text}
    text_lower = note_text.lower()

    # Build (start_pos, end_of_header_pos, canonical_name) for every match
    header_positions: list[tuple[int, int, str]] = []
    for canonical, patterns in _SECTION_PATTERNS:
        for pat in patterns:
            full_pat = rf"(?:^|\n)[ \t]*(?:{pat})[ \t]*:?"
            for m in re.finditer(full_pat, text_lower, re.IGNORECASE | re.MULTILINE):
                header_positions.append((m.start(), m.end(), canonical))

    if not header_positions:
        return sections

    # Sort by position; deduplicate overlapping matches
    header_positions.sort(key=lambda x: x[0])
    deduped: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, name in header_positions:
        if start >= last_end:
            deduped.append((start, end, name))
            last_end = end

    # Slice section bodies between consecutive headers
    for i, (start, end, name) in enumerate(deduped):
        next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(note_text)
        body = note_text[end:next_start].strip()
        if name in sections:
            sections[name] = sections[name] + " " + body
        else:
            sections[name] = body

    return sections


def get_term_section_context(
    term: str,
    sections: dict[str, str],
) -> dict[str, bool]:
    """
    Step 2: For a clinical term, return which sections it appears in.

    Returns {section_name: True} for every section containing the term.
    'full_note' is excluded from results.
    """
    term_lower = term.lower()
    return {
        name: True
        for name, body in sections.items()
        if name != "full_note" and term_lower in body.lower()
    }


def compute_section_aware_boost(
    term: str,
    description: str,
    sections: dict[str, str],
    code_type: str = "ICD-10",
) -> tuple[float, str, list[str]]:
    """
    Steps 2-9: Compute a section-aware evidence boost for a code / term.

    Returns:
      (boost_delta: float, dominant_section: str, matched_sections: list[str])

    boost_delta is ADDED to evidence_strength (can be negative for penalties).

    Boost rules:
      postop_diagnosis / procedure   → +0.30
      preop_diagnosis / assessment   → +0.20
      impression / findings          → +0.10
      plan                           → +0.05
      ONLY in pmh/medications/family → −0.30  (Steps 3, 6, 8)
      2+ high-priority sections      → +0.10 extra  (Step 7)
      CPT in procedure section       → boost ≥ 0.30  (Step 4)
    """
    stop = {"unspecified", "other", "type", "nos", "due", "with", "without",
            "acute", "chronic", "bilateral", "right", "left", "initial",
            "subsequent", "encounter", "specified", "site", "and"}
    desc_words = [
        w for w in re.sub(r"[^a-z\s]", "", description.lower()).split()
        if len(w) > 4 and w not in stop
    ][:3]

    # Collect all candidate terms (original + key description words)
    check_terms: set[str] = set()
    if term:
        check_terms.add(term.lower())
    check_terms.update(desc_words)

    # Union of all sections any candidate term appears in
    matched_sections: set[str] = set()
    for t in check_terms:
        matched_sections.update(get_term_section_context(t, sections).keys())

    if not matched_sections:
        return 0.0, "full_note", []

    dominant = max(
        matched_sections,
        key=lambda s: SECTION_WEIGHTS.get(s, 0.0),
        default="full_note",
    )
    dominant_weight = SECTION_WEIGHTS.get(dominant, 0.0)

    high_matches = matched_sections & HIGH_PRIORITY_SECTIONS
    low_only = matched_sections.issubset(LOW_PRIORITY_SECTIONS)

    # Steps 3, 6, 8: history / medication / family-only → heavy penalty
    if low_only:
        boost = -0.30
        logger.debug(
            "SECTION_PRIORITY_BOOST: term='%s' only_in_low_sections=%s boost=%.2f",
            term, sorted(matched_sections), boost,
        )
        return boost, dominant, sorted(matched_sections)

    # Boost based on dominant section weight
    if dominant_weight >= 0.95:
        boost = 0.30
    elif dominant_weight >= 0.85:
        boost = 0.20
    elif dominant_weight >= 0.70:
        boost = 0.10
    elif dominant_weight >= 0.60:
        boost = 0.05
    else:
        boost = 0.0

    # Step 4: CPT extra boost when in procedure section
    if code_type == "CPT" and "procedure" in matched_sections:
        boost = max(boost, 0.30)
        logger.info(
            "SECTION_PRIORITY_BOOST: code_type=CPT section=procedure boost=%.2f", boost
        )

    # Step 7: Multi-section boost
    if len(high_matches) >= 2:
        boost += 0.10

    logger.debug(
        "SECTION_MATCH: section=%s term='%s' matched_sections=%s boost=%.2f",
        dominant, term, sorted(matched_sections), boost,
    )
    return boost, dominant, sorted(matched_sections)


# ─────────────────────────────────────────────────────────────────────────────
# RELATIONSHIP-AWARE CLINICAL REASONING  (Steps 1-10 in spec)
# ─────────────────────────────────────────────────────────────────────────────

# Step 1 — Clinical modifiers that must stay bound to the diagnosis they modify.
# Key = modifier word; value = boost weight when it matches the code description.
CLINICAL_MODIFIERS: dict[str, float] = {
    # Laterality
    "left":              0.12,
    "right":             0.12,
    "bilateral":         0.10,
    # Displacement / severity
    "displaced":         0.15,
    "nondisplaced":      0.15,
    "comminuted":        0.12,
    "open":              0.12,
    "closed":            0.08,
    "complete":          0.10,
    "incomplete":        0.10,
    # Temporal acuity
    "acute":             0.10,
    "chronic":           0.08,
    "subacute":          0.08,
    # Severity
    "severe":            0.10,
    "moderate":          0.08,
    "mild":              0.06,
    # Fracture-specific
    "femoral":           0.12,
    "femoral neck":      0.15,
    "intertrochanteric": 0.15,
    "subtrochanteric":   0.12,
    "pathological":      0.18,
    "fragility":         0.15,
    # Procedure-context
    "total":             0.10,
    "partial":           0.08,
    "primary":           0.08,
    "revision":          0.10,
}

# Step 2 — Causality linkage phrases (must appear BETWEEN two clinical concepts).
CAUSALITY_PHRASES: tuple[str, ...] = (
    "due to",
    "secondary to",
    "caused by",
    "resulting from",
    "as a result of",
    "associated with",
    "in the setting of",
    "in the context of",
    "related to",
    "complicating",
    "complicated by",
    "from",          # narrow: only counts when between two clinical terms
    "because of",
)

# Step 5 — Strong (explicit) causality phrases vs weak co-occurrence markers (Task 8B Step 4-5)
# Only STRONG phrases should qualify for causality upgrades on combination codes.
STRONG_CAUSALITY_PHRASES: frozenset[str] = frozenset([
    "due to",
    "secondary to",
    "caused by",
    "as a result of",
    "resulting from",
    "in the setting of",   # physician explicitly contextualizes the relationship
    "complicated by",
    "because of",
])

# Weak co-occurrence markers — presence of these alone should NOT create causal upgrades
WEAK_COOCCURRENCE_MARKERS: frozenset[str] = frozenset([
    "associated with",
    "in the context of",
    "related to",
    "from",
    "with",   # simple co-presence
])

# Respiratory workflow context indicators (Task 8B Step 2)
RESPIRATORY_CONTEXT_INDICATORS: dict[str, list[str]] = {
    "hypoxia": ["spo2", "oxygen saturation", "o2 sat", "hypoxia", "hypoxemic", "pao2"],
    "support": ["bipap", "cpap", "mechanical ventilation", "ventilator", "high flow", "non-invasive", "supplemental oxygen"],
    "exacerbation": ["exacerbation", "acute", "worsening", "decompensated", "flare"],
    "abg": ["abg", "arterial blood gas", "pco2", "pao2", "ph 7.", "respiratory acidosis"],
    "effusion": ["thoracentesis", "pleural tap", "transudate", "exudate", "light criteria", "ldh", "protein"],
}

# Step 4 — Symptom terms: presence of ONLY symptom codes should not override confirmed diagnoses.
# These map a symptom description keyword to typical ICD symptom prefixes.
SYMPTOM_ICD_PREFIXES: dict[str, list[str]] = {
    "pain":       ["M54", "M79", "G89", "R52"],
    "swelling":   ["R60", "M79"],
    "fever":      ["R50"],
    "nausea":     ["R11"],
    "fatigue":    ["R53"],
    "dyspnea":    ["R06"],
    "edema":      ["R60"],
    "dizziness":  ["R42"],
    "syncope":    ["R55"],
    "weakness":   ["M62", "R53"],
    "headache":   ["R51", "G44"],
    "cough":      ["R05"],
    "chest pain": ["R07"],
    "back pain":  ["M54"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generalized Symptom Categories
# ─────────────────────────────────────────────────────────────────────────────
SYMPTOM_CATEGORIES = {
    "systemic_symptoms": {
        "terms": ["fever", "chills", "rigors", "malaise", "fatigue", "diaphoresis", "night sweats"],
        "icd_prefixes": ["R50", "R53", "R61"]
    },
    "pain_symptoms": {
        "terms": ["pain", "tenderness", "discomfort", "ache", "soreness", "cramping"],
        "icd_prefixes": ["M54", "M79", "G89", "R52", "R07", "R10"]
    },
    "respiratory_symptoms": {
        "terms": ["dyspnea", "shortness of breath", "sob", "wheezing", "cough", "tachypnea"],
        "icd_prefixes": ["R05", "R06"]
    },
    "GI_symptoms": {
        "terms": ["nausea", "vomiting", "diarrhea", "constipation", "bloating", "dyspepsia", "anorexia"],
        "icd_prefixes": ["R11", "R19", "K59", "R63"]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generalized Explanatory Disease Groups
# ─────────────────────────────────────────────────────────────────────────────
EXPLANATORY_DISEASE_FAMILIES = {
    "infection_systemic": {
        "terms": ["sepsis", "bacteremia", "pyelonephritis", "pneumonia", "sirs", "meningitis", "abscess"],
        "icd_prefixes": ["A41", "A40", "N10", "J18", "J15"],
        "explains": ["systemic_symptoms", "pain_symptoms"]
    },
    "fracture_trauma": {
        "terms": ["fracture", "dislocation", "trauma", "injury", "broken", "laceration", "contusion"],
        "icd_prefixes": ["S72", "S52", "S82", "S42", "S22", "M84", "M80"],
        "explains": ["pain_symptoms"]
    },
    "cardiopulmonary": {
        "terms": ["chf", "congestive heart failure", "pulmonary edema", "respiratory failure", "copd", "asthma", "myocardial infarction", "mi"],
        "icd_prefixes": ["I50", "J81", "J96", "J44", "J45", "I21"],
        "explains": ["respiratory_symptoms", "pain_symptoms"]
    },
    "GI_inflammatory": {
        "terms": ["gastroenteritis", "colitis", "appendicitis", "cholecystitis", "pancreatitis", "gastritis", "diverticulitis"],
        "icd_prefixes": ["A09", "K52", "K35", "K81", "K85", "K29", "K57"],
        "explains": ["GI_symptoms", "pain_symptoms"]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generalized Complication Families
# ─────────────────────────────────────────────────────────────────────────────
COMPLICATION_FAMILIES = {
    "diabetes_complications": {
        "base_terms": ["diabetes", "dm2", "dm1", "t2dm", "t1dm"],
        "complications": ["nephropathy", "neuropathy", "retinopathy", "angiopathy", "ckd", "foot ulcer", "ketoacidosis", "hyperglycemia", "ophthalmic"],
        "icd_prefixes": ["E08", "E09", "E10", "E11", "E13"]
    },
    "hypertension_complications": {
        "base_terms": ["hypertension", "htn", "high blood pressure"],
        "complications": ["ckd", "heart disease", "renal failure", "nephropathy"],
        "icd_prefixes": ["I11", "I12", "I13", "I15"]
    },
    "fracture_complications": {
        "base_terms": ["fracture", "fx"],
        "complications": ["pathological", "pathologic", "displaced", "open", "comminuted", "fragility", "nonunion", "malunion"],
        "icd_prefixes": ["S", "M80", "M84"]
    },
    "heart_failure_complications": {
        "base_terms": ["heart failure", "chf"],
        "complications": ["pulmonary edema", "respiratory failure", "acute on chronic", "systolic", "diastolic"],
        "icd_prefixes": ["I50"]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Clinical Linkage Patterns
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_LINKAGE_PHRASES = [
    "with",
    "due to",
    "associated with",
    "secondary to",
    "complicated by",
    "resulting in",
    "manifested by",
    "evidence of",
]
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generalized Organism Groups
# ─────────────────────────────────────────────────────────────────────────────
ORGANISM_GROUPS = {
    "gram_negative": ["e. coli", "klebsiella", "pseudomonas", "escherichia", "proteus", "enterobacter", "serratia", "acinetobacter", "neisseria"],
    "gram_positive": ["staph", "strep", "enterococcus", "mrsa", "mssa", "pneumococcus", "listeria", "clostridium"],
    "fungal": ["candida", "aspergillus", "cryptococcus", "histoplasma", "mucor", "tinea"],
    "viral": ["influenza", "rsv", "covid", "herpes", "hsv", "hiv", "hepatitis", "ebv", "cmv"],
    "atypical": ["mycoplasma", "chlamydia", "legionella"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Etiology Linkage Phrases
# ─────────────────────────────────────────────────────────────────────────────
ETIOLOGY_LINKAGE_PHRASES = [
    "caused by",
    "due to",
    "secondary to",
    "culture positive for",
    "cultures positive for",
    "bacteremia with",
    "urine culture grew",
    "blood cultures positive for",
    "grew out",
    "organism identified:",
    "consistent with infection by",
    "attributed to",
    "infection with",
]

# Step 5 — Organism temporal/uncertainty markers
ORGANISM_UNCERTAINTY_TOKENS = [
    "possible contamination",
    "suspected organism",
    "pending cultures",
    "rule out bacteremia",
    "potential contaminant",
    "preliminary results",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generalized Temporal Status Categories
# ─────────────────────────────────────────────────────────────────────────────
TEMPORAL_STATUS_CATEGORIES = {
    "active_confirmed": ["acute", "active", "confirmed", "current", "ongoing", "newly diagnosed", "exacerbation"],
    "historical": ["history of", "prior", "previous", "remote", "past medical history", "pmh", "status post", "s/p"],
    "resolved": ["resolved", "improved", "completed", "recovered", "cleared", "no longer present"],
    "ruled_out": ["rule out", "unlikely", "excluded", "negative for", "ruled out", "not present", "r/o"],
    "family_history": ["family history", "mother had", "father had", "sibling had", "brother had", "sister had", "fhx"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Chronic Active Disease Markers
# ─────────────────────────────────────────────────────────────────────────────
CHRONIC_ACTIVE_MARKERS = [
    "stage", "chronic", "persistent", "maintenance", "on medication for",
    "monitored for", "under treatment", "stable on", "daily",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2 — Generalized Procedure ↔ Diagnosis Coherence
# ─────────────────────────────────────────────────────────────────────────────
PROCEDURE_COHERENCE_FAMILIES = {
    "orthopedic_procedures": {
        "proc_terms": ["arthroplasty", "orif", "fixation", "laminectomy", "replacement", "spinal fusion"],
        "diag_terms": ["fracture", "osteoarthritis", "degenerative", "stenosis", "spondylosis", "broken", "injury"],
    },
    "vascular_access": {
        "proc_terms": ["central line", "dialysis catheter", "picc line", "venous catheter", "port-a-cath"],
        "diag_terms": ["sepsis", "dialysis", "renal failure", "shock", "poor access", "infusion", "dehydration"],
    },
    "cardiac_procedures": {
        "proc_terms": ["cabg", "pci", "stent", "bypass", "angioplasty", "valvuloplasty", "pacemaker"],
        "diag_terms": ["cad", "mi", "ischemia", "infarction", "coronary", "stenosis", "heart failure"],
    },
    "GI_procedures": {
        "proc_terms": ["appendectomy", "cholecystectomy", "colectomy", "endoscopy", "colonoscopy", "resection"],
        "diag_terms": ["appendicitis", "cholecystitis", "colitis", "carcinoma", "polyp", "diverticulitis", "obstruction"],
    },
    "neuro_intervention": {
        "proc_terms": ["thrombectomy", "mechanical thrombectomy", "coiling", "embolization", "evt"],
        "diag_terms": ["stroke", "infarction", "cva", "occlusion", "aneurysm", "hemorrhage", "thrombus"],
    },
    "diagnostic_imaging": {
        "proc_terms": ["ct head", "cta head", "mri brain", "ct chest", "cta chest", "echocardiogram", "ultrasound"],
        "diag_terms": ["stroke", "trauma", "headache", "pulmonary embolism", "heart failure", "ischemia", "evaluation"],
    }
}

# Step 6 — Temporal / uncertainty hedges that reduce evidence confidence.
TEMPORAL_HEDGES: dict[str, float] = {
    "possible":       -0.20,
    "probable":       -0.15,
    "suspected":      -0.20,
    "concern for":    -0.20,
    "rule out":       -0.25,
    "r/o":            -0.25,
    "to rule out":    -0.25,
    "query":          -0.20,
    "questionable":   -0.20,
    "cannot exclude": -0.15,
    "may represent":  -0.15,
    "versus":         -0.10,
    "vs.":            -0.10,
    "differential":   -0.10,
}

# Phrases that CONFIRM a diagnosis (override temporal hedges)
CONFIRMATION_PHRASES: tuple[str, ...] = (
    "confirmed",
    "diagnosed with",
    "diagnosis of",
    "known",
    "established",
    "documented",
    "positive for",
    "consistent with",    # radiology language for confirmed
    "pathology confirmed",
    "biopsy proven",
)

# Step 5 — Procedure → diagnosis reinforcement table.
# When a procedure phrase appears near a diagnosis concept, it strongly validates
# both the procedure code AND the diagnosis code.
PROCEDURE_DIAGNOSIS_LINKS: list[tuple[list[str], list[str]]] = [
    # Procedure phrases → reinforced diagnosis terms
    (["total hip arthroplasty", "hip replacement", "tha", "thr"],
     ["femoral", "hip", "fracture", "osteoarthritis", "avascular necrosis"]),
    (["total knee arthroplasty", "knee replacement", "tka", "tkr"],
     ["knee", "tibial", "osteoarthritis", "fracture"]),
    (["orif", "open reduction internal fixation", "intramedullary nail", "im nail"],
     ["fracture", "femoral", "tibial", "humeral", "radius"]),
    (["spinal fusion", "lumbar fusion", "tlif", "plif", "acdf"],
     ["vertebral", "lumbar", "cervical", "disc", "stenosis", "spondylosis"]),
    (["appendectomy"],
     ["appendicitis"]),
    (["cholecystectomy"],
     ["cholelithiasis", "cholecystitis", "gallstone"]),
    (["colectomy", "bowel resection"],
     ["colitis", "carcinoma", "diverticulitis", "obstruction"]),
    (["coronary bypass", "cabg"],
     ["coronary", "ischemia", "infarction", "atherosclerosis"]),
    (["valve replacement", "valvuloplasty"],
     ["stenosis", "regurgitation", "valve"]),
    (["amputation"],
     ["gangrene", "diabetes", "peripheral vascular", "ischemia"]),
    (["mastectomy"],
     ["carcinoma", "breast", "malignancy"]),
]

# Step 8 — Clinically incompatible code pairs (ICD prefix level).
# If both appear simultaneously, prefer the one higher in priority order.
INCOMPATIBLE_CODE_PAIRS: list[tuple[str, str, str]] = [
    # (prefix_a, prefix_b, prefer)  — prefer = "a" or "b" or "specificity"
    ("M80", "M81", "a"),   # M80 (with fracture) supersedes M81 (without)
    ("I21", "I25", "a"),   # Acute MI supersedes chronic ischemic
    ("N17", "N18", "a"),   # AKI supersedes CKD (if both present, AKI is acute)
    ("J18", "J22", "a"),   # Lobar pneumonia supersedes unspecified lower resp
    ("A41", "R65", "a"),   # Sepsis supersedes SIRS
    ("I50", "R00", "a"),   # Heart failure supersedes palpitation symptom
    ("S72", "M84", "specificity"),  # Femur fracture vs pathological — prefer specificity
]


def bind_modifiers_to_diagnosis(
    description: str,
    note_text: str,
    window: int = 120,
) -> tuple[float, list[str]]:
    """
    Step 1: Bind clinical modifiers to a diagnosis and score the binding strength.

    Algorithm:
      1. Extract key modifier words from the ICD code description.
      2. Find each modifier in the note text.
      3. Check if the modifier appears in a window around the diagnosis term.
      4. Each bound modifier contributes its weight to the total boost.

    Returns:
      (boost: float, bound_modifiers: list[str])
      boost is added to evidence_strength if > 0.
    """
    desc_lower = description.lower()
    text_lower = note_text.lower()

    # Extract which modifiers from CLINICAL_MODIFIERS appear in the ICD description
    desc_modifiers = {
        mod: weight
        for mod, weight in CLINICAL_MODIFIERS.items()
        if mod in desc_lower
    }
    if not desc_modifiers:
        return 0.0, []

    # Extract key diagnosis noun from description (first non-modifier content word)
    stop = {"unspecified", "other", "type", "nos", "due", "with", "without",
            "initial", "subsequent", "encounter", "sequela", "specified", "and"}
    mod_words = set(CLINICAL_MODIFIERS.keys())
    desc_words = [
        w for w in re.sub(r"[^a-z\s]", "", desc_lower).split()
        if len(w) > 3 and w not in stop and w not in mod_words
    ]
    if not desc_words:
        return 0.0, []

    # Diagnosis anchor: the first non-modifier meaningful word
    anchor = desc_words[0]
    anchor_idx = text_lower.find(anchor)
    if anchor_idx == -1:
        return 0.0, []

    # For each description modifier, check it appears near the anchor in the note
    bound: list[str] = []
    total_boost = 0.0
    for mod, weight in desc_modifiers.items():
        mod_idx = text_lower.find(mod)
        while mod_idx != -1:
            if abs(mod_idx - anchor_idx) <= window:
                bound.append(mod)
                total_boost += weight
                logger.debug(
                    "RELATIONSHIP_BOUND: diagnosis='%s' modifier='%s' confidence=%.2f",
                    anchor, mod, weight,
                )
                break
            mod_idx = text_lower.find(mod, mod_idx + 1)

    return min(total_boost, 0.35), bound  # cap at 0.35 to prevent runaway boost


def detect_causality(
    term_a: str,
    term_b: str,
    note_text: str,
    window: int = 200,
) -> tuple[bool, str]:
    """
    Step 2: Detect whether a causal relationship exists between two clinical terms.

    Algorithm:
      1. Find term_a in note_text.
      2. In the window around term_a, look for CAUSALITY_PHRASES.
      3. After the causality phrase, look for term_b within the remaining window.

    Returns:
      (causality_found: bool, matched_phrase: str)

    Example:
      "heart failure due to hypertension"
      detect_causality("heart failure", "hypertension", note)
      → (True, "due to")
    """
    text_lower = note_text.lower()
    a_lower = term_a.lower()
    b_lower = term_b.lower()

    a_idx = text_lower.find(a_lower)
    while a_idx != -1:
        context = text_lower[a_idx: min(len(text_lower), a_idx + window)]
        for phrase in CAUSALITY_PHRASES:
            phrase_idx = context.find(phrase)
            if phrase_idx == -1:
                continue
            # term_b must appear after the causality phrase in the same window
            after_phrase = context[phrase_idx + len(phrase):]
            if b_lower in after_phrase:
                logger.info(
                    "CAUSALITY_DETECTED: source='%s' target='%s' phrase='%s'",
                    term_a, term_b, phrase,
                )
                return True, phrase
        a_idx = text_lower.find(a_lower, a_idx + 1)

    return False, ""


def classify_diagnosis_type(
    description: str,
    note_text: str,
    code: str = "",
) -> str:
    """
    Step 4: Classify a code as 'confirmed', 'symptom', or 'hedged'.

    Rules:
      1. 'symptom': code prefix matches SYMPTOM_ICD_PREFIXES AND no explicit
         diagnosis language around the description term in the note.
      2. 'hedged': description term appears ONLY in a temporal hedge context.
      3. 'confirmed': explicit diagnosis token OR no hedge present.

    Returns one of: 'confirmed', 'symptom', 'hedged'
    """
    code_upper = code.strip().upper()
    text_lower = note_text.lower()
    desc_lower = description.lower()

    # Check if this is a pure symptom code
    for sym_term, prefixes in SYMPTOM_ICD_PREFIXES.items():
        if any(code_upper.startswith(pfx) for pfx in prefixes):
            if sym_term in desc_lower:
                # Confirmed if explicit diagnosis language in note for this symptom
                # (e.g. "confirmed hip pain" is still a symptom but explicitly documented)
                has_confirmation = any(phrase in text_lower for phrase in CONFIRMATION_PHRASES)
                if not has_confirmation:
                    return "symptom"

    # Check temporal hedge context for key description terms
    stop = {"unspecified", "other", "type", "nos", "with", "without",
            "initial", "subsequent", "encounter", "and", "the", "of"}
    desc_words = [
        w for w in re.sub(r"[^a-z\s]", "", desc_lower).split()
        if len(w) > 4 and w not in stop
    ][:2]

    hedged_count = 0
    confirmed_count = 0
    for word in desc_words:
        idx = text_lower.find(word)
        while idx != -1:
            ctx_start = max(0, idx - 80)
            ctx_end   = min(len(text_lower), idx + 80)
            ctx = text_lower[ctx_start:ctx_end]
            if any(h in ctx for h in TEMPORAL_HEDGES):
                hedged_count += 1
            if any(c in ctx for c in CONFIRMATION_PHRASES):
                confirmed_count += 1
            idx = text_lower.find(word, idx + 1)

    if confirmed_count > 0:
        return "confirmed"
    if hedged_count > 0 and confirmed_count == 0:
        return "hedged"
    return "confirmed"


def link_procedure_to_diagnosis(
    proc_description: str,
    diag_description: str,
    note_text: str,
) -> tuple[bool, float]:
    """
    Step 5: Determine whether a procedure and diagnosis are clinically linked
    based on the PROCEDURE_DIAGNOSIS_LINKS table and note text proximity.

    Returns:
      (is_linked: bool, boost: float)
      boost: added to the DIAGNOSIS code's evidence_strength when linked.

    Example:
      "total hip arthroplasty" ↔ "femoral neck fracture" → linked, boost=0.20
    """
    proc_lower = proc_description.lower()
    diag_lower = diag_description.lower()
    text_lower = note_text.lower()

    for proc_phrases, diag_terms in PROCEDURE_DIAGNOSIS_LINKS:
        # Check if any procedure phrase matches the procedure description/note
        proc_match = any(pp in proc_lower or pp in text_lower for pp in proc_phrases)
        if not proc_match:
            continue

        # Check if any diagnosis term matches the diagnosis description
        diag_match = any(dt in diag_lower for dt in diag_terms)
        if diag_match:
            # Verify both appear in the note in proximity
            for pp in proc_phrases:
                p_idx = text_lower.find(pp)
                if p_idx == -1:
                    continue
                for dt in diag_terms:
                    d_idx = text_lower.find(dt)
                    if d_idx == -1:
                        continue
                    if abs(p_idx - d_idx) <= 300:
                        logger.info(
                            "PROCEDURE_LINKED: procedure='%s' diagnosis='%s'",
                            pp, dt,
                        )
                        return True, 0.20

    return False, 0.0


def get_temporal_hedge_penalty(
    description: str,
    note_text: str,
    window: int = 100,
) -> tuple[float, str]:
    """
    Step 6: Compute confidence penalty for temporal / uncertainty hedges.

    If a hedge word appears near a key description term in the note,
    return a negative delta to subtract from evidence_strength.

    If a CONFIRMATION_PHRASE also appears in the same window, the hedge
    is overridden and no penalty is applied.

    Returns:
      (penalty: float, matched_hedge: str)
      penalty is negative (e.g. -0.20) or 0.0 if no hedge found.
    """
    text_lower = note_text.lower()
    desc_lower = description.lower()

    stop = {"unspecified", "other", "type", "nos", "with", "without",
            "initial", "subsequent", "encounter", "and", "the", "of"}
    desc_words = [
        w for w in re.sub(r"[^a-z\s]", "", desc_lower).split()
        if len(w) > 4 and w not in stop
    ][:2]

    worst_penalty = 0.0
    worst_hedge   = ""

    for word in desc_words:
        idx = text_lower.find(word)
        while idx != -1:
            ctx_start = max(0, idx - window)
            ctx_end   = min(len(text_lower), idx + len(word) + window)
            ctx = text_lower[ctx_start:ctx_end]

            # Check for confirmation override first
            if any(c in ctx for c in CONFIRMATION_PHRASES):
                idx = text_lower.find(word, idx + 1)
                continue

            for hedge, penalty in TEMPORAL_HEDGES.items():
                if hedge in ctx and penalty < worst_penalty:
                    worst_penalty = penalty
                    worst_hedge   = hedge

            idx = text_lower.find(word, idx + 1)

    return worst_penalty, worst_hedge


def validate_code_relationships(
    code: str,
    description: str,
    code_type: str,
    note_text: str,
    all_codes: list[dict],
    note_sections: dict[str, str] | None = None,
) -> tuple[float, str]:
    """
    Steps 7-9: Master relationship validation pass for a single code.

    Runs all relationship checks and returns a net boost/penalty delta.

    Checks applied (in order):
      1. Modifier binding boost (Step 1)
      2. Temporal hedge penalty (Step 6)
      3. Diagnosis type classification (Step 4)
         - symptom code → penalty if confirmed diagnosis for same anatomy exists
         - hedged code → apply penalty
      4. Procedure↔diagnosis link boost (Step 5, for ICD codes only)
      5. Causality check for combination code candidates (Step 2-3)

    Returns:
      (net_delta: float, reason: str)
      net_delta is added to evidence_strength (clamped externally).
    """
    delta = 0.0
    reasons: list[str] = []

    # --- Step 1: Modifier binding ---
    mod_boost, bound_mods = bind_modifiers_to_diagnosis(description, note_text)
    if mod_boost > 0:
        delta += mod_boost
        reasons.append(f"modifier_binding={mod_boost:.2f} ({','.join(bound_mods[:3])})")

    # --- Step 6: Temporal hedge penalty ---
    hedge_penalty, hedge_word = get_temporal_hedge_penalty(description, note_text)
    if hedge_penalty < 0:
        delta += hedge_penalty
        reasons.append(f"temporal_hedge='{hedge_word}' {hedge_penalty:.2f}")

    # --- Step 4: Symptom vs. confirmed diagnosis ---
    diag_type = classify_diagnosis_type(description, note_text, code)
    if diag_type == "symptom":
        # Penalise symptom codes when a more specific confirmed diagnosis exists
        code_prefix = code.strip().upper()[:3]
        has_confirmed_sibling = any(
            c.get("code", "")[:3] != code_prefix and
            classify_diagnosis_type(
                c.get("description", ""), note_text, c.get("code", "")
            ) == "confirmed"
            for c in all_codes
            if (c.get("code") or "") != code
        )
        if has_confirmed_sibling:
            delta -= 0.20
            reasons.append("symptom_code_with_confirmed_sibling=-0.20")
    elif diag_type == "hedged" and hedge_penalty == 0.0:
        # Additional penalty for hedged diagnoses (hedge not caught above)
        delta -= 0.15
        reasons.append("hedged_diagnosis=-0.15")

    # --- Step 5: Procedure↔diagnosis link (ICD codes only) ---
    if code_type == "ICD-10":
        for other in all_codes:
            if (other.get("type") or "ICD-10").upper() != "CPT":
                continue
            proc_desc = other.get("description") or ""
            linked, link_boost = link_procedure_to_diagnosis(
                proc_desc, description, note_text
            )
            if linked:
                delta += link_boost
                reasons.append(f"procedure_linked={link_boost:.2f} via '{proc_desc[:40]}'")
                break  # one link is enough

    # --- Steps 2-3: Causality for combination codes (Task 8B tightened) ---
    # Only upgrade if STRONG explicit causality is found; co-occurrence is insufficient.
    if " with " in description.lower() and code_type == "ICD-10":
        parts = description.lower().split(" with ", 1)
        if len(parts) == 2:
            term_a, term_b = parts[0].strip(), parts[1].strip()
            # Trim to key words
            term_a = term_a.split()[-1] if term_a.split() else term_a
            term_b = term_b.split()[0]  if term_b.split() else term_b

            # Step 4: Check for STRONG causality phrases only
            text_lower_r = note_text.lower()
            strong_causal = False
            matched_phrase = ""
            for phrase in STRONG_CAUSALITY_PHRASES:
                a_idx = text_lower_r.find(term_a)
                while a_idx != -1:
                    context = text_lower_r[a_idx: min(len(text_lower_r), a_idx + 200)]
                    p_idx = context.find(phrase)
                    if p_idx != -1 and term_b in context[p_idx + len(phrase):]:
                        strong_causal = True
                        matched_phrase = phrase
                        break
                    a_idx = text_lower_r.find(term_a, a_idx + 1)
                if strong_causal:
                    break

            if strong_causal:
                delta += 0.15
                reasons.append(f"causality_validated=+0.15 ('{matched_phrase}')")
            else:
                # Step 5: Check for weak co-occurrence (just both terms present)
                both_present = term_a in text_lower_r and term_b in text_lower_r
                if both_present:
                    # Co-occurrence only — no boost, small penalty for over-claiming
                    delta -= 0.05
                    reasons.append("CO_OCCURRENCE_NOT_CAUSAL=-0.05: terms co-occur but no explicit causality")
                    logger.debug("CO_OCCURRENCE_NOT_CAUSAL: code=%s terms=(%s, %s)", code, term_a, term_b)
                else:
                    delta -= 0.10
                    reasons.append("combination_code_unvalidated=-0.10")

    reason_str = "; ".join(reasons) if reasons else "no_relationship_signals"
    return delta, reason_str


def check_cross_diagnosis_conflicts(codes: list[dict]) -> list[dict]:
    """
    Step 8: Remove clinically incompatible code pairs.

    For each pair in INCOMPATIBLE_CODE_PAIRS:
      - If both are present in codes, remove the lower-priority one.
      - 'prefer=a': keep prefix_a, remove prefix_b
      - 'prefer=b': keep prefix_b, remove prefix_a
      - 'prefer=specificity': keep the one with higher clinical_specificity_score

    All removals are logged with SPECIFICITY_WINNER.
    """
    from services.validation_utils import clinical_specificity_score  # local import avoids circular
    to_remove: set[str] = set()

    for pfx_a, pfx_b, prefer in INCOMPATIBLE_CODE_PAIRS:
        a_codes = [d for d in codes if (d.get("code") or "").upper().startswith(pfx_a)]
        b_codes = [d for d in codes if (d.get("code") or "").upper().startswith(pfx_b)]
        if not (a_codes and b_codes):
            continue

        if prefer == "a":
            for d in b_codes:
                to_remove.add((d.get("code") or "").upper())
                logger.info(
                    "SPECIFICITY_WINNER: winner=%s (prefer=%s) removed=%s [conflict resolution]",
                    pfx_a, prefer, d.get("code"),
                )
        elif prefer == "b":
            for d in a_codes:
                to_remove.add((d.get("code") or "").upper())
                logger.info(
                    "SPECIFICITY_WINNER: winner=%s (prefer=%s) removed=%s [conflict resolution]",
                    pfx_b, prefer, d.get("code"),
                )
        else:  # specificity
            best_a = max(a_codes, key=lambda d: clinical_specificity_score(
                d.get("code", ""), d.get("description", "")))
            best_b = max(b_codes, key=lambda d: clinical_specificity_score(
                d.get("code", ""), d.get("description", "")))
            score_a = clinical_specificity_score(best_a.get("code", ""), best_a.get("description", ""))
            score_b = clinical_specificity_score(best_b.get("code", ""), best_b.get("description", ""))
            loser_codes = b_codes if score_a >= score_b else a_codes
            for d in loser_codes:
                to_remove.add((d.get("code") or "").upper())
                logger.info(
                    "SPECIFICITY_WINNER: winner=%s removed=%s [conflict specificity]",
                    (best_a if score_a >= score_b else best_b).get("code"), d.get("code"),
                )

    if to_remove:
        return [d for d in codes if (d.get("code") or "").upper() not in to_remove]
    return codes


# ─────────────────────────────────────────────────────────────────────────────
# ENCOUNTER DOMAIN DEFINITIONS (Step 6)
# ─────────────────────────────────────────────────────────────────────────────
ENCOUNTER_DOMAINS = {
    "neurology": {
        "keywords": ["stroke", "seizure", "encephalopathy", "cva", "tia", "brain", "neurological", "neuro", "hemiparesis"],
        "icd_prefixes": ["I6", "G4", "G8", "R56", "G40"]
    },
    "cardiology": {
        "keywords": ["mi", "chf", "afib", "heart", "stemi", "cardiac", "valve", "atrial", "ventricular", "bradycardia", "tachycardia"],
        "icd_prefixes": ["I1", "I2", "I3", "I4", "I5"]
    },
    "infectious_disease": {
        "keywords": ["sepsis", "infection", "pneumonia", "uti", "cellulitis", "fever", "organism", "abscess", "bacteremia"],
        "icd_prefixes": ["A4", "B1", "J1", "N39", "L03"]
    },
    "orthopedics": {
        "keywords": ["fracture", "dislocation", "arthritis", "hip", "knee", "shoulder", "bone", "joint", "ortho"],
        "icd_prefixes": ["S7", "S8", "M1", "M8"]
    }
}


def is_generic_parent(code_a: str, code_b: str) -> bool:
    """
    Generalized generic-parent detection (Step 3).
    Returns True if code_a represents a generic version of code_b.
    """
    a = code_a.strip().upper()
    b = code_b.strip().upper()
    if a == b:
        return False

    # Prefix match
    prefix_a = a.split(".")[0]
    prefix_b = b.split(".")[0]

    if prefix_a != prefix_b:
        return False

    # Generic parent checks:
    # 1. code_a is shorter (e.g. I63.9 vs I63.411)
    if len(a) < len(b) and b.startswith(a.split(".")[0]):
        # Check if code_a is an ancestor or unspecified sibling
        if a.endswith(".9") or a.endswith(".91") or a.endswith(".0"):
            return True
        # If it's just shorter like I63 vs I63.411
        if "." not in a:
            return True

    return False


def compute_encounter_domain_signature(note_text: str) -> dict[str, float]:
    """
    Determines the dominant clinical domains for the current encounter (Step 6).
    Returns a mapping of domain -> relevance_score (0.0 - 1.0).
    """
    text_lower = note_text.lower()
    scores = {}

    for domain, data in ENCOUNTER_DOMAINS.items():
        hits = 0
        for kw in data["keywords"]:
            if kw in text_lower:
                hits += 1

        # Normalize score based on number of distinct keywords found
        # (3+ hits = 1.0 dominance)
        scores[domain] = min(1.0, hits / 3.0)

    return scores


def compute_encounter_relevance_score(
    code_dict: dict,
    note_text: str,
    note_sections: dict[str, str],
) -> float:
    """
    Step 5: Compute generalized encounter relevance score [0, 1].
    
    Factors:
      1. Section Authority (Assessment > History)
      2. Active Management Linkage (Proximity to treatment/meds)
      3. Procedural Linkage
      4. Temporal Status (Active > Chronic > PMH)
      5. Specialty Alignment
    """
    relevance = 0.40  # baseline relevance
    
    code = (code_dict.get("code") or "").upper()
    description = (code_dict.get("description") or "").lower()
    sec_name = code_dict.get("section_dominant") or "full_note"
    
    # (a) Section Authority (max +0.35)
    sec_weight = SECTION_WEIGHTS.get(sec_name, 0.40)
    if sec_name in ["postop_diagnosis", "preop_diagnosis", "procedure", "findings"]:
        sec_weight = 1.0  # Force maximum relevance for operative sections
    relevance += (sec_weight * 0.35)
    
    # (b) Temporal Alignment (max +0.20)
    status = code_dict.get("temporal_status", "ACTIVE")
    if status == "ACTIVE":
        relevance += 0.20
    elif status == "CHRONIC_MANAGED":
        relevance += 0.10
    
    # (c) Active Management Weighting (max +0.20)
    # Search for management tokens near the description text in the dominant section
    sec_text = note_sections.get(sec_name, note_text).lower()
    desc_words = [w for w in description.split() if len(w) > 3]
    if desc_words:
        anchor = desc_words[0]
        anchor_idx = sec_text.find(anchor)
        if anchor_idx != -1:
            management_found = False
            for cat, terms in MANAGEMENT_INDICATORS.items():
                for term in terms:
                    term_idx = sec_text.find(term)
                    if term_idx != -1 and abs(term_idx - anchor_idx) < 300:
                        management_found = True
                        break
                if management_found: break
            if management_found:
                relevance += 0.20

    # (d) Procedural Linkage (max +0.10)
    # If code is mentioned near 'procedure' or 'operative'
    if any(p in sec_text for p in ["procedure", "operative", "performed", "surgical"]):
        relevance += 0.10

    # (e) Specialty Alignment (Step 6 — Precision Hardening)
    # Penalize if code family does not match detected encounter domains
    domain_sigs = compute_encounter_domain_signature(note_text)
    if domain_sigs:
        has_domain_match = False
        for domain, sig in domain_sigs.items():
            if sig > 0.5:
                prefixes = ENCOUNTER_DOMAINS[domain].get("prefixes", [])
                if prefixes and any(code.startswith(pfx) for pfx in prefixes):
                    has_domain_match = True
                    relevance += 0.10  # Domain alignment boost
                    break
        
        # If encounter has strong domains but code doesn't match any of them
        if not has_domain_match and any(s > 0.7 for s in domain_sigs.values()):
            # Only penalize if it's a major diagnosis code (exclude Z, R, CPT)
            if code[0].isalpha() and code[0] not in ["Z", "R"]:
                relevance -= 0.15 # Cross-specialty penalty
                
    return min(1.0, max(0.0, relevance))


# ─── Task: Temporal & Encounter-Aware Clinical Reasoning Hardening ────────────

_TEMPORAL_MARKERS = {
    "historical":  ["history of", "prior", "previous", "past medical", "h/o", "hx of", "known"],
    "resolved":    ["resolved", "no longer", "remission", "cured", "cleared", "discharged from"],
    "ruled_out":   ["ruled out", "no evidence of", "negative for", "r/o", "without findings of", "unlikely"],
    "suspected":   ["possible", "probable", "suspected", "likely", "concern for", "cannot exclude"],
    "worsening":   ["worsening", "acute on chronic", "exacerbation", "decompensated", "deteriorating"],
    "improving":   ["improving", "resolving", "stable", "responding"],
    "post_op":     ["post-op", "status post", "s/p", "following", "post procedure"],
    "chronic":     ["chronic", "long-standing", "baseline", "underlying", "ongoing"],
}

def compute_temporal_clinical_state(code_dict: dict, note_text: str) -> dict:
    """
    Step 1 — Temporal Clinical State Engine.
    Classifies the temporal status of a clinical entity from its context window.
    Returns temporal_state, temporal_confidence, and active_relevance_score.
    """
    code = code_dict.get("code", "")
    desc = (code_dict.get("description") or "").lower()
    window = note_text.lower()

    state        = "ACTIVE"
    confidence   = 0.70
    active_score = 0.80

    for marker_state, phrases in _TEMPORAL_MARKERS.items():
        if any(p in window for p in phrases):
            if marker_state == "historical":
                state        = "HISTORICAL"
                confidence   = 0.80
                active_score = 0.15
            elif marker_state == "resolved":
                state        = "RESOLVED"
                confidence   = 0.85
                active_score = 0.05
            elif marker_state == "ruled_out":
                state        = "RULED_OUT"
                confidence   = 0.90
                active_score = 0.00
            elif marker_state == "suspected":
                state        = "SUSPECTED"
                confidence   = 0.55
                active_score = 0.45
            elif marker_state == "worsening":
                state        = "ACUTE_WORSENING"
                confidence   = 0.80
                active_score = 0.95
            elif marker_state == "post_op":
                state        = "POST_PROCEDURE"
                confidence   = 0.75
                active_score = 0.50
            elif marker_state == "chronic":
                state        = "CHRONIC_STABLE"
                confidence   = 0.70
                active_score = 0.40
            break  # First match wins; most specific marker should be checked first

    return {
        "temporal_state":       state,
        "temporal_confidence":  confidence,
        "active_relevance_score": active_score,
    }


def compute_advanced_negation_scope(mention: str, context_window: str) -> dict:
    """
    Step 2 — Advanced Negation Scope Expansion.
    Multi-token, window-based negation detection with uncertainty handling.
    """
    _NEGATION_PHRASES = [
        "no evidence of", "negative for", "without findings of",
        "denies", "no ", "without ", "absent", "ruled out",
        "cannot exclude",  # Uncertainty — not hard negation
        "unlikely", "resolved",
    ]
    _UNCERTAIN_PHRASES = ["cannot exclude", "unlikely", "possible", "suspected"]

    window_lower  = context_window.lower()
    mention_lower = mention.lower()

    negation_strength = 0.0
    uncertainty_mod   = 0.0
    neg_tokens: list[str] = []

    for phrase in _NEGATION_PHRASES:
        idx = window_lower.find(phrase)
        if idx == -1:
            continue
        # Check if mention appears within 120 chars after the negation phrase
        post_window = window_lower[idx : idx + 120]
        if mention_lower in post_window or any(w in post_window for w in mention_lower.split()[:3]):
            neg_tokens.append(phrase)
            if phrase in _UNCERTAIN_PHRASES:
                uncertainty_mod = max(uncertainty_mod, 0.40)
                negation_strength = max(negation_strength, 0.30)
            else:
                negation_strength = max(negation_strength, 0.85)

    return {
        "negation_scope_strength": negation_strength,
        "uncertainty_modifier":    uncertainty_mod,
        "negation_window_tokens":  neg_tokens,
    }


def compute_encounter_alignment_confidence(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Encounter Isolation Engine.
    Scores how well a diagnosis aligns with the *current* encounter rather than
    imported/historical documentation.
    """
    score = 0.60

    sec = (code_dict.get("section_dominant") or "").lower()
    temporal = code_dict.get("temporal_state", "ACTIVE")
    active_rel = float(code_dict.get("active_relevance_score") or 0.50)

    # Boost for current-encounter sections
    if any(k in sec for k in ["assessment", "discharge", "operative", "attending", "plan"]):
        score += 0.25
    # Penalize imported/stale sections
    if any(k in sec for k in ["pmh", "imported", "nursing", "problem_list"]):
        score -= 0.30

    # Temporal state decay
    if temporal in ["HISTORICAL", "RESOLVED", "RULED_OUT"]:
        score -= 0.40
    elif temporal == "SUSPECTED":
        score -= 0.10

    score += (active_rel - 0.50) * 0.30
    return min(1.0, max(0.0, score))


def compute_specialty_context_weighting(note_text: str) -> dict:
    """
    Step 6 — Specialty-Aware Reasoning.
    Infers likely clinical specialty and returns adjusted weights.
    """
    text = note_text.lower()

    SPECIALTY_SIGNALS = {
        "icu":        ["intubated", "ventilator", "vasopressor", "icu", "critical care", "sicu", "micu"],
        "surgery":    ["operative", "incision", "anastomosis", "laparotomy", "post-op", "sterile field"],
        "cardiology": ["stemi", "catheterization", "pci", "afib", "echo", "ejection fraction", "lvef"],
        "nephrology": ["creatinine", "dialysis", "aki", "ckd", "proteinuria", "renal biopsy"],
        "oncology":   ["chemotherapy", "metastasis", "tumor", "malignancy", "radiation", "staging"],
        "emergency":  ["emergency department", "ed presentation", "triage", "er", "trauma"],
        "neurology":  ["seizure", "stroke", "cva", "mri brain", "neurology consult", "eeg"],
    }

    THRESHOLDS = {
        "icu":        0.55, "surgery":    0.60, "cardiology": 0.55,
        "nephrology": 0.60, "oncology":   0.55, "emergency":  0.65,
        "neurology":  0.60, "default":    0.70,
    }

    detected = "general"
    best_count = 0
    for specialty, signals in SPECIALTY_SIGNALS.items():
        count = sum(1 for s in signals if s in text)
        if count > best_count:
            best_count = count
            detected = specialty

    return {
        "detected_specialty":     detected,
        "evidence_threshold":     THRESHOLDS.get(detected, 0.70),
        "authority_boost":        0.10 if detected != "general" else 0.0,
    }


def calibrate_prediction_confidence(code_dict: dict) -> dict:
    """
    Step 7 — Confidence Calibration Layer.
    Reduces overconfidence, penalizes contradictory evidence, scales by reliability.
    """
    raw_conf   = float(code_dict.get("confidence") or 0.5)
    ev_strength = float(code_dict.get("evidence_strength") or 0.5)
    reliability = float(code_dict.get("DOCUMENT_RELIABILITY_VAL") or 0.6)
    negation    = float(code_dict.get("negation_scope_strength") or 0.0)
    temporal_conf = float(code_dict.get("temporal_confidence") or 0.7)

    # Reliability-weighted blend
    calibrated = (raw_conf * 0.40 + ev_strength * 0.40 + reliability * 0.20)

    # Negation penalty
    calibrated *= (1.0 - negation * 0.80)

    # Temporal confidence scaling
    calibrated *= temporal_conf

    # Uncertainty band
    uncertainty = max(0.0, 1.0 - calibrated) * 0.25
    audit_risk  = 1.0 - calibrated

    return {
        "calibrated_confidence": round(min(1.0, max(0.0, calibrated)), 4),
        "uncertainty_band":      round(uncertainty, 4),
        "audit_risk_score":      round(min(1.0, audit_risk), 4),
    }


def build_code_evidence_graph(code_dict: dict, note_text: str) -> dict:
    """
    Step 5 — Evidence Traceability Graph.
    Constructs structured explainability metadata for each ICD/CPT code.
    """
    code  = code_dict.get("code", "")
    desc  = (code_dict.get("description") or "").lower()

    # Find supporting spans (simple first-occurrence proximity)
    supporting_spans: list[str] = []
    idx = note_text.lower().find(desc[:15]) if desc else -1
    if idx != -1:
        supporting_spans.append(note_text[max(0, idx-30): idx+80].strip())

    graph = {
        "code":                     code,
        "section_origin":           code_dict.get("section_dominant", "unknown"),
        "temporal_state":           code_dict.get("temporal_state", "ACTIVE"),
        "temporal_confidence":      code_dict.get("temporal_confidence", 0.7),
        "supporting_spans":         supporting_spans,
        "negation_tokens":          code_dict.get("negation_window_tokens", []),
        "contradiction_resolved":   "CONTRADICTION_RESOLVED" in code_dict.get("audit_traces", []),
        "reliability_contribution": code_dict.get("DOCUMENT_RELIABILITY_VAL", 0.6),
        "calibrated_confidence":    code_dict.get("calibrated_confidence", None),
        "audit_risk_score":         code_dict.get("audit_risk_score", None),
    }
    return graph


# ─── Task: Evidence Hierarchy & False-Confidence Suppression ──────────────────

def compute_evidence_tier(code_dict: dict) -> int:
    """
    Step 1 — Evidence Tier Model.
    Classifies a code's evidence quality into tiers 1–4.
    TIER 1 = highest authority; TIER 4 = weakest.
    """
    traces = code_dict.get("audit_traces", [])
    sec    = (code_dict.get("section_dominant") or "").lower()
    source = (code_dict.get("source") or "").lower()

    # Tier 1: Direct confirmed, operative, discharge, pathology
    if (
        source == "deterministic"
        or code_dict.get("protected")
        or any(k in sec for k in ["discharge", "operative", "pathology", "final_diagnosis"])
        or "AUTHORITATIVE_EVIDENCE_CONFIRMED" in traces
    ):
        return 1

    # Tier 2: Assessment/impression, specialist, treatment-linked, imaging
    if (
        any(k in sec for k in ["assessment", "plan", "attending", "consult", "radiology"])
        or code_dict.get("PRINCIPAL_ENCOUNTER_LOCKED")
        or "TREATMENT_LINKED_CONFIRMED" in traces
    ):
        return 2

    # Tier 3: Lab/medication inference, physiologic, workflow-derived
    if (
        float(code_dict.get("evidence_strength") or 0) >= 0.55
        and float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0) >= 0.40
    ):
        return 3

    # Tier 4: Ontology similarity, weak semantic inference
    return 4


def compute_direct_grounding_authority(code_dict: dict, note_text: str) -> float:
    """
    Step 4 — Direct Evidence Dominance.
    Scores the density of direct phrase-level grounding for a code,
    dominating semantic / ontology-derived confidence boosts.
    """
    desc  = (code_dict.get("description") or "").lower()
    text  = note_text.lower()
    score = 0.0

    # Exact description match
    if desc and desc[:20] in text:
        score += 0.55

    # Authoritative section grounding
    sec = (code_dict.get("section_dominant") or "").lower()
    if any(k in sec for k in ["discharge", "operative", "assessment", "attending"]):
        score += 0.25

    # Direct management linkage
    if float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0) > 0.60:
        score += 0.15

    # Procedural linkage
    if "PROCEDURAL_BILLING_CONFIRMED" in code_dict.get("audit_traces", []):
        score += 0.15

    return min(1.0, score)


def compute_evidence_conflict_priority(code_dict: dict) -> float:
    """
    Step 6 — Evidence Conflict Priority.
    When evidence conflicts, returns a priority score reflecting
    the authority of the code's best available evidence source.
    Higher = more trusted under conflict.
    """
    tier  = code_dict.get("EVIDENCE_TIER", 4)
    score = {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.20}.get(tier, 0.20)

    # Discharge / operative override
    sec = (code_dict.get("section_dominant") or "").lower()
    if any(k in sec for k in ["discharge", "operative"]):
        score = max(score, 0.90)

    # Protected / deterministic
    if code_dict.get("protected") or (code_dict.get("source") or "") == "deterministic":
        score = 1.0

    return score


# ─── Task: Clinical Relationship Graph & Anatomical Coherence Hardening ───────

# Generalized relationship extractors — purely linguistic, no ICD hardcoding
_RELATIONSHIP_PATTERNS = {
    "disease_complication": ["caused by", "due to", "secondary to", "as a result of", "complicated by"],
    "procedure_indication": ["for treatment of", "indicated for", "performed for", "secondary to", "due to"],
    "treatment_effect":     ["following", "after", "in response to", "due to", "secondary to"],
    "anatomy_pathology":    ["of the", "involving the", "affecting the", "at the site of"],
}

_ANATOMICAL_MARKERS = [
    "left", "right", "bilateral", "upper", "lower", "anterior", "posterior",
    "proximal", "distal", "medial", "lateral", "central", "peripheral",
    "gastric", "femoral", "ureteral", "cerebral", "renal", "hepatic",
    "pulmonary", "coronary", "lumbar", "cervical", "thoracic",
    "femur", "hip", "humerus", "shoulder", "tibia", "fibula", "radius", "ulna",
]


def build_clinical_relationship_graph(codes: list[dict], note_text: str) -> dict:
    """
    Step 1 — Clinical Relationship Graph.
    Extracts generalized encounter-local relationships between code entities.
    Returns an adjacency structure: {code: {"supports": [...], "rel_type": ...}}
    """
    text = note_text.lower()
    graph: dict[str, dict] = {}

    for d in codes:
        code = (d.get("code") or "").upper()
        desc = (d.get("description") or "").lower()
        if not code:
            continue
        graph[code] = {"supports": [], "supported_by": [], "rel_types": []}

        for other in codes:
            o_code = (other.get("code") or "").upper()
            o_desc = (other.get("description") or "").lower()
            if o_code == code or not o_desc:
                continue

            for rel_type, phrases in _RELATIONSHIP_PATTERNS.items():
                for phrase in phrases:
                    pattern = f"{o_desc[:15]}.*{phrase}.*{desc[:15]}"
                    if re.search(pattern, text):
                        graph[code]["supported_by"].append(o_code)
                        graph[code]["rel_types"].append(rel_type)
                        graph[o_code] = graph.get(o_code, {"supports": [], "supported_by": [], "rel_types": []})
                        graph[o_code]["supports"].append(code)
                        break

    return graph


def compute_anatomical_coherence(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Anatomical Coherence Engine.
    Scores how specifically the code's anatomical site/laterality
    is supported in the note text. High score = specific anatomy confirmed.
    """
    desc = (code_dict.get("description") or "").lower()
    text = note_text.lower()
    score = 0.40  # Baseline

    matched_markers = [m for m in _ANATOMICAL_MARKERS if m in desc and m in text]
    score += min(0.45, len(matched_markers) * 0.15)

    # Exact description window match near relevant section
    sec = (code_dict.get("section_dominant") or "").lower()
    if any(k in sec for k in ["assessment", "discharge", "operative", "attending", "postop", "preop", "diagnosis"]):
        score += 0.15

    return min(1.0, score)


def compute_procedural_intent_alignment(code_dict: dict, note_text: str) -> float:
    """
    Step 5 — Procedure-Intent Coherence.
    Infers generalized procedural intent to reinforce anatomically
    coherent diagnoses without hallucinating unsupported conditions.
    """
    desc = (code_dict.get("description") or "").lower()
    text = note_text.lower()

    _INTENT_VERBS = [
        "performed", "indicated", "placed", "resected", "infused",
        "administered", "biopsied", "drained", "stented", "catheterized",
        "repaired", "clipped", "replaced", "implanted",
    ]
    intent_score = 0.0
    for verb in _INTENT_VERBS:
        if verb in text:
            intent_score += 0.12
            if verb in desc:
                intent_score += 0.10  # Double-match boost

    return min(1.0, intent_score)


def compute_relationship_certainty(rel_type: str, code_dict: dict, note_text: str) -> float:
    """
    Step 8 — Relationship Certainty Control.
    Measures confidence in a specific clinical relationship
    based on language, proximity, and section authority.
    """
    text  = note_text.lower()
    score = 0.30  # Baseline; weak co-occurrence is low

    desc = (code_dict.get("description") or "").lower()
    sec  = (code_dict.get("section_dominant") or "").lower()

    # Direct linkage language
    direct_phrases = _RELATIONSHIP_PATTERNS.get(rel_type, [])
    if any(p in text for p in direct_phrases):
        score += 0.30

    # Proximity: description mentioned in authoritative section
    if any(k in sec for k in ["assessment", "discharge", "operative"]) and desc[:12] in text:
        score += 0.25

    # Treatment coherence proxy
    if float(code_dict.get("MANAGEMENT_INTENSITY_VAL") or 0) > 0.50:
        score += 0.15

    return min(1.0, score)


# ─── Task: Provider Assertion & Clinical Severity Reconciliation ───────────────

_AUTHORITY_SECTIONS_HIGH = {"discharge", "final_diagnosis", "operative", "pathology", "attending"}
_AUTHORITY_SECTIONS_MED  = {"assessment", "plan", "consult", "specialist", "radiology", "impression"}
_AUTHORITY_SECTIONS_LOW  = {"nursing", "pmh", "imported", "problem_list", "autogenerated"}

_SEVERITY_KEYWORDS = [
    "hemorrhage", "obstruction", "failure", "shock", "ketoacidosis", "infarction",
    "septic", "pathological", "acute-on-chronic", "neutropenia", "critical",
    "respiratory failure", "organ failure", "decompensated", "severe",
]

_NOS_MARKERS = ["unspecified", "nos", "not otherwise specified", "other", "not elsewhere classified"]


def compute_provider_assertion_strength(code_dict: dict) -> float:
    """
    Step 1 — Provider Assertion Authority.
    Scores how directly this code was asserted by a clinician vs. inferred.
    """
    sec    = (code_dict.get("section_dominant") or "").lower()
    source = (code_dict.get("source") or "").lower()
    traces = code_dict.get("audit_traces", [])

    if source == "deterministic" or code_dict.get("protected"):
        return 1.0
    if any(k in sec for k in _AUTHORITY_SECTIONS_HIGH):
        return 0.90
    if any(k in sec for k in _AUTHORITY_SECTIONS_MED):
        return 0.65
    if "AUTHORITATIVE_EVIDENCE_CONFIRMED" in traces:
        return 0.80
    if any(k in sec for k in _AUTHORITY_SECTIONS_LOW):
        return 0.25
    return 0.50


def compute_clinical_severity_weight(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Clinical Severity Governance.
    Detects generalised severity signals to grant reconciliation resistance.
    """
    desc = (code_dict.get("description") or "").lower()
    text = note_text.lower()
    score = 0.0

    for kw in _SEVERITY_KEYWORDS:
        if kw in desc:
            score += 0.25
        if kw in text:
            score += 0.10

    return min(1.0, score)


def compute_procedural_documentation_strength(code_dict: dict) -> float:
    """
    Step 5 — Procedural Documentation Dominance.
    Highest weighting for codes grounded in operative/IR/procedure reports.
    Prevents false unsupported CPT rejections.
    """
    sec    = (code_dict.get("section_dominant") or "").lower()
    source = (code_dict.get("source") or "").lower()
    traces = code_dict.get("audit_traces", [])

    if source == "deterministic" or code_dict.get("protected"):
        return 1.0
    if any(k in sec for k in ["operative", "procedure", "cath", "interventional", "infusion"]):
        return 0.92
    if "PROCEDURAL_BILLING_CONFIRMED" in traces or "FRAGMENTED_PROCEDURE_RECONSTRUCTED" in traces:
        return 0.78
    if (code_dict.get("type") or "").upper() == "CPT":
        return 0.65
    return 0.30


def compute_overgeneralization_risk(code_dict: dict, all_codes: list[dict]) -> float:
    """
    Step 8 — Overgeneralization Suppression.
    Returns a risk score [0–1] for this code being a generic abstraction
    when a more specific sibling already exists in the code set.
    """
    desc = (code_dict.get("description") or "").lower()
    code = (code_dict.get("code") or "").upper()

    is_generic = any(m in desc for m in _NOS_MARKERS)
    if not is_generic:
        return 0.0  # Not a generic variant — no risk

    # Check if a more specific sibling (same 3-char prefix, longer code) exists
    for other in all_codes:
        o_code = (other.get("code") or "").upper()
        if o_code == code:
            continue
        if o_code.startswith(code[:3]) and len(o_code) > len(code):
            o_ev = float(other.get("evidence_strength") or 0)
            if o_ev >= 0.40:
                return min(1.0, 0.50 + o_ev * 0.50)

    return 0.10  # Generic but no specific sibling found


# ─── Task: Pipeline Convergence & Reasoning Conflict Stabilization ─────────────

# Step 1 — Global Reasoning Priority Map (higher index = higher authority)
GLOBAL_REASONING_PRIORITY: dict[str, int] = {
    "ontology_similarity":      1,
    "semantic_support":         2,
    "encounter_relevance":      3,
    "direct_grounding":         4,
    "relationship_integrated":  5,
    "anatomical_coherence":     6,
    "severe_clinical_state":    7,
    "procedural_documentation": 8,
    "provider_confirmed":       9,
    "temporal_truth":           10,
}


def compute_monotonic_confidence_update(
    current: float, proposed: float, code_dict: dict
) -> float:
    """
    Step 3 — Monotonic Confidence Evolution.
    Bounds confidence changes to prevent wild volatility between passes.
    Strong grounded codes cannot collapse dramatically; weak codes cannot spike.
    """
    tier        = int(code_dict.get("EVIDENCE_TIER") or 4)
    is_locked   = bool(code_dict.get("PROVIDER_TRUTH_LOCKED") or code_dict.get("SEVERITY_LOCK"))
    is_terminal = bool(code_dict.get("TERMINAL_SUPPRESSION"))

    if is_terminal:
        return min(current, proposed)  # Terminally suppressed: never revive

    # Max allowed single-pass change
    if tier == 1 or is_locked:
        max_drop = 0.10   # Strong — allow very small drops only
        max_rise = 0.20
    elif tier == 2:
        max_drop = 0.20
        max_rise = 0.15
    elif tier == 3:
        max_drop = 0.30
        max_rise = 0.10
    else:
        max_drop = 0.40
        max_rise = 0.05   # Tier 4 cannot spike

    delta = proposed - current
    if delta < 0:
        return max(current - max_drop, proposed)
    else:
        return min(current + max_rise, proposed)


def compute_reconciliation_stability(code_dict: dict) -> float:
    """
    Step 6 — Deterministic Reconciliation Stability.
    Computes a stable authority score that survives pass conflicts.
    Prioritises: provider truth > severity > anatomy > relationship > grounding.
    """
    score = 0.0
    score += float(code_dict.get("PROVIDER_ASSERTION_VAL") or 0) * 0.35
    score += float(code_dict.get("CLINICAL_SEVERITY_VAL") or 0) * 0.25
    score += float(code_dict.get("ANATOMICAL_COHERENCE_VAL") or 0) * 0.20
    score += float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0) * 0.15
    score += (1.0 - float(code_dict.get("OVERGENERALIZATION_RISK_VAL") or 0)) * 0.05
    return min(1.0, score)


def compute_lock_strength(code_dict: dict) -> float:
    """
    Step 8 — Locked State Governance.
    Returns the validity strength of a code's existing lock.
    Weak locks decay; strong locks persist only with high grounding quality.
    """
    is_locked = (
        code_dict.get("PROVIDER_TRUTH_LOCKED")
        or code_dict.get("SEVERITY_LOCK")
        or code_dict.get("protected")
    )
    if not is_locked:
        return 0.0

    ev     = float(code_dict.get("evidence_strength") or 0.5)
    tier   = int(code_dict.get("EVIDENCE_TIER") or 4)
    assert_val = float(code_dict.get("PROVIDER_ASSERTION_VAL") or 0.5)

    # Lock is valid only if grounding remains high
    if tier <= 2 and ev >= 0.50 and assert_val >= 0.65:
        return 1.0  # Strong valid lock
    elif tier == 3 and ev >= 0.45:
        return 0.60  # Moderate lock — survives unless higher-priority conflict
    else:
        return 0.20  # Weak lock — decays under conflict


# ─── Task: Threshold Calibration & False-Positive / False-Negative Balancing ──

# Step 1 — Centralized Calibration Thresholds
# CALIBRATION_THRESHOLDS (Consolidated at top of file)

# Step 2 — Specialty Sensitivity Profiles (modifier deltas applied to key thresholds)
_SPECIALTY_MODIFIERS: dict[str, dict[str, float]] = {
    "oncology":    {"severe_diagnosis_floor": -0.05, "relationship_certainty_floor": -0.05},
    "neurology":   {"anatomy_coherence_threshold": -0.05, "overgeneralization_suppress": -0.05},
    "nephrology":  {"severe_diagnosis_floor": -0.05, "semantic_drift_limit": +0.05},
    "pulmonology": {"severe_diagnosis_floor": -0.08, "terminal_negation_threshold": +0.03},
    "orthopedics": {"anatomy_coherence_threshold": -0.08, "overgeneralization_suppress": -0.08},
    "gi":          {"severe_diagnosis_floor": -0.05, "false_positive_ev_ceiling": +0.05},
    "rheumatology":{"anatomy_coherence_threshold": -0.05, "relationship_certainty_floor": -0.05},
    "general":     {},
}


def compute_specialty_calibration_modifier(specialty: str) -> dict[str, float]:
    """
    Step 2 — Specialty-Sensitive Calibration.
    Returns adjusted threshold deltas for the detected specialty.
    Slightly tunes thresholds without creating hardcoded disease logic.
    """
    base = dict(CALIBRATION_THRESHOLDS)
    deltas = _SPECIALTY_MODIFIERS.get(specialty.lower(), {})
    for key, delta in deltas.items():
        if key in base:
            base[key] = max(0.05, min(0.99, base[key] + delta))
    return base


def compute_cross_specialty_stability(code_dict: dict, specialty: str) -> float:
    """
    Step 6 — Cross-Specialty Stability.
    Returns a stability score ensuring specialty-specific boosts
    don't destabilize generalized domain behavior.
    """
    recon = float(code_dict.get("RECONCILIATION_STABILITY_VAL") or 0.5)
    tier  = int(code_dict.get("EVIDENCE_TIER") or 4)

    # ICU / surgery contexts accept slightly lower thresholds for severity
    is_critical = specialty in {"icu", "surgery", "emergency"}
    base_score  = recon * (0.90 if is_critical else 1.0)

    # Penalize if purely semantic (T4) in unfamiliar specialty
    if tier == 4 and not code_dict.get("protected"):
        base_score *= 0.70

    return min(1.0, max(0.0, base_score))


# ─── Task: Retrieval Purity & Candidate Ranking Stabilization ─────────────────

_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "dka":   "diabetic ketoacidosis",
    "htn":   "hypertension",
    "mi":    "myocardial infarction",
    "chf":   "congestive heart failure",
    "arf":   "acute renal failure",
    "aki":   "acute kidney injury",
    "ckd":   "chronic kidney disease",
    "copd":  "chronic obstructive pulmonary disease",
    "pe":    "pulmonary embolism",
    "dvt":   "deep vein thrombosis",
    "afib":  "atrial fibrillation",
    "cva":   "cerebrovascular accident",
    "tia":   "transient ischemic attack",
    "gi":    "gastrointestinal",
    "r mca": "right middle cerebral artery",
    "l mca": "left middle cerebral artery",
    "mca":   "middle cerebral artery",
    "sob":   "shortness of breath",
    "uti":   "urinary tract infection",
    "gib":   "gastrointestinal bleeding",
    "cbc":   "complete blood count",
    "bmp":   "basic metabolic panel",
    "picc":  "peripherally inserted central catheter",
    "cvc":   "central venous catheter",
}

_LATERALITY_MAP = {
    r"\bR\b": "right", r"\bL\b": "left",
    r"\bBi\b": "bilateral", r"\bBilat\b": "bilateral",
    r"\bLT\b": "left", r"\bRT\b": "right",
}

_PROCEDURAL_QUALIFIERS = {
    "us-guided": "ultrasound-guided",
    "ct-guided": "computed tomography guided",
    "lap ":      "laparoscopic ",
    "perc ":     "percutaneous ",
}


def normalize_clinical_terminology(text: str) -> str:
    """
    Step 1 — Clinical Terminology Normalization.
    Standardizes abbreviations, laterality tokens, and procedural qualifiers
    to improve retrieval grounding consistency.
    Preserves severity tokens and anatomical specificity.
    """
    import re
    normalized = text.lower().strip()

    # Abbreviation expansion (whole-word only)
    for abbr, expansion in _ABBREVIATION_EXPANSIONS.items():
        normalized = re.sub(rf"\b{re.escape(abbr)}\b", expansion, normalized)

    # Laterality standardization (case-sensitive short tokens)
    for pattern, replacement in _LATERALITY_MAP.items():
        normalized = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = normalized  # carry forward for chaining

    # Procedural qualifier expansion (prefix patterns)
    for short, full in _PROCEDURAL_QUALIFIERS.items():
        normalized = normalized.replace(short, full)

    return normalized.strip()


def compute_semantic_neighbor_risk(code_dict: dict, all_codes: list[dict]) -> float:
    """
    Step 5 — Semantic Neighbor Pollution Control.
    Scores the risk that this code is a vague semantic sibling
    rather than a clinically grounded candidate.
    """
    desc  = (code_dict.get("description") or "").lower()
    tier  = int(code_dict.get("EVIDENCE_TIER") or 4)
    ev    = float(code_dict.get("evidence_strength") or 0.5)
    overgen = float(code_dict.get("OVERGENERALIZATION_RISK_VAL") or 0)

    # Base risk from tier and evidence
    risk = 0.0
    if tier == 4:
        risk += 0.45
    elif tier == 3:
        risk += 0.20

    # Unspecified/NOS terminology
    if any(m in desc for m in ["unspecified", "nos", "other", "not elsewhere"]):
        risk += 0.25

    # Low direct grounding
    if float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.25:
        risk += 0.20

    # Already flagged as high overgen
    risk += overgen * 0.15

    # Presence of stronger specific sibling deflates risk (it will be suppressed naturally)
    code = (code_dict.get("code") or "").upper()
    for other in all_codes:
        o_code = (other.get("code") or "").upper()
        if o_code != code and o_code.startswith(code[:3]) and len(o_code) > len(code):
            if float(other.get("evidence_strength") or 0) >= 0.50:
                risk += 0.15  # Sibling exists — this one is likely noise
                break

    return min(1.0, risk)


# ─── Task: Integrated Disease State Stabilization Layer ───────────────────────

_INTEGRATED_CONNECTORS = [
    "with", "due to", "secondary to", "associated with", "complicated by",
    "induced by", "acute on chronic", "with hemorrhage", "with nephropathy",
    "with ketoacidosis", "with shock", "with respiratory failure",
    "resulting in", "manifesting as", "complication of"
]

_SEVERE_SYNDROMES = [
    "hemorrhage", "shock", "respiratory failure", "ketoacidosis", "obstruction",
    "pathological fracture", "neutropenia", "encephalopathy", "acute blood loss",
    "organ failure", "sepsis", "infarction", "crisis", "malignant", "severe"
]

def compute_integrated_disease_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Integrated Disease State Detection.
    Detects integrated disease states vs fragmented concepts using linguistic
    connectors and complication linkage.
    """
    desc = (code_dict.get("description") or "").lower()
    strength = 0.0
    
    # 1. Linguistic Connectors (Direct Evidence of Integration)
    matches = [c for c in _INTEGRATED_CONNECTORS if c in desc]
    strength += len(matches) * 0.25
    
    # 2. Severe Syndrome Coupling
    if any(s in desc for s in _SEVERE_SYNDROMES):
        strength += 0.30
        
    # 3. Structural Integration (Specific Subtypes or Combinations)
    if " and " in desc or ", " in desc:
        strength += 0.15
        
    # 4. Multi-signal convergence (Audit Trace Check)
    traces = code_dict.get("audit_traces", [])
    if any(t in traces for t in ["RELATIONSHIP_GRAPH_SUPPORTED", "CLINICAL_RELATIONSHIP_CONFIRMED"]):
        strength += 0.20

    if strength >= 0.40:
        code_dict.setdefault("audit_traces", []).append("INTEGRATED_DISEASE_STATE_CONFIRMED")
        
    return min(1.0, strength)


def compute_specificity_dominance_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Specificity Dominance Weight.
    Measures if a candidate should dominate broader relatives based on
    grounding authority and anatomical/subtype specificity.
    """
    strength = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Direct Grounding Authority
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    strength += grounding * 0.40
    
    # 2. Anatomical/Subtype Specificity
    anatomy = float(code_dict.get("ANATOMICAL_COHERENCE_VAL") or 0)
    strength += anatomy * 0.30
    
    # 3. Avoidance of Generic Indicators
    is_generic = any(m in desc for m in ["unspecified", "nos", "other", "not elsewhere"])
    if not is_generic:
        strength += 0.20
    else:
        strength -= 0.15
        
    # 4. Provider Assertion
    assertion = float(code_dict.get("PROVIDER_ASSERTION_VAL") or 0)
    strength += assertion * 0.20

    if strength >= 0.50:
        code_dict.setdefault("audit_traces", []).append("SPECIFICITY_DOMINANCE_CONFIRMED")
        
    return min(1.0, max(0.0, strength))


def compute_fragmentation_risk(code_dict: dict, all_codes: list[dict]) -> float:
    """
    Step 3 — Fragmentation Risk Detection.
    Detects if a unified disease state is being broken into weaker semantic fragments.
    """
    risk = 0.0
    code = (code_dict.get("code") or "").upper()
    pfx3 = code[:3]
    desc = (code_dict.get("description") or "").lower()
    desc_words = set(desc.split())
    
    for other in all_codes:
        o_code = (other.get("code") or "").upper()
        if o_code == code:
            continue
            
        o_desc = (other.get("description") or "").lower()
        o_words = set(o_desc.split())
        
        # Word overlap suggesting fragmentation
        overlap = len(desc_words & o_words)
        if overlap >= 2 and pfx3 == o_code[:3]:
            risk += 0.25
            
        # Management coherence check
        if float(other.get("MANAGEMENT_COHERENCE_VAL") or 0) > 0.60 and pfx3 != o_code[:3]:
            risk += 0.20

    if risk >= 0.45:
        code_dict.setdefault("audit_traces", []).append("FRAGMENTATION_RISK_DETECTED")
        
    return min(1.0, risk)


# ─── Task: Procedural Subtype Governance & Intervention Stabilization ─────────

_PROCEDURAL_ACTION_VERBS = [
    "performed", "inserted", "clipped", "embolized", "intubated", "stented",
    "biopsied", "drained", "guided", "placed", "aspiration", "resection",
    "excision", "repair", "reconstruction", "delivery", "extraction"
]

_PROCEDURAL_MODALITIES = [
    "fluoroscopic", "ultrasound-guided", "ct-guided", "mri-guided",
    "laparoscopic", "endoscopic", "percutaneous", "open", "robotic"
]

def compute_procedural_grounding_authority(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Procedural Grounding Authority.
    Measures direct procedural grounding quality via section authority, action verbs,
    modality, and workflow sequence.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    desc = (code_dict.get("description") or "").lower()
    strength = 0.0
    
    # 1. Section Authority (Operative/Procedure sections are dominant)
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["operative", "procedure", "interventional", "intervention"]):
        strength += 0.40
    elif any(s in section for s in ["plan", "assessment", "summary"]):
        strength += 0.15
        
    # 2. Action Verbs (Confirmed performance)
    if any(v in desc for v in _PROCEDURAL_ACTION_VERBS):
        strength += 0.25
        
    # 3. Modality Confirmation
    if any(m in desc for m in _PROCEDURAL_MODALITIES):
        strength += 0.20
        
    # 4. Direct Grounding Authority from entities
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    strength += grounding * 0.20

    if strength >= 0.50:
        code_dict.setdefault("audit_traces", []).append("PROCEDURAL_GROUNDING_CONFIRMED")
        
    return min(1.0, strength)


def compute_procedural_subtype_stability(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Procedural Subtype Certainty.
    Determines if subtype-level CPT specificity is supported. 
    Vague semantic similarity triggers a safe downgrade.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 1.0
        
    desc = (code_dict.get("description") or "").lower()
    stability = 0.5 # start at neutral
    
    # Check for specific qualifiers in description
    qualifiers = [
        "guided", "unguided", "open", "percutaneous", "diagnostic", "therapeutic",
        "biopsy", "hemostasis", "simple", "complex", "initial", "subsequent"
    ]
    
    active_qualifiers = [q for q in qualifiers if q in desc]
    if not active_qualifiers:
        return 1.0 # No subtype specificity to worry about
        
    # Evaluate grounding of active qualifiers in note context
    found_count = 0
    for q in active_qualifiers:
        if q in note_text.lower():
            found_count += 1
            stability += 0.20
        else:
            stability -= 0.15
            
    if stability >= 0.70:
        code_dict.setdefault("audit_traces", []).append("PROCEDURAL_SUBTYPE_CONFIRMED")
    elif stability < 0.40:
        code_dict.setdefault("audit_traces", []).append("SUBTYPE_DOWNGRADE_TRIGGERED")
        
    return min(1.0, max(0.0, stability))


def compute_procedural_family_stability(code_dict: dict, all_codes: list[dict]) -> float:
    """
    Step 6 — Procedural Family Governance.
    Maintains coherent procedural family hierarchy (e.g. prioritizing therapeutic over diagnostic).
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 1.0
        
    desc = (code_dict.get("description") or "").lower()
    stability = 0.5
    
    # therapeutic markers
    therapeutic = any(t in desc for t in ["resection", "excision", "repair", "stent", "bypass", "hemostasis", "therapeutic"])
    diagnostic = any(d in desc for d in ["biopsy", "diagnostic", "aspiration", "drainage", "exploratory"])
    
    if therapeutic:
        stability += 0.30
        code_dict.setdefault("audit_traces", []).append("THERAPEUTIC_INTERVENTION_DOMINANT")
        
    # Check for family substitution risk
    # If a diagnostic variant of the same family exists, therapeutic should dominate
    for other in all_codes:
        if (other.get("type") or "").upper() != "CPT": continue
        if other.get("code") == code_dict.get("code"): continue
        
        o_desc = (other.get("description") or "").lower()
        # Simple overlap check for family
        if any(word in o_desc for word in desc.split() if len(word) > 4):
            if therapeutic and any(d in o_desc for d in ["diagnostic", "biopsy"]):
                stability += 0.15
                
    if stability >= 0.65:
        code_dict.setdefault("audit_traces", []).append("PROCEDURAL_FAMILY_STABILIZED")
        
    return min(1.0, stability)


# ─── Task: Specialty Vocabulary Calibration & Semantic Drift Suppression ─────

def compute_specialty_vocabulary_density(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Specialty Vocabulary Density.
    Measures how strongly a candidate aligns with the dominant specialty terminology.
    Uses keywords and prefixes from ENCOUNTER_DOMAINS.
    """
    desc = (code_dict.get("description") or "").lower()
    density = 0.0
    
    # 1. Domain Terminology Check
    for domain, meta in ENCOUNTER_DOMAINS.items():
        kws = meta.get("keywords", [])
        prefixes = meta.get("prefixes", [])
        
        # Word overlap with specialty keywords
        matches = [k for k in kws if k in desc]
        if matches:
            density += len(matches) * 0.15
            
        # Code prefix alignment
        code = (code_dict.get("code") or "").upper()
        if any(code.startswith(p) for p in prefixes):
            density += 0.25
            
    # 2. Section alignment
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["consult", "specialty", "procedure"]):
        density += 0.20
        
    # 3. Direct Grounding check
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    density += grounding * 0.20

    if density >= 0.45:
        code_dict.setdefault("audit_traces", []).append("SPECIALTY_VOCABULARY_CONFIRMED")
        
    return min(1.0, density)


def compute_semantic_drift_risk(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Semantic Drift Detection.
    Detects if a candidate is semantically related but drifting away from
    the actual encounter specialty or context.
    """
    risk = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Low Grounding + High Semantic (Embedding) Score
    # We use RAG_SCORE as a proxy for semantic similarity
    rag_score = float(code_dict.get("rag_score") or 0.5)
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    
    if rag_score > 0.70 and grounding < 0.30:
        risk += 0.40 # High semantic but low grounding
        
    # 2. Cross-domain misalignment
    # If the code's domain doesn't match the detected specialty
    detected_specialty = (code_dict.get("DETECTED_SPECIALTY") or "general").lower()
    if detected_specialty != "general":
        domain_meta = ENCOUNTER_DOMAINS.get(detected_specialty, {})
        prefixes = domain_meta.get("prefixes", [])
        code = (code_dict.get("code") or "").upper()
        
        if prefixes and not any(code.startswith(p) for p in prefixes):
            risk += 0.30 # Domain mismatch
            
    # 3. Vague Abstraction Check
    if any(m in desc for m in ["unspecified", "nos", "other", "not elsewhere"]):
        risk += 0.20

    if risk >= 0.60:
        code_dict.setdefault("audit_traces", []).append("SEMANTIC_DRIFT_DETECTED")
        
    return min(1.0, risk)


# ─── Task: Evidence-Convergence Severity & Encounter-Driver Stabilization ───

def compute_severity_convergence_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Severity Convergence Strength.
    Measures support by converging severe physiologic evidence (ICU, shock, etc).
    """
    desc = (code_dict.get("description") or "").lower()
    strength = 0.0
    
    # 1. Severe physiologic markers
    severe_markers = [
        "shock", "respiratory failure", "organ failure", "ketoacidosis", 
        "infarction", "hemorrhage", "septic", "malignant", "crisis",
        "decompensated", "acute blood loss", "pathological fracture"
    ]
    if any(m in desc for m in severe_markers):
        strength += 0.35
        
    # 2. Management Intensity
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["icu", "critical", "procedure", "operative"]):
        strength += 0.25
        
    # 3. Treatment Intensity (e.g. insulin infusion, vasopressors - proxies)
    # Check for management markers in audit traces
    traces = code_dict.get("audit_traces", [])
    if "MULTISIGNAL_SEVERITY_SUPPORTED" in traces or "PROCEDURE_SURVIVAL_CONFIRMED" in traces:
        strength += 0.20
        
    # 4. Direct grounding weight
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    strength += grounding * 0.20

    if strength >= 0.55:
        code_dict.setdefault("audit_traces", []).append("SEVERITY_CONVERGENCE_CONFIRMED")
        
    return min(1.0, strength)


def compute_encounter_driver_dominance(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Encounter Driver Dominance.
    Determines if diagnosis is one of the primary clinical drivers.
    """
    dominance = 0.0
    
    # 1. Assessment/Plan or Discharge Section dominance
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["assessment", "plan", "discharge", "principal", "admission"]):
        dominance += 0.40
        
    # 2. Repeated mention/density (proxy from entity score)
    entity_score = float(code_dict.get("entity_score") or 0.5)
    dominance += entity_score * 0.20
    
    # 3. Procedural linkage
    if code_dict.get("PROCEDURAL_GROUNDING_VAL", 0) > 0.60:
        dominance += 0.25
        
    # 4. Severity convergence alignment
    sev_conv = compute_severity_convergence_strength(code_dict, note_text)
    dominance += sev_conv * 0.15

    if dominance >= 0.65:
        code_dict.setdefault("audit_traces", []).append("ENCOUNTER_DRIVER_DOMINANT")
        
    return min(1.0, dominance)


def compute_physiologic_coherence(code_dict: dict, note_text: str) -> float:
    """
    Step 5 — Physiologic Coherence Governance.
    Measures coherence with physiologic state (labs, O2 req, metabolic derangement).
    """
    coherence = 0.5
    desc = (code_dict.get("description") or "").lower()
    
    # physiologic state markers
    markers = [
        "hypoxia", "acidosis", "hypotension", "tachycardia", "azotemia",
        "hyperkalemia", "anemia", "leukocytosis", "thrombocytopenia", "ketones"
    ]
    
    found_markers = [m for m in markers if m in note_text.lower()]
    # If diagnosis explains found markers
    explained = [m for m in found_markers if m in desc]
    if explained:
        coherence += len(explained) * 0.15
        
    # ICU/Critical workflow check
    if "icu" in note_text.lower() or "critical care" in note_text.lower():
        if float(code_dict.get("CLINICAL_SEVERITY_VAL") or 0) > 0.70:
            coherence += 0.20
            
    if coherence >= 0.70:
        code_dict.setdefault("audit_traces", []).append("PHYSIOLOGIC_COHERENCE_CONFIRMED")
        
    return min(1.0, coherence)


# ─── Task: Sparse-Evidence Survival & Rare-Specialty Stability Hardening ─────

def compute_sparse_evidence_authority(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Sparse Evidence Authority.
    Identifies diagnoses that are sparsely mentioned but highly authoritative.
    """
    authority = 0.0
    
    # 1. Section Authority (Pathology, Operative, Discharge are dominant)
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["pathology", "operative", "procedure", "discharge", "consult"]):
        authority += 0.45
        
    # 2. Provider Assertion Strength
    assertion = float(code_dict.get("PROVIDER_ASSERTION_VAL") or 0)
    authority += assertion * 0.25
    
    # 3. Direct Grounding Authority
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    authority += grounding * 0.20
    
    # 4. Anatomical/Syndromic specificity
    if float(code_dict.get("ANATOMICAL_COHERENCE_VAL") or 0) >= 0.70:
        authority += 0.15

    if authority >= 0.60:
        code_dict.setdefault("audit_traces", []).append("SPARSE_EVIDENCE_AUTHORITY_CONFIRMED")
        
    return min(1.0, authority)


def compute_rare_specialty_density(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Rare Specialty Density.
    Detects concentrated specialty terminology even when total evidence volume is low.
    """
    density = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # Rare specialties keyword check
    rare_specialties = [
        "rheumatology", "oncology", "transplant", "urology", "vascular",
        "nephrology", "immunology", "hematology", "genetics"
    ]
    
    # 1. Detected specialty check
    spec = (code_dict.get("DETECTED_SPECIALTY") or "general").lower()
    if spec in rare_specialties:
        density += 0.40
        
    # 2. Terminology cluster check
    from services.validation_utils import ENCOUNTER_DOMAINS
    meta = ENCOUNTER_DOMAINS.get(spec, {})
    kws = meta.get("keywords", [])
    if any(k in desc for k in kws):
        density += 0.30
        
    # 3. Vocabulary density check
    vocab_density = float(code_dict.get("SPECIALTY_VOCAB_DENSITY_VAL") or 0)
    density += vocab_density * 0.30

    if density >= 0.50:
        code_dict.setdefault("audit_traces", []).append("RARE_SPECIALTY_CONTEXT_CONFIRMED")
        
    return min(1.0, density)


# ─── Task: Candidate Purity & Semantic Saturation Reduction ──────────────────

def compute_candidate_purity_score(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Candidate Purity Scoring.
    Measures how clinically pure and directly grounded a candidate is before reasoning.
    """
    purity = 0.0
    
    # 1. Direct Grounding Authority (Principal factor)
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    purity += grounding * 0.40
    
    # 2. Specialty alignment
    spec_alignment = float(code_dict.get("SPECIALTY_VOCAB_DENSITY_VAL") or 0.5)
    purity += spec_alignment * 0.20
    
    # 3. Anatomical coherence
    anatomy = float(code_dict.get("ANATOMICAL_COHERENCE_VAL") or 0)
    purity += anatomy * 0.15
    
    # 4. Procedural linkage
    if code_dict.get("PROCEDURAL_GROUNDING_VAL", 0) > 0.60:
        purity += 0.15
        
    # 5. Semantic drift penalty (Inverse of risk)
    drift_risk = float(code_dict.get("SEMANTIC_DRIFT_RISK_VAL") or 0)
    purity += (1.0 - drift_risk) * 0.10

    if purity >= 0.65:
        code_dict.setdefault("audit_traces", []).append("CANDIDATE_PURITY_CONFIRMED")
        
    return min(1.0, purity)


def compute_semantic_saturation_risk(code_dict: dict, all_candidates: list[dict]) -> float:
    """
    Step 2 — Semantic Saturation Detection.
    Detects if too many semantically similar candidates are competing within the same family.
    """
    risk = 0.0
    code_pfx = (code_dict.get("code") or "")[:3]
    
    if not code_pfx:
        return 0.0
        
    # Count siblings in the same 3-character prefix family
    siblings = [c for c in all_candidates if (c.get("code") or "").startswith(code_pfx)]
    sibling_count = len(siblings)
    
    if sibling_count >= 5:
        risk += 0.40 # High family crowding
    elif sibling_count >= 3:
        risk += 0.20
        
    # Check for NOS/Unspecified density in the family
    nos_siblings = [c for c in siblings if "NOS" in (c.get("code") or "") or "unspecified" in (c.get("description") or "").lower()]
    if len(nos_siblings) >= 2:
        risk += 0.30
        
    # Semantic similarity check (proxy: rag_score overlap)
    my_rag = float(code_dict.get("rag_score") or 0)
    similar_rag = [c for c in siblings if abs(float(c.get("rag_score") or 0) - my_rag) < 0.05]
    if len(similar_rag) >= 3:
        risk += 0.30

    if risk >= 0.60:
        code_dict.setdefault("audit_traces", []).append("SEMANTIC_SATURATION_DETECTED")
        
    return min(1.0, risk)


# ─── Task: Lightweight Adaptive Calibration Stabilization ────────────────────

def compute_domain_adaptive_profile(all_codes: list[dict], note_text: str) -> dict:
    """
    Step 1 — Domain Calibration Profiles.
    Generates lightweight adaptive modifiers from encounter characteristics.
    """
    profile = {
        "suppression_modifier": 1.0,
        "specificity_modifier": 1.0,
        "sparse_survival_modifier": 1.0,
        "semantic_penalty_modifier": 1.0,
        "procedural_survival_modifier": 1.0
    }
    
    if not all_codes:
        return profile
        
    # 1. Measure aggregate characteristics
    total = len(all_codes)
    high_spec = len([c for c in all_codes if len(c.get("code") or "") > 5])
    rare_spec = len([c for c in all_codes if "RARE_SPECIALTY_CONTEXT_CONFIRMED" in c.get("audit_traces", [])])
    saturated = len([c for c in all_codes if "SEMANTIC_SATURATION_DETECTED" in c.get("audit_traces", [])])
    grounded = len([c for c in all_codes if float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) > 0.60])
    
    # 2. Derive modifiers (bounded)
    # High saturation → stronger suppression and specificity preservation
    if (saturated / total) > 0.40:
        profile["suppression_modifier"] += 0.15
        profile["specificity_modifier"] += 0.20
        profile["semantic_penalty_modifier"] += 0.10
        
    # Sparse specialty density → stronger rare survival
    if (rare_spec / total) > 0.25:
        profile["sparse_survival_modifier"] += 0.25
        profile["specificity_modifier"] += 0.15
        
    # Low aggregate grounding → higher FP sensitivity (higher semantic penalty)
    if (grounded / total) < 0.30:
        profile["semantic_penalty_modifier"] += 0.20
        profile["suppression_modifier"] += 0.10
        
    # Note compactness proxy (short note + specific codes)
    if len(note_text) < 1000 and high_spec > 3:
        profile["sparse_survival_modifier"] += 0.15
        
    # Trace builder
    # (Traces are normally on codes, but we'll mark the first one to signify profile build)
    all_codes[0].setdefault("audit_traces", []).append("DOMAIN_ADAPTIVE_PROFILE_BUILT")
    
    # Final Bounding
    for k in profile:
        profile[k] = min(1.5, max(0.5, profile[k]))
        
    return profile


# ─── Task: Final Specificity & Dominant-Syndrome Governance ──────────────────

def compute_dominant_clinical_state_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Dominant Clinical State Governance.
    Determines whether a diagnosis represents the primary encounter-defining syndrome.
    """
    strength = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Section Authority
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["assessment", "plan", "discharge", "principal", "admission"]):
        strength += 0.35
        
    # 2. Procedure Linkage
    if code_dict.get("PROCEDURAL_GROUNDING_VAL", 0) > 0.60:
        strength += 0.25
        
    # 3. Severity & Physiological Evidence
    if "SEVERITY_CONVERGENCE_CONFIRMED" in code_dict.get("audit_traces", []):
        strength += 0.20
    if "PHYSIOLOGIC_COHERENCE_CONFIRMED" in code_dict.get("audit_traces", []):
        strength += 0.15
        
    # 4. Multi-signal/Direct Grounding
    grounding = float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    strength += grounding * 0.15
    
    # Negative signals: Symptom or vague markers
    symptom_markers = ["pain", "nausea", "fever", "edema", "cough", "weakness"]
    if any(m == desc for m in symptom_markers) or "unspecified" in desc or " nos" in desc:
        strength *= 0.5

    if strength >= 0.65:
        code_dict.setdefault("audit_traces", []).append("DOMINANT_CLINICAL_STATE_CONFIRMED")
        
    return min(1.0, strength)


def compute_specificity_survival_priority(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Specificity Survival Governance.
    Prevents specific diagnoses from collapsing into generic semantic siblings.
    """
    priority = 0.0
    code = (code_dict.get("code") or "").upper()
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Code-length/Specificity proxy (Generalized)
    if len(code.replace(".", "")) >= 5:
        priority += 0.40
        
    # 2. Anatomical/Laterality qualifiers
    anatomy = float(code_dict.get("ANATOMICAL_COHERENCE_VAL") or 0)
    priority += anatomy * 0.30
    
    # 3. Direct Phrase Grounding
    if "EXACT_PHRASE_MATCHED" in code_dict.get("audit_traces", []) or "DOMAIN_VOCABULARY_CONFIRMED" in code_dict.get("audit_traces", []):
        priority += 0.25
        
    # 4. Severity Qualifiers
    if any(m in desc for m in ["acute", "chronic", "severe", "malignant", "decompensated"]):
        priority += 0.15
        
    # Penalize NOS/Unspecified
    if "NOS" in code or "unspecified" in desc:
        priority -= 0.50
        code_dict.setdefault("audit_traces", []).append("GENERIC_ABSTRACTION_PENALIZED")

    if priority >= 0.70:
        code_dict.setdefault("audit_traces", []).append("SPECIFICITY_SURVIVAL_CONFIRMED")
        
    return min(1.0, max(0.0, priority))


def compute_combination_state_integrity(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Combination State Preservation.
    Preserves integrated disease states from semantic decomposition.
    """
    integrity = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Integrated disease state marker
    if "INTEGRATED_DISEASE_STATE_CONFIRMED" in code_dict.get("audit_traces", []):
        integrity += 0.40
        
    # 2. Causal link indicators (Generalized keywords)
    causal_markers = [" due to ", " with ", " secondary to ", " manifest ", " related to "]
    if any(m in desc for m in causal_markers) or any(m in note_text.lower() for m in causal_markers):
        integrity += 0.30
        
    # 3. Multi-signal support for both components (Complex grounding)
    if float(code_dict.get("DIRECT_GROUNDING_AUTHORITY") or 0) > 0.65:
        integrity += 0.20
        
    # 4. Relationship graph alignment
    if "RELATIONSHIP_GRAPH_SUPPORTED" in code_dict.get("audit_traces", []):
        integrity += 0.15

    if integrity >= 0.65:
        code_dict.setdefault("audit_traces", []).append("COMBINATION_STATE_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("INTEGRATED_STATE_PRIORITY_CONFIRMED")
    elif integrity < 0.30 and "unspecified" in desc:
        code_dict.setdefault("audit_traces", []).append("SEMANTIC_FRAGMENTATION_DETECTED")
        
    return min(1.0, integrity)


# ─── Task: Procedural–Diagnostic Intent Coherence Hardening ──────────────────

def compute_procedural_intent_authority(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Procedural Intent Authority.
    Determine how strongly a procedure defines the clinical intent of the encounter.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    authority = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Operative/Procedure section authority
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["operative", "procedure", "surgical", "interventional"]):
        authority += 0.40
        
    # 2. Intervention verbs (Generalized)
    intervention_verbs = ["thrombectomy", "stent", "biopsy", "drainage", "infusion", "placement", "shunting", "fixation"]
    if any(v in desc for v in intervention_verbs):
        authority += 0.30
        
    # 3. Grounding authority
    grounding = float(code_dict.get("PROCEDURAL_GROUNDING_VAL") or 0)
    authority += grounding * 0.20
    
    # 4. Specialty alignment
    if "SPECIALTY_VOCABULARY_CONFIRMED" in code_dict.get("audit_traces", []):
        authority += 0.10

    if authority >= 0.65:
        code_dict.setdefault("audit_traces", []).append("PROCEDURAL_INTENT_AUTHORITY_CONFIRMED")
        if authority >= 0.80:
            code_dict.setdefault("audit_traces", []).append("INTERVENTION_DRIVEN_ENCOUNTER_DETECTED")
            
    return min(1.0, authority)


def compute_procedure_diagnosis_coherence(code_dict: dict, note_text: str, all_codes: list[dict]) -> float:
    """
    Step 2 — Procedure–Diagnosis Coherence.
    Measure whether a diagnosis is coherently supported by documented interventions.
    """
    if (code_dict.get("type") or "").upper() != "ICD":
        return 0.0
        
    coherence = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Identify grounded procedures in the pool
    procedures = [c for c in all_codes if (c.get("type") or "").upper() == "CPT" and float(c.get("evidence_strength") or 0) > 0.60]
    
    for p in procedures:
        p_desc = (p.get("description") or "").lower()
        
        # 2. Anatomical overlap check (Generalized)
        p_anatomy = p.get("anatomy_regions") or []
        d_anatomy = code_dict.get("anatomy_regions") or []
        if any(a in d_anatomy for a in p_anatomy):
            coherence += 0.35
            
        # 3. Intent linkage (Keyword-based generalized mapping)
        # We look for common roots between diagnosis and procedure
        # e.g. "renal" in both, "stroke" and "thrombectomy" (handled by intent verbs)
        common_roots = ["renal", "hepatic", "cardiac", "cerebral", "pulmonary", "gastric", "neoplasm", "malignant"]
        if any(r in desc and r in p_desc for r in common_roots):
            coherence += 0.30
            
        # 4. Direct relationship (Proximal linkage in note)
        # If diagnosis and procedure are mentioned close together
        if "RELATIONSHIP_GRAPH_SUPPORTED" in code_dict.get("audit_traces", []):
            coherence += 0.25

    if coherence >= 0.60:
        code_dict.setdefault("audit_traces", []).append("PROCEDURE_DIAGNOSIS_ALIGNMENT_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("INTERVENTION_CONTEXT_LINKED")
        
    return min(1.0, coherence)


def compute_therapeutic_priority_strength(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Therapeutic Priority Governance.
    Distinguish therapeutic interventions from diagnostic/supportive ones.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    priority = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Therapeutic keywords (Generalized)
    therapeutic_markers = ["removal", "fixation", "stenting", "infusion", "therapy", "drainage", "shunting", "hemostatic", "clipping", "coiling"]
    diagnostic_markers = ["imaging", "screening", "observation", "diagnostic", "exploration"]
    
    if any(m in desc for m in therapeutic_markers):
        priority += 0.50
        code_dict.setdefault("audit_traces", []).append("THERAPEUTIC_INTERVENTION_CONFIRMED")
    elif any(m in desc for m in diagnostic_markers):
        priority -= 0.30
        
    # 2. Workflow intensity (Proxy: operative section)
    section = (code_dict.get("section_dominant") or "").lower()
    if "operative" in section or "surgical" in section:
        priority += 0.35
        
    # 3. Clinical severity alignment
    if "SEVERITY_CONVERGENCE_CONFIRMED" in code_dict.get("audit_traces", []):
        priority += 0.15

    if priority >= 0.65:
        code_dict.setdefault("audit_traces", []).append("THERAPEUTIC_PRIORITY_GRANTED")
        
    return min(1.0, max(0.0, priority))


# ─── Task: Clinical Causality & State-Transition Governance ──────────────────

def compute_clinical_causality_authority(code_dict: dict, note_text: str, all_codes: list[dict]) -> float:
    """
    Step 1 — Clinical Causality Authority.
    Determine whether a diagnosis relationship reflects true clinical causality.
    """
    authority = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Causal language check (Generalized)
    causal_markers = [" due to ", " with ", " secondary to ", " manifest ", " related to ", " caused by ", " resultant "]
    if any(m in desc for m in causal_markers) or any(m in note_text.lower() for m in causal_markers):
        authority += 0.35
        
    # 2. Procedural reinforcement linkage
    if "INTERVENTION_CONTEXT_LINKED" in code_dict.get("audit_traces", []):
        authority += 0.25
        
    # 3. Treatment linkage (Generalized proxies)
    if "THERAPEUTIC_INTERVENTION_CONFIRMED" in code_dict.get("audit_traces", []):
        authority += 0.20
        
    # 4. Multi-signal convergence (Integrated disease state)
    if "INTEGRATED_DISEASE_STATE_CONFIRMED" in code_dict.get("audit_traces", []):
        authority += 0.20

    if authority >= 0.65:
        code_dict.setdefault("audit_traces", []).append("CLINICAL_CAUSALITY_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("PATHOPHYSIOLOGIC_LINK_CONFIRMED")
        
    return min(1.0, authority)


def compute_state_transition_coherence(code_dict: dict, note_text: str, all_codes: list[dict]) -> float:
    """
    Step 2 — State-Transition Coherence.
    Identify coherent transitions from base disease to escalation.
    """
    coherence = 0.0
    
    # 1. Severity escalation signal
    if "SEVERITY_CONVERGENCE_CONFIRMED" in code_dict.get("audit_traces", []):
        coherence += 0.40
        code_dict.setdefault("audit_traces", []).append("SEVERITY_ESCALATION_CONFIRMED")
        
    # 2. Intervention alignment
    if "PROCEDURAL_INTENT_AUTHORITY_CONFIRMED" in code_dict.get("audit_traces", []):
        coherence += 0.30
        
    # 3. Physiologic progression (Proxies from traces)
    if "PHYSIOLOGIC_COHERENCE_CONFIRMED" in code_dict.get("audit_traces", []):
        coherence += 0.20
        
    # 4. Temporal coherence (Heuristic: Mention density in AP/Discharge)
    if "ENCOUNTER_DRIVER_DOMINANT" in code_dict.get("audit_traces", []):
        coherence += 0.10

    if coherence >= 0.60:
        code_dict.setdefault("audit_traces", []).append("STATE_TRANSITION_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("CLINICAL_PROGRESSION_CONFIRMED")
        
    return min(1.0, coherence)


def compute_complication_dominance_strength(code_dict: dict, note_text: str, all_codes: list[dict]) -> float:
    """
    Step 3 — Complication Dominance Strength.
    Determine whether complication should dominate parent disease.
    """
    strength = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Complication markers (Generalized)
    if any(m in desc for m in ["complication", "acute", "hemorrhage", "failure", "crisis", "pathological"]):
        strength += 0.35
        
    # 2. Intervention linkage
    if "INTERVENTION_DRIVEN_SEVERITY_CONFIRMED" in code_dict.get("audit_traces", []):
        strength += 0.30
        
    # 3. Dominant state alignment
    if "DOMINANT_CLINICAL_STATE_CONFIRMED" in code_dict.get("audit_traces", []):
        strength += 0.20
        
    # 4. Multi-signal/Pathophysiologic link
    if "PATHOPHYSIOLOGIC_LINK_CONFIRMED" in code_dict.get("audit_traces", []):
        strength += 0.15

    if strength >= 0.70:
        code_dict.setdefault("audit_traces", []).append("COMPLICATION_DOMINANCE_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("SEVERE_COMPLICATION_PRIORITY_GRANTED")
        
    return min(1.0, strength)


# ─── Task: Temporal State & Encounter Timeline Governance ────────────────────

def compute_temporal_encounter_authority(code_dict: dict, note_text: str) -> float:
    """
    Step 1 — Temporal Encounter State Authority.
    Determine whether a diagnosis/procedure is ACTIVE, HISTORICAL, RESOLVED, etc.
    """
    authority = 0.5 # Neutral baseline
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Active Temporal Signals
    active_phrases = ["currently", "ongoing", "acute", "underwent", "confirmed", "requiring", "active", "presently"]
    if any(p in desc for p in active_phrases) or any(p in note_text.lower() for p in active_phrases):
        authority += 0.30
        
    # 2. Non-Active Temporal Signals (Generalized)
    nonactive_phrases = ["history of", "remote", "resolved", "rule out", "possible", "prophylaxis", "planned", "scheduled", "monitoring"]
    if any(p in desc for p in nonactive_phrases) or any(p in note_text.lower() for p in nonactive_phrases):
        authority -= 0.40
        
    # 3. Section Authority
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["history", "pmh", "social", "family"]):
        authority -= 0.30
    elif any(s in section for s in ["assessment", "plan", "operative", "procedure", "discharge"]):
        authority += 0.20
        
    # 4. Completion verbs (Procedural timing)
    if "COMPLETED_INTERVENTION_CONFIRMED" in code_dict.get("audit_traces", []):
        authority += 0.20

    if authority >= 0.70:
        code_dict.setdefault("audit_traces", []).append("TEMPORAL_STATE_CONFIRMED")
        code_dict.setdefault("audit_traces", []).append("ACTIVE_ENCOUNTER_STATE_CONFIRMED")
        
    return min(1.0, max(0.0, authority))


def compute_procedural_timeline_coherence(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Procedural Timeline Governance.
    Distinguish planned, completed, and prior procedures.
    """
    if (code_dict.get("type") or "").upper() != "CPT":
        return 0.0
        
    coherence = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Completion verbs/Past tense (Generalized)
    completion_markers = ["completed", "performed", "underwent", "inserted", "placed", "removed", "fixed"]
    if any(m in desc for m in completion_markers) or any(m in note_text.lower() for m in completion_markers):
        coherence += 0.50
        code_dict.setdefault("audit_traces", []).append("COMPLETED_INTERVENTION_CONFIRMED")
        
    # 2. Planning markers (Suppression)
    planning_markers = ["planned", "scheduled", "possible", "referral", "discussion", "consider"]
    if any(m in desc for m in planning_markers) or any(m in note_text.lower() for m in planning_markers):
        coherence -= 0.40
        code_dict.setdefault("audit_traces", []).append("PLANNED_PROCEDURE_SUPPRESSED")
        
    # 3. Section linkage
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["operative", "procedure", "surgical"]):
        coherence += 0.30

    if coherence >= 0.65:
        code_dict.setdefault("audit_traces", []).append("PROCEDURE_TIMELINE_CONFIRMED")
        
    return min(1.0, max(0.0, coherence))


def compute_historical_leakage_risk(code_dict: dict, note_text: str) -> float:
    """
    Step 3 — Historical Leakage Suppression.
    Detect when historical/resolved states are surviving incorrectly.
    """
    risk = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Historical markers
    historical_markers = ["prior", "old", "remote", "history of", "pmh", "previously", "resolved"]
    if any(m in desc for m in historical_markers) or any(m in note_text.lower() for m in historical_markers):
        risk += 0.60
        
    # 2. Section context
    section = (code_dict.get("section_dominant") or "").lower()
    if any(s in section for s in ["pmh", "past medical history", "history"]):
        risk += 0.25
        
    # 3. Absence of active signals (Suppression resistance)
    if "ACTIVE_ENCOUNTER_STATE_CONFIRMED" not in code_dict.get("audit_traces", []):
        risk += 0.15
    else:
        risk -= 0.40 # Active state confirmed, lower risk

    if risk >= 0.70:
        code_dict.setdefault("audit_traces", []).append("HISTORICAL_LEAKAGE_DETECTED")
        code_dict.setdefault("audit_traces", []).append("NONACTIVE_STATE_SUPPRESSED")
        
    return min(1.0, max(0.0, risk))


# ─── Task: Document Structure Governance & Note Reliability Hardening ────────

def compute_document_structure_authority(code_dict: dict) -> float:
    """
    Step 1 — Document Structure Reliability.
    Assign authority weights to note sections.
    """
    authority = 0.5 # Neutral baseline
    section = (code_dict.get("section_dominant") or "").lower()
    
    # High Authority
    high_authority_sections = [
        "operative report", "discharge diagnosis", "assessment", "plan", 
        "pathology", "cath report", "procedure note", "icu assessment"
    ]
    if any(s in section for s in high_authority_sections):
        authority += 0.40
        code_dict.setdefault("audit_traces", []).append("HIGH_AUTHORITY_SECTION_CONFIRMED")
        
    # Low Authority
    low_authority_sections = [
        "ros", "nursing", "pmh", "medication list", "discharge instructions", 
        "problem list", "template", "administrative"
    ]
    if any(s in section for s in low_authority_sections):
        authority -= 0.35
        code_dict.setdefault("audit_traces", []).append("LOW_AUTHORITY_SECTION_DOWNRANKED")
        
    if authority >= 0.70:
        code_dict.setdefault("audit_traces", []).append("DOCUMENT_STRUCTURE_AUTHORITY_CONFIRMED")
        
    return min(1.0, max(0.0, authority))


def compute_copy_forward_risk(code_dict: dict, note_text: str) -> float:
    """
    Step 2 — Template & Copy-Forward Detection.
    Detect duplicated text blocks and templated carry-forward content.
    """
    risk = 0.0
    desc = (code_dict.get("description") or "").lower()
    
    # 1. Imported list formatting (Heuristic)
    if "----" in note_text or "====" in note_text or ":" in desc:
        risk += 0.20
        
    # 2. Medication-only carry-forward proxy
    if "medication" in (code_dict.get("section_dominant") or "").lower():
        risk += 0.40
        
    # 3. Repeated phrase density (Check if description appears many times in a list format)
    mentions = note_text.lower().count(desc)
    if mentions > 5 and len(desc) > 5:
        risk += 0.30
        
    # 4. Stale temporal wording in proximity
    if "STALE_TIMELINE_FRAGMENT_SUPPRESSED" in code_dict.get("audit_traces", []):
        risk += 0.15

    if risk >= 0.65:
        code_dict.setdefault("audit_traces", []).append("COPY_FORWARD_CONTAMINATION_DETECTED")
        code_dict.setdefault("audit_traces", []).append("STALE_TEMPLATE_CONTENT_SUPPRESSED")
        
    return min(1.0, risk)


def compute_section_intent_reliability(code_dict: dict) -> float:
    """
    Step 3 — Section Intent Governance.
    Determine whether section intent is diagnostic/therapeutic vs monitoring/admin.
    """
    reliability = 0.5
    section = (code_dict.get("section_dominant") or "").lower()
    
    # 1. Diagnostic/Therapeutic intent
    active_intent = ["assessment", "plan", "operative", "procedure", "pathology", "discharge diagnosis"]
    if any(s in section for s in active_intent):
        reliability += 0.40
        code_dict.setdefault("audit_traces", []).append("SECTION_INTENT_CONFIRMED")
        
    # 2. Low intent (Monitoring/Admin/Instructional)
    passive_intent = ["instruction", "education", "checklist", "monitoring", "ros", "pmh", "social"]
    if any(s in section for s in passive_intent):
        reliability -= 0.40
        code_dict.setdefault("audit_traces", []).append("LOW_INTENT_RELIABILITY_SUPPRESSED")
        
    return min(1.0, max(0.0, reliability))


# ─── Task: Benchmark Evaluation & Error Analysis Framework ───────────────────

def classify_audit_failure_type(pred: dict, expected: dict, note_text: str) -> str:
    """
    Step 1 — Structured Error Taxonomy.
    Categorize incorrect predictions into a structured taxonomy.
    """
    p_code = (pred.get("code") or "").upper()
    e_code = (expected.get("code") or "").upper()
    
    # 1. Specificity Collapse (Predicted is parent/prefix of expected)
    if e_code.startswith(p_code) and len(p_code) < len(e_code):
        return "SPECIFICITY_COLLAPSE"
        
    # 2. Symptom Overcoding
    if p_code.startswith("R") and "INTEGRAL_SYMPTOM_TERMINALLY_SUPPRESSED" in pred.get("audit_traces", []):
        return "SYMPTOM_OVERCODING"
        
    # 3. Manifestation Fragmentation
    if "FRAGMENTED_RELATIVE_SUPPRESSED" in pred.get("audit_traces", []):
        return "MANIFESTATION_FRAGMENTATION"
        
    # 4. Procedure Subtype Downgrade
    if (pred.get("type") or "").upper() == "CPT" and p_code != e_code:
        if "SAFE_SUBTYPE_RECONCILIATION_APPLIED" in pred.get("audit_traces", []):
            return "PROCEDURE_SUBTYPE_DOWNGRADE"
            
    # 5. Temporal Leakage
    if "HISTORICAL_LEAKAGE_DETECTED" in pred.get("audit_traces", []):
        return "TEMPORAL_LEAKAGE"
        
    # 6. Hallucination (No grounding and low purity)
    if float(pred.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.20:
        return "SEMANTIC_HALLUCINATION"
        
    # 7. Combination State Fragmentation
    if "COMBINATION_STATE_TERMINALLY_CONFIRMED" in pred.get("audit_traces", []):
        return "COMBINATION_STATE_FRAGMENTATION"

    return "GENERIC_PARENT_SURVIVAL"


def compute_domain_performance_profile(domain: str, results: list[dict]) -> dict:
    """
    Step 3 — Domain-Wise Performance Tracking.
    Track performance across specialized clinical domains.
    """
    profile = {
        "domain": domain,
        "fp_rate": 0.0,
        "fn_rate": 0.0,
        "specificity_retention": 0.0,
        "procedure_stability": 0.0,
        "hallucination_frequency": 0.0
    }
    
    if not results: return profile
    
    fps = [r for r in results if r.get("is_fp")]
    fns = [r for r in results if r.get("is_fn")]
    hallucinations = [r for r in results if "SEMANTIC_HALLUCINATION" in r.get("failure_type", "")]
    
    profile["fp_rate"] = len(fps) / len(results)
    profile["fn_rate"] = len(fns) / len(results)
    profile["hallucination_frequency"] = len(hallucinations) / len(results)
    
    # Specificity retention proxy
    spec_codes = [r for r in results if "SPECIFICITY_LOCK_GRANTED" in r.get("audit_traces", [])]
    profile["specificity_retention"] = len(spec_codes) / len(results)
    
    return profile


def compute_failure_priority_score(failure_type: str, frequency: int, severity: float) -> float:
    """
    Step 5 — Failure Frequency Prioritization.
    Score failures by clinical, billing, and hallucination impact.
    """
    impact_map = {
        "SEMANTIC_HALLUCINATION": 10.0,
        "SPECIFICITY_COLLAPSE": 7.0,
        "PROCEDURE_SUBTYPE_DOWNGRADE": 8.0,
        "COMBINATION_STATE_FRAGMENTATION": 6.0,
        "TEMPORAL_LEAKAGE": 9.0,
        "SYMPTOM_OVERCODING": 4.0,
        "GENERIC_PARENT_SURVIVAL": 3.0
    }
    
    base_impact = impact_map.get(failure_type, 5.0)
    # Priority = frequency * base_impact * severity
    return frequency * base_impact * (severity or 1.0)


# ─── Task: Final Representation Collapse & Duplicate Survival Suppression ───

def build_representation_family_index(codes: list[dict]) -> dict:
    """
    Step 1 — Build Representation Family Index.
    Group candidates by ICD prefix, CPT families, and semantic overlap.
    """
    families = {}
    
    for c in codes:
        if not c.get("code"): continue
        
        # 1. Deterministic Prefix Grouping (ICD-10)
        code_str = c.get("code", "").upper()
        pfx = code_str[:3]
        
        # 2. Procedural Family Grouping (CPT)
        if (c.get("type") or "").upper() == "CPT":
            pfx = f"CPT_{code_str[:2]}" # Group by CPT category
            
        family_id = pfx
        families.setdefault(family_id, {"members": [], "dominant": None, "manifestations": [], "generic": []})
        families[family_id]["members"].append(c)
        
        # 3. Manifestation tagging
        desc = (c.get("description") or "").lower()
        if any(m in desc for m in ["manifestation", "symptom", "due to", "resulting in"]):
            families[family_id]["manifestations"].append(c)
            
        # 4. Generic tagging
        if "NOS" in code_str or "unspecified" in desc:
            families[family_id]["generic"].append(c)
            
    # Assign Family ID back to codes
    for fid, data in families.items():
        # Trace assignment
        for m in data["members"]:
            m["REPRESENTATION_FAMILY_ID"] = fid
            m.setdefault("audit_traces", []).append("REPRESENTATION_FAMILY_INDEX_BUILT")
            
    return families


# ─── Task: Audit Decision Calibration & Conservative Review Governance ──────

def compute_audit_decision_confidence(c: dict, note_text: str) -> float:
    """
    Step 1 — Build Audit Decision Confidence Model.
    Aggregate grounding, dominance, and clinical context into a final audit score.
    """
    score = 0.0
    traces = c.get("audit_traces", [])
    
    # 1. Grounding Authority (Base)
    grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
    score += grounding * 0.40
    
    # 2. Provider Assertion / Intent
    if "PROVIDER_INTENT_CONFIRMED" in traces: score += 0.15
    if "AUTHORITATIVE_EVIDENCE_CONFIRMED" in traces: score += 0.10
    
    # 3. Family Dominance
    if "DOMINANT_REPRESENTATION_ELECTED" in traces: score += 0.15
    if "INTEGRATED_DISEASE_STATE_CONFIRMED" in traces: score += 0.10
    
    # 4. Procedure Linkage
    if "PROCEDURE_DIAGNOSIS_COHERENCE_CONFIRMED" in traces: score += 0.10
    
    # 5. Temporal Certainty
    if c.get("temporal_status") == "ACTIVE": score += 0.05
    
    final_conf = min(1.0, score)
    c["AUDIT_DECISION_CONFIDENCE_VAL"] = final_conf
    c.setdefault("audit_traces", []).append("AUDIT_DECISION_CONFIDENCE_COMPUTED")
    
    return final_conf


def compute_auditor_conservatism_weight(codes: list[dict]) -> float:
    """
    Step 5 — Auditor Conservatism Calibration.
    Bias final decisions toward defensible, high-precision findings.
    """
    # Conservatism increases if many candidates survive (noise reduction)
    count = len(codes)
    base_conservatism = 0.70
    
    if count > 15: base_conservatism = 0.85
    if count > 25: base_conservatism = 0.95
    
    # Trace conservatism
    for c in codes:
        c.setdefault("audit_traces", []).append(f"CONSERVATIVE_AUDIT_MODE_ACTIVE: {base_conservatism}")
        
    return base_conservatism


# ─── Task: Pipeline Regression Debugging & Stability Restoration ────────────

PIPELINE_DEBUG_MODE = False
SAFE_MODE = False

def validate_candidate_schema(c: dict) -> bool:
    """
    Step 3 — Build Candidate Schema Validator.
    Verify structural integrity before major processing.
    """
    if not isinstance(c, dict):
        logger.error("CANDIDATE_SCHEMA_INVALID: Not a dict")
        return False
        
    required = ["code", "confidence", "evidence_strength", "audit_traces"]
    for field in required:
        if field not in c:
            logger.error(f"CANDIDATE_SCHEMA_INVALID: Missing field {field}")
            return False
            
    # Validate types & values
    if not isinstance(c.get("audit_traces"), list):
        return False
        
    # Clamp & Normalize
    try:
        c["confidence"] = min(1.0, max(0.0, float(c.get("confidence") or 0)))
        c["evidence_strength"] = min(1.0, max(0.0, float(c.get("evidence_strength") or 0)))
    except (ValueError, TypeError):
        return False
        
    if not c.get("code"):
        return False
        
    return True


def record_pipeline_telemetry(stage: str, start_time: float, inputs: list, outputs: list, failures: list = None) -> dict:
    """
    Step 1 & 7 — Pipeline Telemetry & Timing.
    Track runtime and throughput per pipeline stage.
    """
    runtime = (time.time() - start_time) * 1000 # ms
    
    telemetry = {
        "stage": stage,
        "input_count": len(inputs) if inputs else 0,
        "output_count": len(outputs) if outputs else 0,
        "runtime_ms": runtime,
        "failures": failures or [],
        "warnings": []
    }
    
    if PIPELINE_DEBUG_MODE:
        logger.info(f"PIPELINE_TELEMETRY: Stage={stage}, In={telemetry['input_count']}, Out={telemetry['output_count']}, Time={runtime:.1f}ms")
        
    return telemetry

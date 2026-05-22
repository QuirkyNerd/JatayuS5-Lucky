"""
services/selection_engine.py – High-Precision Clinical Retrieval Selection Engine.

RESPONSIBILITIES:
  1. Executes the multi-stage code selection and hierarchy resolution pipeline.
  2. Prioritizes clinical evidence (grounding) over semantic approximation.
  3. Enforces deterministic evidence gates to prevent over-emission.
  4. Manages final code set pruning and clinical sibling discrimination.
"""

import re
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from utils.logging import get_logger
from services.clinical_rules_config import (
    COMPOUND_RULES,
    CROSS_PREFIX_SUPPRESS,
    HIERARCHY_SUPPRESSION,
    HARD_REJECT_PREFIXES,
    ALWAYS_REJECT_PREFIXES,
    RENAL_SYNDROME_PREFIXES,
    CLINICAL_EXCLUSIVITY_RULES,
    RELATIONSHIP_VALIDATION_RULES,
    ENTITY_PREFIX_MAP,
    MANDATORY_GROUPS,
    CKD_ENTITY_SIGNALS,
    DOMAIN_SPECIFIC_BOOSTS,
    DOMAIN_MERGE_RULES,
)
from services.universal_hierarchy import UniversalHierarchyEngine
from services.validation_utils import (
    extract_anatomy_regions,
    check_anatomy_consistency,
    validate_procedure_evidence,
    get_code_anatomy,
    clinical_specificity_score,
    compute_procedural_survival_score,
    apply_specificity_hierarchy,
    SECTION_WEIGHTS,
    LOW_PRIORITY_SECTIONS,
    check_cross_diagnosis_conflicts,
    clamp_score,
    ENCOUNTER_DOMAINS,
    PROCEDURE_COHERENCE_FAMILIES,
    clean_rag_description,
    normalize_clinical_terminology,
    compute_semantic_neighbor_risk,
    compute_candidate_purity_score,
    compute_semantic_saturation_risk,
    compute_encounter_domain_signature,
    calculate_soft_fusion_confidence,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_FINAL_CODES = 10 
MIN_RAG_CONFIDENCE = 0.20          
PRINCIPAL_BOOST = 0.35             
CONFIDENCE_THRESHOLD_STRICT = 0.85 

GENERIC_NOS_CODES = {
    "R52", "I10", "E11.9", "E78.5", "R06.00", "R07.9", "M54.9", "G62.9", "I50.9", "N18.9", "I82.401",
}

ACUTE_ACTIVE_PREFIXES = {
    "I21", "I22", "I63", "G45", "A41", "A40", "J96", "J80", "I50.21", "I50.23", "I50.31", "I50.33", "N17",
}

CHRONIC_BACKGROUND_PREFIXES = {
    "I10", "E78", "E11.9", "E66", "F17", "Z85", "Z86", "Z87", "Z88",
}


def _infer_primary_focus(text: str) -> set[str]:
    text_lower = text.lower()
    focus_keywords = set()
    markers = [
        "principal diagnosis", "primary diagnosis", "reason for encounter", 
        "admission for", "presents for", "chief complaint", "indication for procedure",
        "operative diagnosis", "postoperative diagnosis", "assessment and plan",
        "impression:", "final diagnosis"
    ]
    for m in markers:
        if m in text_lower:
            idx = text_lower.find(m)
            sentence = text_lower[max(0, idx-20):min(len(text_lower), idx+250)]
            for term in [
                "sepsis", "infarction", "failure", "stroke", "pneumonia", "fracture", "bypass",
                "appendicitis", "cholecystitis", "osteoarthritis", "diabetes", "hypertension",
                "renal", "pulmonary", "cardiac", "atrial", "vascular", "stenosis", "clogged",
                "hemorrhage", "bleeding", "aneurysm", "tumor", "malignancy", "cancer"
            ]:
                if term in sentence:
                    focus_keywords.add(term)
    return focus_keywords


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 Validators
# ─────────────────────────────────────────────────────────────────────────────

_ICD10_RE = re.compile(r"^[A-Z][0-9][A-Z0-9]{1,7}$|^[A-Z][0-9]{2}\.[A-Z0-9]{1,4}$", re.IGNORECASE)
_ICD9_NUMERIC_RE = re.compile(r"^\d{3,5}(\.\d{0,2})?$")
_ICD9_ECODE_RE   = re.compile(r"^E\d{3,4}(\.\d)?$", re.IGNORECASE)

def _is_valid_icd10(code: str) -> bool:
    if not code or len(code) < 3 or len(code) > 8:
        return False
    if _ICD9_NUMERIC_RE.match(code):
        return False
    if _ICD9_ECODE_RE.match(code):
        return False
    return bool(_ICD10_RE.match(code))

def _specificity(code: str) -> int:
    return len(code.replace(".", ""))

def _prefix3(code: str) -> str:
    return code.split(".")[0].upper() if "." in code else code[:3].upper()

def _auto_group(code: str, code_type: str) -> str:
    if code_type.upper() == "CPT":
        return f"cpt_{code}"
    return _prefix3(code)


# ─────────────────────────────────────────────────────────────────────────────
# _ScoredCode dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ScoredCode:
    code: str
    description: str
    code_type: str          
    group: str              
    det_score: float = 0.0
    rag_score: float = 0.0
    specificity: int = 0
    entity_score: float = 0.0
    confidence: float = 0.0
    source: str = "rag"
    rationale: str = ""
    evidence_span: str = ""
    final_score: float = 0.0
    protected: bool = False
    section_priority: int = 3
    reliability_tier: str = "Low" 
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "code": self.code,
            "description": self.description,
            "type": self.code_type,
            "confidence": round(self.final_score, 3),
            "source": self.source,
            "rationale": self.rationale,
            "evidence_span": self.evidence_span,
            "det_score": round(self.det_score, 3),
            "rag_score": round(self.rag_score, 3),
            "section_priority": self.section_priority,
            "protected": self.protected,
            "evidence_strength": round(self.final_score, 3),
            "audit_traces": self.extra.get("audit_traces", []),
        }
        d.update(self.extra)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# SelectionEngine Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

class SelectionEngine:
    """
    High-Precision Clinical Retrieval Engine.
    Implements Task 53 engineering principles: simplicity, precision, explainability.
    """

    def select(
        self,
        candidates: list[dict],
        note_text: str = "",
        deterministic_codes: Optional[list[dict]] = None,
        gold_codes: list[str] = None
    ) -> dict:
        """
        Main entry point for High-Precision selection.
        """
        logger.info("SE_START: %d candidates", len(candidates))
        if not candidates:
            return {"selected": [], "rejected": [], "gold_ranks": {}}

        note_norm = note_text.lower()
        det_set = {c.get("code", "").upper() for c in (deterministic_codes or [])}
        
        # ── Stage 1: Validation & Initialization ──
        pool = self._validate_convert(candidates, det_set, note_text)
        
        # ── Stage 2: Evidence-Dominant Ranking ──
        # Replaces complex penalty stacks with additive clinical evidence.
        pool = self._apply_evidence_scoring(pool, note_text)

        # Urology showcase: downrank symptom noise when etiology present
        try:
            from services.urology_demo_pathway import downrank_symptom_noise_in_pool
            downrank_symptom_noise_in_pool(pool, note_text)
        except ImportError:
            from urology_demo_pathway import downrank_symptom_noise_in_pool
            downrank_symptom_noise_in_pool(pool, note_text)
        
        # ── Stage 3: Deterministic Evidence Gates ──
        # Rejects codes lacking explicit clinical signals.
        rejected_candidates = []
        gate_output = self._apply_precision_gates(pool, note_norm, det_set)
        for sc in pool:
            if sc not in gate_output:
                rejected_candidates.append({
                    "code": sc.code,
                    "description": sc.description,
                    "reason": sc.rationale if "[REJECTED:" in sc.rationale else "Gate rejection",
                    "score": sc.final_score,
                    "stage": "Precision Gating"
                })
        pool = gate_output
        
        # ── Stage 4: Clinical Sibling Discrimination ──
        # Picks the most specific grounded representative in a family.
        sib_output = self._apply_sibling_discrimination(pool, note_text)
        for sc in pool:
            if sc not in sib_output:
                rejected_candidates.append({
                    "code": sc.code,
                    "description": sc.description,
                    "reason": "Pruned by sibling specificity discrimination",
                    "score": sc.final_score,
                    "stage": "Sibling Discrimination"
                })
        pool = sib_output
        
        # ── Stage 5: Domain-Specific Mergers ──
        pool = self._apply_domain_merger_rules(pool, note_norm)
        
        # ── Final Emission & Audit Trail ──
        pool.sort(key=lambda x: x.final_score, reverse=True)
        final_pool = pool[:MAX_FINAL_CODES]
        
        for sc in pool:
            if sc not in final_pool:
                rejected_candidates.append({
                    "code": sc.code,
                    "description": sc.description,
                    "reason": "Rank tail trimming (Top-10 only)",
                    "score": sc.final_score,
                    "stage": "Final Selection"
                })

        # Forensic Logging
        self._log_forensics(pool, final_pool, gold_codes)

        return {
            "selected": [sc.as_dict() for sc in final_pool],
            "rejected": rejected_candidates, 
            "gold_ranks": {},
            "audit_trail": {
                "logic_applied": ["Evidence Gating", "Sibling Discrimination", "Interaction Merger", "Escalation Control"],
                "transparency_status": "HIGH_CONFIDENCE_GROUNDED" if final_pool and final_pool[0].final_score > 0.85 else "PARTIAL_MATCH"
            }
        }

    def _validate_convert(self, candidates: list[dict], det_set: set[str], note_text: str) -> list[_ScoredCode]:
        result: list[_ScoredCode] = []
        seen = set()
        for c in candidates:
            code = c.get("code", "").strip().upper()
            if not code or code in seen: continue
            seen.add(code)

            ctype = c.get("type", "ICD-10").upper()
            if ctype == "ICD": ctype = "ICD-10"
            if ctype != "CPT" and not _is_valid_icd10(code): continue

            sc = _ScoredCode(
                code=code,
                description=c.get("description", ""),
                code_type=ctype,
                group=_auto_group(code, ctype),
                det_score=float(c.get("det_score", 0.0)),
                rag_score=float(c.get("rag_score", 0.75)),
                specificity=_specificity(code),
                source=c.get("source", "rag"),
                section_priority=int(c.get("section_priority", 3)),
                extra=c.copy()
            )
            if code in det_set: sc.protected = True
            result.append(sc)
        return result

    def _apply_evidence_scoring(self, pool: list[_ScoredCode], note_text: str) -> list[_ScoredCode]:
        """
        TASK 53: EVIDENCE-DOMINANT RANKING.
        Final Score = Evidence(0.38) + Section(0.22) + Anatomy(0.15) + Procedure(0.10) + Specificity(0.10) - Penalty(0.08)
        """
        note_lower = note_text.lower()
        note_anatomy = extract_anatomy_regions(note_lower)
        active_cpts = {s.code for s in pool if s.code_type == "CPT"}

        # Coherence Graphs
        DX_PROC_LINKS = {"I25": ["33533"], "M16": ["27130"], "M17": ["27447"], "S72": ["27244", "27245"]}

        for sc in pool:
            # 1. Component Extraction
            semantic = sc.rag_score
            
            # Use aggregated features from extra if present, otherwise fallback to exact-match logic
            terminology = sc.extra.get("terminology_overlap")
            if terminology is None:
                terminology = 1.0 if sc.description.lower() in note_lower else 0.0
            
            # Evidence Component (0.38)
            evidence = 0.6 * terminology + 0.4 * semantic
            
            # Section Component (0.22)
            section = min(sc.section_priority / 10.0, 1.0)
            
            # Anatomy Component (0.15)
            anatomy = sc.extra.get("anatomy_overlap")
            if anatomy is None:
                code_anat = get_code_anatomy(sc.code, sc.description)
                anatomy = 1.0 if (code_anat and note_anatomy and (code_anat & note_anatomy)) else 0.0
            
            # Procedure Component (0.10)
            procedure = sc.extra.get("procedure_linkage")
            if procedure is None:
                procedure = 0.0
                for dx_pfx, procs in DX_PROC_LINKS.items():
                    if sc.code.startswith(dx_pfx) and any(p in active_cpts for p in procs):
                        procedure = 1.0
                        break
            
            # Soft cap on procedure linkage contribution to prevent CPT linkage dominance
            procedure = min(procedure, 0.25)
            
            # Specificity Component (0.10)
            spec_val = min(sc.specificity / 8.0, 1.0)
            
            # 2. Additive Score
            sc.final_score = (
                0.38 * evidence +
                0.22 * section +
                0.15 * anatomy +
                0.10 * procedure +
                0.10 * spec_val
            )
            
            # 3. Micro Penalties (-0.08 max)
            if "unspecified" in sc.description.lower() or "nos" in sc.description.lower():
                sc.final_score -= 0.05
                # TASK 87: Severe Unspecified Suppression
                if any(sc.code.startswith(pfx) for pfx in ["A41", "I21", "N17", "R57", "I50"]):
                    sc.final_score -= 0.15 # Massive penalty for severe unspecified
                    
            if sc.code in GENERIC_NOS_CODES and not sc.protected:
                sc.final_score -= 0.03
            
            sc.final_score = round(max(0.0, min(0.99, sc.final_score)), 3)
            
            # 4. Domain-Specific Targeted Boosts (TASK 85)
            self._apply_domain_specific_boosts(sc, note_lower)
            
            sc.extra["scoring_breakdown"] = {
                "evidence": evidence, "section": section, "anatomy": anatomy, 
                "procedure": procedure, "specificity": spec_val
            }
        return pool

    def _apply_domain_specific_boosts(self, sc: _ScoredCode, note_lower: str):
        """
        TASK 85: Targeted Domain Weakness Optimization.
        Boosts codes based on high-confidence domain markers.
        """
        for domain, config in DOMAIN_SPECIFIC_BOOSTS.items():
            if any(sc.code.startswith(pfx) for pfx in config["prefixes"]):
                # Check for triggers
                if any(trig in note_lower for trig in config["triggers"]):
                    sc.final_score += config["boost_amount"]
                    sc.rationale += f" [{domain.upper()}_DOM_BOOST]"
                    
                # Check for laterality (Orthopedics)
                if config.get("laterality_required"):
                    if "left" in note_lower and "left" in sc.description.lower():
                        sc.final_score += 0.10
                    elif "right" in note_lower and "right" in sc.description.lower():
                        sc.final_score += 0.10
                
                sc.final_score = round(max(0.0, min(0.99, sc.final_score)), 3)
                break

    def _apply_precision_gates(self, pool: list[_ScoredCode], note_norm: str, det_set: set[str]) -> list[_ScoredCode]:
        """
        TASK 87/89: Precision Gating with Explainable Rejections.
        """
        result: list[_ScoredCode] = []
        for sc in pool:
            # Rejection Reason tracking (internal use during this loop)
            rejection_reason = None
            grounded = True

            # Gate 1: Precision Barrier
            # If not in det_set and score too low, reject
            if sc.code not in det_set and sc.final_score < 0.65 and not sc.protected:
                grounded = False
                rejection_reason = f"Insufficient grounding ({sc.final_score} < 0.65)"
            
            # Gate 2: Negation Check
            if grounded and self.is_negated(sc.description, note_norm):
                grounded = False
                rejection_reason = "Negation detected (e.g. 'no evidence of')"

            # Gate 2b: Urology showcase — suppress weak symptoms when etiology present
            if grounded:
                try:
                    from services.urology_demo_pathway import (
                        has_strong_urology_etiology,
                        is_symptom_noise_code,
                        is_urology_showcase_note,
                    )
                except ImportError:
                    from urology_demo_pathway import (
                        has_strong_urology_etiology,
                        is_symptom_noise_code,
                        is_urology_showcase_note,
                    )
                if is_urology_showcase_note(note_norm) and is_symptom_noise_code(sc.code):
                    etiology_pool = [{"code": x.code} for x in pool if x is not sc]
                    if has_strong_urology_etiology(etiology_pool, note_norm):
                        grounded = False
                        rejection_reason = "Symptom downranked — urology etiology present (R11/R52)"
                
            # Gate 3: High-Risk Condition Hardening (TASK 87)
            if grounded:
                risk_config = [
                    {"prefix": "A41", "markers": ["shock", "sirs", "qsofa", "vasopressor", "hypotension", "organ failure", "sepsis"]},
                    {"prefix": "A40", "markers": ["shock", "sirs", "qsofa", "vasopressor", "hypotension", "organ failure", "sepsis"]},
                    {"prefix": "I21", "markers": ["stemi", "nstemi", "infarction", "troponin", "st-segment", "acute myocardial"]},
                    {"prefix": "N17", "markers": ["aki", "acute kidney injury", "acute renal failure", "cr elevation", "creatinine elevation"]},
                    {"prefix": "R57", "markers": ["shock", "hypoperfusion", "cardiogenic", "septic", "hypovolemic"]},
                    {"prefix": "I50.21", "markers": ["acute", "decompensated", "systolic", "exacerbation"]},
                    {"prefix": "I50.31", "markers": ["acute", "decompensated", "diastolic", "exacerbation"]},
                ]
                
                for risk in risk_config:
                    if sc.code.startswith(risk["prefix"]):
                        if not any(m in note_norm for m in risk["markers"]) and sc.final_score < 0.90:
                            grounded = False
                            rejection_reason = f"High-risk condition missing required clinical markers ({', '.join(risk['markers'][:2])}...)"
                            break
            
            if grounded:
                result.append(sc)
            else:
                # Store rejection reason in rationale for audit trail
                sc.rationale = f"[REJECTED: {rejection_reason}] {sc.rationale}"

        return result

    def _apply_domain_merger_rules(self, pool: list[_ScoredCode], note_norm: str) -> list[_ScoredCode]:
        """
        TASK 88: Merges separate codes into domain combination codes (e.g. HTN+HF).
        Enforces DUAL-EVIDENCE for high-risk promotions.
        """
        code_set = {sc.code[:3] for sc in pool}
        to_remove = set()
        
        # Risk markers for escalation control (TASK 88)
        HIGH_RISK_TARGETS = {
            "I21": ["stemi", "nstemi", "infarction", "acute myocardial", "troponin"],
            "N17": ["aki", "acute kidney injury", "acute renal failure", "cr elevation"],
            "A41": ["shock", "sepsis", "septic", "organ failure"],
            "I50": ["acute", "decompensated", "exacerbation"]
        }
        
        for rule in DOMAIN_MERGE_RULES:
            if all(m in code_set for m in rule["members"]):
                target_prefix = rule["target"][:3]
                
                # Check for direct supporting evidence for high-risk targets
                has_direct_evidence = True
                if target_prefix in HIGH_RISK_TARGETS:
                    markers = HIGH_RISK_TARGETS[target_prefix]
                    has_direct_evidence = any(m in note_norm for m in markers)
                
                # Target promotion (Conditional on evidence)
                for sc in pool:
                    if sc.code.startswith(rule["target"]):
                        if has_direct_evidence:
                            sc.final_score = max(sc.final_score, 0.88)
                            sc.rationale += f" [MERGE_TARGET_PROMOTED:{rule['id']}]"
                            sc.protected = True
                        else:
                            # Prevent escalation: limit score if missing direct evidence
                            sc.final_score = min(sc.final_score, 0.60)
                            sc.rationale += " [ESCALATION_CONTROLLED:MISSING_DIRECT_EVIDENCE]"
                
                # Member suppression vs protection
                if not rule.get("protect_members"):
                    for m in rule["members"]:
                        # Only suppress if target is actually in pool and strong
                        if any(sc.code.startswith(rule["target"]) and sc.final_score > 0.80 for sc in pool):
                            to_remove.add(m)
                else:
                    # Protection: ensure members survive too
                    for sc in pool:
                        if any(sc.code.startswith(m) for m in rule["members"]):
                            sc.protected = True
                            # Boost member score slightly to ensure survival
                            sc.final_score = max(sc.final_score, 0.82)
        
        if not to_remove: return pool
        return [sc for sc in pool if sc.code[:3] not in to_remove]

    def _apply_sibling_discrimination(self, pool: list[_ScoredCode], note_text: str) -> list[_ScoredCode]:
        """
        Pick the strongest grounded sibling in a clinical family.
        """
        note_lower = note_text.lower()
        families: dict[str, list[_ScoredCode]] = {}
        for sc in pool:
            if sc.code_type != "ICD-10": continue
            fid = sc.code[:3]
            if fid.startswith("S"): fid = sc.code[:5] # Fracture precision
            families.setdefault(fid, []).append(sc)

        to_remove = set()
        for fid, members in families.items():
            if len(members) <= 1: continue
            
            # Winner selection: Specificity + Laterality + Evidence
            SPECIFICITY_MARKERS = [
                "displaced", "nondisplaced", "bilateral", "stage 3", "stage 4", "stage 5", 
                "septic shock", "decompensated", "acute on chronic", "exacerbation"
            ]

            def sibling_rank(s: _ScoredCode):
                bonus = 0.0
                desc = s.description.lower()
                # 1. Laterality Alignment
                if "left" in note_lower and "left" in desc: bonus += 0.2
                if "right" in note_lower and "right" in desc: bonus += 0.2
                
                # 2. Specificity Recovery (TASK 90)
                if any(marker in desc and marker in note_lower for marker in SPECIFICITY_MARKERS):
                    bonus += 0.35 # Strong recovery bonus
                    s.rationale += " [SPECIFICITY_RECOVERED]"
                
                return (s.protected, s.final_score + bonus, s.specificity)

            members.sort(key=sibling_rank, reverse=True)
            winner = members[0]
            for m in members[1:]:
                if not m.protected:
                    to_remove.add(m.code)
                    logger.debug("SE_SIBLING_PRUNE: %s (winner: %s)", m.code, winner.code)
        
        return [s for s in pool if s.code not in to_remove]

    def is_negated(self, keyword: str, text: str) -> bool:
        NEGATIONS = ["no", "not", "without", "denies", "denied", "negative for", "ruled out", "exclude"]
        text_lower = text.lower()
        term_lower = keyword.lower()
        positions = [m.start() for m in re.finditer(rf"\b{re.escape(term_lower)}\b", text_lower)]
        if not positions: return False
        
        any_positive = False
        for pos in positions:
            pre_window = text_lower[max(0, pos-40):pos]
            if not any(neg in pre_window for neg in NEGATIONS):
                any_positive = True
                break
        return not any_positive

    def _log_forensics(self, pool: list[_ScoredCode], final: list[_ScoredCode], gold: list[str]):
        final_codes = {f.code for f in final}
        logger.info("POOL_MRR_TRACE: Final Emission Count: %d", len(final))
        if gold:
            for gc in gold:
                match = next((s for s in pool if s.code == gc), None)
                if match:
                    status = "HIT" if gc in final_codes else "MISS"
                    logger.info("GOLD_RANK_FORENSIC: code=%s | status=%s | score=%.3f | breakdown=%s", 
                                gc, status, match.final_score, match.extra.get("scoring_breakdown"))
                else:
                    logger.info("GOLD_RANK_FORENSIC: code=%s | status=NOT_RETRIEVED", gc)

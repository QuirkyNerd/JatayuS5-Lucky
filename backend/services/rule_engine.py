"""
services/rule_engine.py – Deterministic Clinical Coding Rule Engine.

RESPONSIBILITIES:
  1. Executes the 5-stage deterministic coding rule-set.
  2. Enforces ICD-10 hierarchy upgrades and compound code injections.
  3. Validates CPT-to-ICD coherence and mutual exclusivity.
  4. Manages clinical specificity preservation via declarative rule-sets.
"""

import copy
import re
import logging
from typing import Optional

from utils.logging import get_logger
from services.clinical_rules_config import (
    COMPOUND_RULES,
    CROSS_PREFIX_SUPPRESS,
    HIERARCHY_SUPPRESSION,
    MANDATORY_GROUPS,
    ENTITY_PREFIX_MAP,
)
from services.validation_utils import (
    is_negated,
    has_prophylaxis_context,
    compute_evidence_strength,
    apply_specificity_hierarchy,
    check_anatomy_consistency,
    validate_procedure_evidence,
    clinical_specificity_score,
    SECTION_WEIGHTS,
    LOW_PRIORITY_SECTIONS,
    check_cross_diagnosis_conflicts,
    ENCOUNTER_DOMAINS,
    PROCEDURE_COHERENCE_FAMILIES,
    is_generic_parent,
    clean_rag_description,
)
from services.clinical_reasoning_engine import ClinicalReasoningEngine as _CRE
from services.final_validator import run_final_validation as _run_final_validation

logger = get_logger(__name__)

# Singleton — avoid re-instantiating on every call
_cre_instance = _CRE() if _CRE is not None else None


# ---------------------------------------------------------------------------
# Clinical indicator words — used by evidence and rule validation
# ---------------------------------------------------------------------------
_CLINICAL_INDICATORS = frozenset([
    "diagnosed", "diagnosis", "history", "presents", "documented", "noted",
    "confirmed", "complains", "exhibits", "demonstrates", "positive",
    "elevated", "decreased", "chronic", "acute", "severe", "mild", "moderate",
    "nephropathy", "neuropathy", "retinopathy", "cardiomyopathy",
    "insufficiency", "failure", "infection", "sepsis", "pneumonia",
    "diabetes", "hypertension", "ckd", "esrd", "fever", "pain",
    "hba1c", "creatinine", "wbc", "blood", "culture", "organism",
    "e. coli", "escherichia", "staphylococcus", "streptococcus",
    "klebsiella", "pseudomonas",
])

# Symptoms integral to a diagnosis — should not be coded separately
# Format: {symptom_code: [diagnosing_codes_that_include_it]}
_INTEGRAL_SYMPTOMS: dict[str, list[str]] = {
    "R50.9":  ["A41.9", "A41.51", "A41.89", "A40.9", "A40.1"],   # Fever in sepsis
    "R50.81": ["A41.9", "A41.51", "A41.89"],                       # Fever in sepsis
    "R05.9":  ["J18.9", "J15.0", "J15.1", "J15.4", "J15.6"],      # Cough in pneumonia
    "R06.00": ["J18.9", "J15.0", "J96.0", "J96.9"],               # Dyspnea in pneumonia/resp fail
    "R09.02": ["J96.0", "J96.9", "J80"],                           # Hypoxemia in resp failure
    "R73.09": ["E11.9", "E11.21", "E11.22", "E11.40", "E11.42",
               "E11.65"],                                           # Hyperglycemia in DM
    "R41.3":  ["G30.9", "G31.9", "F01.51", "F03.90"],              # Memory impairment in dementia
    "R52":    ["S", "M", "I", "J"],                                 # Generic pain suppressed by any S/M/I/J code (Precision v15)
    "R10.9":  ["K", "N"],                                           # Abdominal pain suppressed by GI/Renal diagnosis
    "R07.9":  ["I20", "I21", "I25", "I50"],                         # Chest pain suppressed by Cardiac diagnosis
}

# Mutually exclusive code pairs (keep higher specificity — second in pair wins)
_MUTEX_PAIRS: list[tuple[str, str]] = [
    ("E11.9",  "E11.21"),   # Unspecified DM vs DM with nephropathy
    ("E11.9",  "E11.22"),   # Unspecified DM vs DM with CKD
    ("E11.9",  "E11.40"),   # Unspecified DM vs DM with neuropathy
    ("E11.9",  "E11.42"),   # Unspecified DM vs DM with polyneuropathy
    ("E11.9",  "E11.65"),   # Unspecified DM vs DM with hyperglycemia
    ("E11.9",  "E11.319"),  # Unspecified DM vs DM with NPDR
    ("E11.9",  "E11.339"),  # Unspecified DM vs DM with PDR
    ("E10.9",  "E10.40"),
    ("E10.9",  "E10.42"),
    ("A41.9",  "A41.51"),   # Unspecified sepsis vs E-coli sepsis
    ("A41.9",  "A41.01"),   # Unspecified sepsis vs MRSA sepsis
    ("A41.9",  "A41.1"),    # Unspecified sepsis vs Strep sepsis
    ("I50.9",  "I50.21"),   # Unspecified HF vs acute systolic HF
    ("I50.9",  "I50.22"),   # Unspecified HF vs chronic systolic HF
    ("I50.9",  "I50.23"),   # Unspecified HF vs acute-on-chronic systolic HF
    ("I50.9",  "I50.31"),   # Unspecified HF vs acute diastolic HF
    ("I50.9",  "I50.32"),   # Unspecified HF vs chronic diastolic HF
    ("I50.9",  "I50.33"),   # Unspecified HF vs acute-on-chronic diastolic HF
    ("N18.9",  "N18.1"),
    ("N18.9",  "N18.2"),
    ("N18.9",  "N18.3"),
    ("N18.9",  "N18.4"),
    ("N18.9",  "N18.5"),
    ("N18.9",  "N18.6"),
    ("B99.9",  "A41.9"),   # Other infectious disease superseded by specific infection
    ("B99.9",  "A41.51"),
    ("B99.9",  "A40.9"),
    ("B99.9",  "J15.0"),
    ("B99.9",  "J18.9"),
]

# CPT codes that must not appear in output — mapped to their correct replacements
_CPT_BLOCK: dict[str, str] = {
    "86900": "87040",   # ABO typing → Blood culture (clinically incorrect mapping)
    "86901": "87040",   # Rh typing → Blood culture
    "86902": "87040",
    "87205": "87070",   # Gram stain only → culture preferred
}

# Blood culture CPT
_CPT_BLOOD_CULTURE = "87040"
_CPT_BLOOD_CULTURE_TRIGGERS = frozenset([
    "blood culture", "bacteremia", "sepsis", "septicemia", "blood cx",
    "cultures drawn", "blood drawn for culture",
])

# Central line CPT
_CPT_CENTRAL_LINE_TUNNELED = "36558"
_CPT_CENTRAL_LINE_NON_TUNNELED = "36556"
_CPT_TUNNELED_TRIGGERS = frozenset(["tunneled", "tunnelled", "hickman", "broviac", "groshong"])
_CPT_NON_TUNNELED_TRIGGERS = frozenset([
    "femoral", "icu", "critical care", "intensive care",
    "non-tunneled", "nontunneled", "temporary", "short-term",
    "triple lumen", "quad lumen", "double lumen",
])


class RuleEngine:

    # ------------------------------------------------------------------
    # STAGE 1 (unchanged): inject deterministic codes
    # ------------------------------------------------------------------
    @staticmethod
    def inject_deterministic_codes(
        existing_codes: list[dict],
        deterministic_codes: list[dict],
    ) -> list[dict]:
        """
        Merge deterministic codes into the existing AI code list.
        Deterministic codes ALWAYS win — never dropped by confidence filtering.
        """
        merged = list(existing_codes)
        existing_code_strs = {c.get("code", "").upper() for c in merged}

        injected = 0
        for det_code in deterministic_codes:
            code_str = det_code.get("code", "").upper()
            if code_str and code_str not in existing_code_strs:
                injected_entry = {
                    **det_code,
                    "source": "deterministic",
                    "confidence": max(det_code.get("confidence", 0.95), 0.90),
                }
                merged.append(injected_entry)
                existing_code_strs.add(code_str)
                injected += 1

        logger.info("RuleEngine: injected %d deterministic codes into pool.", injected)
        return merged

    # ------------------------------------------------------------------
    # STAGE 2 (unchanged): ICD hierarchy upgrade rules
    # ------------------------------------------------------------------
    @staticmethod
    def apply_hierarchy_rules(clinical_facts_str: str, ai_codes: list[dict]) -> list[dict]:
        """
        Apply ICD hierarchy upgrade rules and SECONDARY REINFORCEMENT (Task 2).
        """
        processed = copy.deepcopy(ai_codes)
        facts_lower = clinical_facts_str.lower()
        code_set = {c.get("code", "").upper() for c in processed}
        
        # ── HELPER: Identify Strongly Grounded Primary Domains ──────────
        primary_grounding = {
            "DM": any(c.get("code", "").startswith("E11") and c.get("confidence", 0) >= 0.90 for c in processed),
            "CKD": any(c.get("code", "").startswith("N18") and c.get("confidence", 0) >= 0.90 for c in processed),
            "HF": any(c.get("code", "").startswith("I50") and c.get("confidence", 0) >= 0.90 for c in processed),
            "HTN": any(c.get("code", "").startswith("I10") and c.get("confidence", 0) >= 0.90 for c in processed),
        }

        # Rule 1: DM2 + Neuropathy → E11.40
        has_dm2 = any(kw in facts_lower for kw in ["diabetes mellitus type 2", "t2dm", "dm2"])
        has_neuropathy = any(kw in facts_lower for kw in ["neuropathy", "neuropathic", "nerve pain", "burning in feet"])
        if has_dm2 and has_neuropathy:
            for c in processed:
                if c.get("code", "").upper() in ("E11.9", "E119"):
                    c["code"] = "E11.40"
                    c["description"] = "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"
                    c["confidence"] = max(c.get("confidence", 0), 0.92)
                    c["rationale"] = (c.get("rationale", "") + " [RULE: DM2+Neuropathy upgrade]")
                    logger.info("RuleEngine: upgraded E11.9 -> E11.40")

        # Rule 2: DM2 + CKD → E11.22
        has_ckd = any(kw in facts_lower for kw in ["chronic kidney disease", "ckd", "renal disease", "gfr"])
        if has_dm2 and has_ckd:
            for c in processed:
                if c.get("code", "").upper() in ("E11.9", "E119"):
                    c["code"] = "E11.22"
                    c["description"] = "Type 2 diabetes mellitus with diabetic CKD"
                    c["confidence"] = max(c.get("confidence", 0), 0.92)
                    c["rationale"] = (c.get("rationale", "") + " [RULE: DM2+CKD upgrade]")

        # Rule 2.5: HTN + CKD → I12.9 (Task 6)
        has_htn = any(kw in facts_lower for kw in ["hypertension", "htn", "high blood pressure"])
        if has_htn and has_ckd:
            found_htn = False
            for c in processed:
                if c.get("code", "").upper() == "I10":
                    c["code"] = "I12.9"
                    c["description"] = "Hypertensive chronic kidney disease with stage 1-4 CKD"
                    c["confidence"] = max(c.get("confidence", 0), 0.92)
                    c["rationale"] = (c.get("rationale", "") + " [RULE: HTN+CKD combination upgrade]")
                    found_htn = True
            
            # If I12.9 was created or already exists, suppress generic CKD N18.9 if present
            if found_htn or "I12.9" in code_set:
                for c in processed:
                    if c.get("code", "").upper() == "N18.9":
                        c["confidence"] = 0.40 # Penalize instead of hard remove to allow other stages to win

        # Rule 2.7: DM2 + Retinopathy → E11.319 (Task 6)
        has_retinopathy = any(kw in facts_lower for kw in ["retinopathy", "macular edema", "retinal hemorrhage"])
        if has_dm2 and has_retinopathy:
            for c in processed:
                if c.get("code", "").upper() in ("E11.9", "E119"):
                    c["code"] = "E11.319"
                    c["description"] = "Type 2 diabetes mellitus with unspecified diabetic retinopathy"
                    c["confidence"] = max(c.get("confidence", 0), 0.92)
                    c["rationale"] = (c.get("rationale", "") + " [RULE: DM2+Retinopathy upgrade]")

        # ── SECONDARY REINFORCEMENT (Task 2) ─────────────────────────────
        for c in processed:
            code = c.get("code", "").upper()
            
            # Reinforce Diabetic Complications if DM is strong
            if primary_grounding["DM"] and (code.startswith("E11.2") or code.startswith("E11.3") or code.startswith("E11.4")):
                if c.get("confidence", 0) < 0.90:
                    c["confidence"] = min(0.95, c["confidence"] + 0.15)
                    c["rationale"] += " [REINFORCED: Primary DM strongly grounded]"
            
            # Reinforce CKD stages if CKD is strong
            if primary_grounding["CKD"] and code.startswith("N18."):
                 if c.get("confidence", 0) < 0.90:
                    c["confidence"] = min(0.95, c["confidence"] + 0.10)
                    c["rationale"] += " [REINFORCED: Primary CKD strongly grounded]"
                    
            # Reinforce HF manifestations if HF is strong
            if primary_grounding["HF"] and code.startswith("I50."):
                 if c.get("confidence", 0) < 0.90:
                    c["confidence"] = min(0.95, c["confidence"] + 0.10)
                    c["rationale"] += " [REINFORCED: Primary HF strongly grounded]"

        # Rule 3: CPT without ICD → lower confidence, flag
        has_cpt = any(c.get("type", "").upper() == "CPT" for c in processed)
        has_icd = any(c.get("type", "").upper() in ("ICD-10", "ICD-10-CM") for c in processed)
        if has_cpt and not has_icd:
            for c in processed:
                if c.get("type", "").upper() == "CPT":
                    c["rationale"] = (
                        "[WARNING: Procedure billed without primary ICD-10 diagnosis] "
                        + c.get("rationale", "")
                    )
                    c["confidence"] = min(0.65, c.get("confidence", 1.0))
                    logger.warning("RuleEngine: CPT without ICD flagged for %s", c.get("code"))

        # Rule 4: Sepsis + Pneumonia → inject J18.9 if absent
        has_sepsis = any(c.get("code", "").upper() in ("A41.9",) for c in processed)
        has_pneumonia_fact = "pneumonia" in facts_lower
        if has_sepsis and has_pneumonia_fact:
            existing_codes = {c.get("code", "").upper() for c in processed}
            if "J18.9" not in existing_codes:
                processed.append({
                    "code": "J18.9",
                    "description": "Pneumonia, unspecified organism",
                    "type": "ICD-10",
                    "confidence": 0.92,
                    "source": "rule_injection",
                    "rationale": "[RULE: Sepsis with documented pneumonia — J18.9 injected]",
                    "det_score": 0.92, "rag_score": 0.0, "llm_score": 0.0,
                })
                logger.info("RuleEngine: injected J18.9 (sepsis+pneumonia rule)")

        return processed

    # ------------------------------------------------------------------
    # STAGE 3 (NEW): Clinical rules — diabetes hierarchy, symptom
    #                exclusion, infection specificity, redundancy
    # ------------------------------------------------------------------
    @staticmethod
    def apply_clinical_rules(ai_codes: list[dict], note_text: str) -> list[dict]:
        """
        Enforce clinical coding guidelines AFTER code generation.

        Rules applied (in order):
          1. Diabetes hierarchy: E11.21 present → remove E11.9; remove E11.65
             unless "hyperglycemia" is documented alongside the complication.
          2. Symptom exclusion: remove symptoms integral to a confirmed diagnosis.
          3. Infection specificity: A41.9 → A41.51 when E. coli documented.
          4. Redundancy: remove B99.9 when specific infection code present.
          5. Mutex conflict resolver: drop generic when specific exists.

        Returns a NEW list. Caller must reassign:
            ai_codes = RuleEngine.apply_clinical_rules(ai_codes, note_text)
        """
        if not ai_codes:
            return ai_codes

        # ── PRE-FILTER: Clinical Reasoning Engine (prophylaxis + negation + evidence) ──
        # This is the primary hallucination guard. It runs BEFORE clinical rules
        # so that spurious codes (e.g. DVT from prophylaxis context) are eliminated
        # before expensive hierarchy/mutex logic runs on them.
        if _cre_instance is not None and note_text:
            print(f"DEBUG RE: before cre: {ai_codes}")
            ai_codes = _cre_instance.validate_codes(ai_codes, note_text)
            print(f"DEBUG RE: after cre: {ai_codes}")
            logger.info(
                "RuleEngine[clinical]: ClinicalReasoningEngine passed %d codes to clinical rules",
                len(ai_codes),
            )

        processed = copy.deepcopy(ai_codes)
        note_lower = note_text.lower() if note_text else ""
        code_set = {c.get("code", "").upper() for c in processed}
        to_remove: set[str] = set()

        # ── Rule 1: Diabetes hierarchy ──────────────────────────────────
        has_nephropathy = "E11.21" in code_set or "E11.22" in code_set
        has_neuropathy  = any(c in code_set for c in ("E11.40", "E11.42", "E11.641"))
        has_retinopathy = any(c in code_set for c in ("E11.319", "E11.329", "E11.339", "E11.349"))

        has_complication = has_nephropathy or has_neuropathy or has_retinopathy

        if has_complication and "E11.9" in code_set:
            to_remove.add("E11.9")
            logger.info("RuleEngine[clinical]: E11.9 removed — complication code present (%s)", code_set & {"E11.21","E11.22","E11.40","E11.42"})

        # Remove E11.65 (hyperglycemia) unless hyperglycemia is independently documented
        if has_complication and "E11.65" in code_set:
            hyperglycemia_documented = any(kw in note_lower for kw in [
                "hyperglycemia", "blood glucose", "blood sugar elevated",
                "glucose elevated", "uncontrolled glucose",
            ])
            if not hyperglycemia_documented:
                to_remove.add("E11.65")
                logger.info("RuleEngine[clinical]: E11.65 removed — not independently documented")

        # ── Rule 2: Symptom exclusion (Safe Application) ──────────────
        for symptom_code, diagnosing_codes in _INTEGRAL_SYMPTOMS.items():
            if symptom_code in code_set:
                # Check for independent documentation (e.g., "fever of unknown origin", "independent fever")
                independently_documented = False
                if symptom_code in ("R50.9", "R50.81"):
                    independently_documented = any(
                        kw in note_lower for kw in ["fever of unknown origin", "fuo", "fever secondary to"]
                    )
                
                if not independently_documented:
                    # Generic Prefix Match (Precision v15)
                    for diag in diagnosing_codes:
                        # Exact match OR prefix match (e.g. "S" prefix covers all injury codes)
                        if any(c.startswith(diag) for c in code_set):
                            to_remove.add(symptom_code)
                            logger.info(
                                "RuleEngine[clinical]: %s removed — integral to %s family",
                                symptom_code, diag,
                            )
                            break
                else:
                    logger.info("RuleEngine[clinical]: %s retained — independently documented", symptom_code)

        # ── Rule 3: Infection specificity upgrade ──────────────────────
        ecoli_present = any(kw in note_lower for kw in [
            "e. coli", "e.coli", "escherichia coli", "escherichia col",
            "gram-negative rod", "gram negative rod",
        ])
        if "A41.9" in code_set and ecoli_present and "A41.51" not in code_set:
            for c in processed:
                if c.get("code", "").upper() == "A41.9":
                    c["code"] = "A41.51"
                    c["description"] = "Sepsis due to Escherichia coli"
                    c["rationale"] = (
                        (c.get("rationale") or "") +
                        " [RULE: Upgraded A41.9 -> A41.51 — E. coli documented in note]"
                    )
                    logger.info("RuleEngine[clinical]: upgraded A41.9 -> A41.51 (E. coli present)")
            # Rebuild code_set after upgrade
            code_set = {c.get("code", "").upper() for c in processed}

        # ── Rule 4: Redundancy — remove B99.9 when specific infection present ──
        specific_infection = any(
            c.startswith("A") or c.startswith("B") and c != "B99.9"
            for c in code_set if c != "B99.9"
        )
        if "B99.9" in code_set and specific_infection:
            to_remove.add("B99.9")
            logger.info("RuleEngine[clinical]: B99.9 removed — specific infection code present")

        # ── Rule 5: Clinical Specificity Guard (Generalized — Task 6) ─────────
        # Automatically remove generic parents if a specific child exists in the pool.
        # This replaces the hardcoded _MUTEX_PAIRS with a generalized approach.
        families: dict[str, list[str]] = {}
        for c in code_set:
            pfx = c[:3]
            if pfx not in families:
                families[pfx] = []
            families[pfx].append(c)

        for pfx, sibling_codes in families.items():
            if len(sibling_codes) <= 1:
                continue
            
            # Specificity detection: if we have codes with length > 4 or more dots/digits, 
            # and we have codes ending in .9 or .0, the .9/.0 are suspect.
            specifics = [c for c in sibling_codes if not (c.endswith(".9") or c.endswith(".0") or len(c) == 3)]
            if specifics:
                for c in sibling_codes:
                    if c.endswith(".9") or c.endswith(".0") or len(c) == 3:
                        to_remove.add(c)
                        logger.info("RuleEngine[specificity]: removing generic %s in favor of specific siblings %s", c, specifics)

        # ── Rule 7: Refined Symptom Suppression (Task 10.3) ────────────────
        # Suppress generic R-codes (Symptoms) if definitive A-Q codes are present,
        # BUT only if the symptom is low-confidence or lacks direct textual support.
        has_definitive = any(c[0] >= 'A' and c[0] <= 'Q' for c in code_set)
        if has_definitive:
            # Only suppress generic R-codes (length <= 5) that are not high-confidence
            r_codes_to_check = [c for c in processed if c.get("code", "").startswith("R") and len(c.get("code", "")) <= 5]
            for r_code_dict in r_codes_to_check:
                r_code = r_code_dict.get("code", "").upper()
                strength = float(r_code_dict.get("evidence_strength") or 0.5)
                is_protected = r_code_dict.get("protected") or r_code_dict.get("source") == "deterministic"
                
                # Keep if protected, high-strength, or explicitly confirmed
                if is_protected or strength >= 0.78:
                    continue
                
                to_remove.add(r_code)
                logger.info("RuleEngine[clinical]: Suppressing weak symptom code %s (strength %.2f) due to presence of definitive diagnoses", r_code, strength)

        # ── Rule 8: Clinical Believability Trim (Task 10.3) ────────────────
        # Professional audits rarely have 15+ codes. Trim noisy tails.
        # Task 10.3: Increased limit from 12 to 15 to recover secondary codes.
        if len(processed) > 15:
            # Sort by confidence + specificity + protected status
            processed.sort(key=lambda x: (
                x.get("protected", False) or x.get("source") == "deterministic",
                x.get("confidence", 0),
                len(x.get("code", ""))
            ), reverse=True)
            
            trimmed = processed[:15]
            removed_tail = [c.get("code") for c in processed[15:]]
            logger.info("RuleEngine[trust]: Trimming noisy results from %d to 15. Removed: %s", len(processed), removed_tail)
            processed = trimmed

        if to_remove:
            processed = [c for c in processed if c.get("code", "").upper() not in to_remove]
            logger.info("RuleEngine[clinical]: removed %d codes total: %s", len(to_remove), to_remove)

        print(f"DEBUG RE: returning from apply_clinical_rules: {processed}")
        return processed

    # ------------------------------------------------------------------
    # STAGE 4 (NEW): CPT validation and correction
    # ------------------------------------------------------------------
    @staticmethod
    def apply_cpt_rules(ai_codes: list[dict], note_text: str) -> list[dict]:
        """
        Validate and correct CPT codes.

        Rules:
          1. Block forbidden CPT codes and replace with correct ones.
          2. Blood culture: replace any wrong code with 87040 when evidence present.
          3. Central line: 36558 if tunneled, 36556 if ICU/femoral, else keep.

        Caller must reassign:
            ai_codes = RuleEngine.apply_cpt_rules(ai_codes, note_text)
        """
        if not ai_codes:
            return ai_codes

        processed = copy.deepcopy(ai_codes)
        note_lower = note_text.lower() if note_text else ""

        # Pre-compute clinical context flags
        is_tunneled = any(kw in note_lower for kw in _CPT_TUNNELED_TRIGGERS)
        is_icu_femoral = any(kw in note_lower for kw in _CPT_NON_TUNNELED_TRIGGERS)
        has_blood_culture_context = any(kw in note_lower for kw in _CPT_BLOOD_CULTURE_TRIGGERS)

        for c in processed:
            code = c.get("code", "").upper()
            if c.get("type", "").upper() != "CPT":
                continue

            # ── Block invalid CPT mappings ─────────────────────────────
            if code in _CPT_BLOCK:
                replacement = _CPT_BLOCK[code]
                logger.warning(
                    "RuleEngine[CPT]: blocked invalid code %s -> replaced with %s",
                    code, replacement,
                )
                c["code"] = replacement
                c["rationale"] = (
                    f"[RULE: {code} is not valid for this clinical context — replaced with {replacement}] "
                    + (c.get("rationale") or "")
                )
                code = replacement  # continue with corrected code

            # ── Central line validation ────────────────────────────────
            if code in ("36556", "36558", "36555", "36557"):
                if is_tunneled:
                    # Tunneled takes priority, regardless of prior mapping
                    correct = _CPT_CENTRAL_LINE_TUNNELED
                    if code != correct:
                        logger.info(
                            "RuleEngine[CPT]: central line corrected %s -> %s (tunneled documented)",
                            code, correct,
                        )
                        c["rationale"] = (
                            f"[RULE: Tunneled line documented — corrected to {correct}] "
                            + (c.get("rationale") or "")
                        )
                        c["code"] = correct
                elif is_icu_femoral:
                    # Non-tunneled ICU/femoral placement
                    correct = _CPT_CENTRAL_LINE_NON_TUNNELED
                    if code != correct:
                        logger.info(
                            "RuleEngine[CPT]: central line corrected %s -> %s (ICU/femoral, non-tunneled)",
                            code, correct,
                        )
                        c["rationale"] = (
                            f"[RULE: ICU/femoral non-tunneled line — corrected to {correct}] "
                            + (c.get("rationale") or "")
                        )
                        c["code"] = correct
                # else: leave as-is (no context found)

            # ── Blood culture validation ───────────────────────────────
            if code in ("87040", "86900", "86901", "87205"):
                if has_blood_culture_context and code != _CPT_BLOOD_CULTURE:
                    logger.info(
                        "RuleEngine[CPT]: blood culture corrected %s -> %s",
                        code, _CPT_BLOOD_CULTURE,
                    )
                    c["rationale"] = (
                        f"[RULE: Blood culture context — corrected to {_CPT_BLOOD_CULTURE}] "
                        + (c.get("rationale") or "")
                    )
                    c["code"] = _CPT_BLOOD_CULTURE

        return processed

    # ------------------------------------------------------------------
    # STAGE 5 (NEW): Final validation — dedup + invalid combo removal
    # ------------------------------------------------------------------
    @staticmethod
    def apply_final_validation(ai_codes: list[dict]) -> list[dict]:
        """
        Final cleanup pass before output.

        1. Remove duplicate codes (keep highest confidence copy).
        2. Remove empty/malformed entries.
        3. Within same 3-char ICD prefix — keep highest specificity.
        4. Re-enforce mutex pairs after any upstream changes.

        Returns a NEW, clean list. Caller must reassign:
            ai_codes = RuleEngine.apply_final_validation(ai_codes)
        """
        if not ai_codes:
            return ai_codes

        # Step 1: Remove empty/malformed entries
        valid = [
            c for c in ai_codes
            if c.get("code", "").strip()
            and c.get("type", "") in ("ICD-10", "ICD-10-CM", "CPT", "ICD-9")
        ]

        # Step 2: Dedup — keep highest confidence per code
        seen: dict[str, dict] = {}
        for c in valid:
            key = c.get("code", "").upper()
            if key not in seen or c.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = c
        deduped = list(seen.values())

        # Step 3: Within same 3-char ICD prefix — keep highest specificity (most specific code)
        icd_codes = [c for c in deduped if c.get("type", "").upper() != "CPT"]
        cpt_codes = [c for c in deduped if c.get("type", "").upper() == "CPT"]

        prefix_groups: dict[str, list[dict]] = {}
        for c in icd_codes:
            code = c.get("code", "").upper()
            prefix = code.split(".")[0] if "." in code else code[:3]
            prefix_groups.setdefault(prefix, []).append(c)

        to_remove: set[str] = set()
        for prefix, group in prefix_groups.items():
            if len(group) <= 1:
                continue
            # Within group — apply same mutex logic
            group_codes = {c.get("code", "").upper() for c in group}
            for generic, specific in _MUTEX_PAIRS:
                if generic in group_codes and specific in group_codes:
                    to_remove.add(generic)
                    logger.info(
                        "RuleEngine[final_validation]: removed generic %s (specific %s present)",
                        generic, specific,
                    )

        cleaned_icd = [c for c in icd_codes if c.get("code", "").upper() not in to_remove]
        final_codes = cleaned_icd + cpt_codes

        removed = len(ai_codes) - len(final_codes)
        if removed:
            logger.info("RuleEngine[final_validation]: cleaned %d codes, output %d", removed, len(final_codes))

        # ── TERMINAL GATE: Final evidence validator ───────────────────────
        # This is called with the note_text stored in the RuleEngine context.
        # Because apply_final_validation is a static method without note_text,
        # this gate is best applied via apply_clinical_rules (which has note_text).
        # The gate in final_validator.py is exposed for direct use by audit_pipeline.

        return final_codes

    # ------------------------------------------------------------------
    # Legacy: apply_rules — backward compatibility shim
    # ------------------------------------------------------------------
    @staticmethod
    def apply_rules(clinical_facts: dict, ai_codes: list[dict]) -> list[dict]:
        """Legacy method — calls hierarchy rules for backward compatibility."""
        facts_str = str(clinical_facts)
        return RuleEngine.apply_hierarchy_rules(facts_str, ai_codes)

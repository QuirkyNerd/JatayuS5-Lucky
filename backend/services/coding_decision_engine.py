"""
services/coding_decision_engine.py - Clinical Coding Reasoning and Decision System.

RESPONSIBILITIES:
  1. Decomposes clinical notes into structured entities (Diagnosis, Procedure, etc.).
  2. Implements Principal vs. Secondary sequencing logic.
  3. Executes Clinical Consistency Validation (Anatomy, Laterality, Conflicts).
  4. Manages Manifestation and Excludes1/2 guideline reasoning.
  5. Phase 12: Multi-condition clinical reasoning and causal relationship detection.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from services.validation_utils import (
    is_negated,
    has_prophylaxis_context,
    detect_temporal_status,
    HIGH_AUTHORITY_SECTIONS,
    CHRONIC_MANAGED_PREFIXES,
    compute_evidence_strength,
    clamp_score
)

logger = logging.getLogger(__name__)

class CodingDecisionEngine:
    def __init__(self):
        # Specific clinical priority triggers for Principal Diagnosis
        self.principal_triggers = [
            "sepsis", "myocardial infarction", "stroke", "respiratory failure",
            "acute", "major trauma", "perforated", "ruptured", "hemorrhage",
            "malignant", "embolism", "obstruction"
        ]
        
        # Phase 12: Causal relationship markers
        self.causal_markers = {
            "due to": "CAUSED_BY",
            "secondary to": "CAUSED_BY",
            "caused by": "CAUSED_BY",
            "complicated by": "COMPLICATED_BY",
            "manifestation of": "MANIFESTATION_OF",
            "associated with": "ASSOCIATED_WITH",
            "with": "ASSOCIATED_WITH"
        }

        # Phase 13: NCCI Bundling & Compliance Maps
        self.ncci_bundles = {
            "exploratory laparotomy": ["appendectomy", "cholecystectomy", "colectomy", "gastrectomy"],
            "diagnostic laparoscopy": ["laparoscopic appendectomy", "laparoscopic cholecystectomy"],
            "conscious sedation": ["endoscopy", "colonoscopy", "bronchoscopy"],
            "debridement": ["open reduction", "internal fixation"],
            "imaging guidance": ["aspiration", "biopsy", "injection"]
        }

        self.modifier_triggers = {
            "bilateral": "50",
            "distinct": "59",
            "separate": "25",
            "repeat": "76",
            "staged": "58"
        }

        # Task 2 & 6: Centralized Scoring Weights (Calibrated for Fusion)
        self.weights = {
            "cross_encoder": 0.40,      # Semantic relevance
            "sapbert": 0.35,            # Ontological alignment
            "anatomy": 0.15,            # Regional consistency
            "procedural_intent": 0.10   # Operative approach match
        }

        # Task 10-11: Symbolic Clinical Safety Rules (Registry)
        self.contradiction_rules = {
            "SEX_MISMATCH": {
                "MALE": ["hysterectomy", "oophorectomy", "vaginoplasty", "pregnancy", "obstetrical", "uterine", "cervical"],
                "FEMALE": ["prostatectomy", "vasectomy", "orchiectomy", "penile", "scrotal"]
            },
            "ANATOMY_REGION_MISMATCH": {
                "UPPER_EXTREMITY": ["femur", "tibia", "ankle", "foot", "pelvis", "hip", "toe", "knee"],
                "LOWER_EXTREMITY": ["radius", "humerus", "wrist", "hand", "shoulder", "finger", "elbow"]
            },
            "APPROACH_MISMATCH": {
                "laparoscopic": ["open incision", "major opening", "thoracotomy", "laparotomy", "radical open"],
                "open": ["minimally invasive", "percutaneous", "endoscopic", "arthroscopic", "laparoscopic"]
            },
            "AGE_MISMATCH": {
                "ADULT": ["pediatric-only", "neonatal", "newborn care"],
                "PEDIATRIC": ["geriatric", "adult-only"]
            }
        }

    def process_coding_decisions(self, note_text: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for Phase 12 Reasoning.
        Converts raw RAG candidates into a coherent multi-condition coding decision.
        """
        # 1. Decompose Note Context & Detect Relationships
        decomposition = self._decompose_note(note_text)
        causal_links = self._detect_causal_relationships(note_text)
        
        # 2. Extract and Filter Candidates
        icd_candidates = self._filter_and_score_candidates(rag_results.get("icd_candidates", []), note_text, "ICD")
        cpt_candidates = self._filter_and_score_candidates(rag_results.get("cpt_candidates", []), note_text, "CPT")
        
        # 3. Separate Chronic vs Acute (Phase 12 Task 4)
        acute_dx = []
        chronic_dx = []
        for c in icd_candidates:
            code = str(c.get("normed_code", "")).upper().strip()
            is_chron = False
            for p in CHRONIC_MANAGED_PREFIXES:
                if code.startswith(str(p).upper().strip()):
                    is_chron = True
                    break
            
            if is_chron: chronic_dx.append(c)
            else: acute_dx.append(c)
        
        # 4. Apply Sequencing Logic (Principal vs Secondary)
        principal_dx, secondary_dx = self._sequence_diagnoses(acute_dx, decomposition)
        
        # 5. Redundant Symptom Suppression (Phase 12 Task 3)
        final_secondary = self._suppress_redundant_symptoms(secondary_dx, principal_dx)
        
        # 6. Procedure-Diagnosis Linking
        linked_procedures = self._link_procedures(cpt_candidates, principal_dx + final_secondary)
        
        # 7. Final Consistency & Conflict Resolution (Task 8 & 15: Scoring-aware collapse)
        # Combine and let the smartest candidate win.
        final_icd = self._collapse_duplicate_families(principal_dx + final_secondary + chronic_dx, "ICD")
        final_cpt = self._collapse_duplicate_families(linked_procedures, "CPT")

        # 8. Phase 13/14: Compliance, Billing & Ontological Logic
        try:
            compliance_results = self._enforce_coding_compliance(final_icd, final_cpt, decomposition, note_text)
            final_cpt_compliant = compliance_results["compliant_procedures"]
            compliance_flags = compliance_results["flags"]
            modifiers = compliance_results["modifiers"]
        except Exception as e:
            logger.error(f"CRITICAL: Compliance Engine Failed: {e}")
            final_cpt_compliant = final_cpt
            compliance_results = {"flags": ["COMPLIANCE_CRASH"], "modifiers": [], "necessity_map": {}, "suppressed": [], "compliance_score": 0.0, "ncci_log": [], "exclusion_log": []}
            compliance_flags = ["COMPLIANCE_CRASH"]
            modifiers = []
        
        # 9. Task 1, 2, 7: Clinical Validation & Confidence Calibration
        validated_icd = []
        try:
            for c in final_icd:
                contradictions = self._detect_contradictions(c, decomposition, rag_results.get("anatomy", {}))
                calibration = self._calibrate_confidence(c, contradictions)
                c.update(calibration)
                c["contradiction_trace"] = contradictions
                validated_icd.append(c)
        except Exception as e:
            logger.error(f"CRITICAL: ICD Validation Failed: {e}")
            validated_icd = final_icd
            
        validated_cpt = []
        try:
            for c in final_cpt_compliant:
                contradictions = self._detect_contradictions(c, decomposition, rag_results.get("anatomy", {}))
                calibration = self._calibrate_confidence(c, contradictions)
                c.update(calibration)
                c["contradiction_trace"] = contradictions
                validated_cpt.append(c)
        except Exception as e:
            logger.error(f"CRITICAL: CPT Validation Failed: {e}")
            validated_cpt = final_cpt_compliant

        # 10. Task 13 & 15: Structuring Explainable Output
        final_chronic = []
        final_secondary_only = []
        for c in validated_icd:
            code = str(c.get("normed_code", "")).upper().strip()
            is_chron = False
            for p in CHRONIC_MANAGED_PREFIXES:
                if code.startswith(str(p).upper().strip()):
                    is_chron = True; break
            
            if is_chron: final_chronic.append(c)
            else: final_secondary_only.append(c)

        # Overall Case Metrics
        all_calibrations = [c.get("score", 0.0) for c in validated_icd + validated_cpt]
        avg_case_conf = sum(all_calibrations) / len(all_calibrations) if all_calibrations else 0.0
        
        return {
            "principal_diagnosis": [validated_icd[0]] if validated_icd else [],
            "secondary_diagnoses": [c for c in final_secondary_only if c != (validated_icd[0] if validated_icd else None)],
            "chronic_conditions": final_chronic,
            "procedures": validated_cpt,
            "modifiers": modifiers,
            "compliance_flags": compliance_flags,
            "medical_necessity": compliance_results["necessity_map"],
            "decomposition": decomposition,
            "causal_relationships": causal_links,
            "guidelines_used": self._extract_relevant_guidelines(rag_results.get("guideline_candidates", [])),
            "suppressed_candidates": compliance_results["suppressed"] + [c for c in (icd_candidates + cpt_candidates) if c not in final_icd + final_cpt],
            "confidence": {
                "score": round(avg_case_conf, 3),
                "level": "HIGH" if avg_case_conf > 0.8 else ("MEDIUM" if avg_case_conf > 0.6 else "LOW"),
                "review_required": avg_case_conf < 0.65 or any(c.get("review_required") for c in validated_icd + validated_cpt),
                "clinical_reasoning": validated_icd[0].get("reasoning") if validated_icd else "No diagnosis identified.",
                "compliance_score": compliance_results["compliance_score"]
            },
            "audit_trace": {
                "engine_version": "Phase 14 (Ontology Constrained Platform)",
                "anatomy_grounding": rag_results.get("anatomy", {}),
                "procedural_intent": rag_results.get("procedural_intent", {}),
                "ontology_shift": rag_results.get("ontology_shift", []),
                "ncci_edits": compliance_results["ncci_log"],
                "exclusion_logic": compliance_results["exclusion_log"],
                "semantic_validation": {
                    "model": "SapBERT",
                    "top_concept_match": validated_icd[0].get("sapbert_score", 0.0) if validated_icd else 0.0,
                    "ontology_precision": 1.0 - (validated_icd[0].get("ontology_distance", 0.0) if validated_icd else 0.0)
                }
            }
        }

    def _decompose_note(self, note_text: str) -> Dict[str, Any]:
        lower = note_text.lower()
        return {
            "acuity": "ACUTE" if any(k in lower for k in ["acute", "critical", "emergency", "severe", "perforated", "ruptured"]) else "CHRONIC",
            "temporality": detect_temporal_status(note_text),
            "negated_entities": [tok for tok in ["pneumonia", "fracture", "infection", "sepsis"] if is_negated(tok, note_text)],
            "laterality": "LEFT" if "left" in lower else ("RIGHT" if "right" in lower else "UNSPECIFIED"),
            "encounter_type": self._infer_encounter_type(lower),
            "demographics": self._infer_patient_demographics(lower)
        }

    def _infer_patient_demographics(self, text: str) -> Dict[str, Any]:
        """Task 2: Demographic Contradiction detection."""
        sex = "UNKNOWN"
        if re.search(r"\b(male|gentleman|man|he|his|him)\b", text): sex = "MALE"
        if re.search(r"\b(female|lady|woman|she|her|hers)\b", text): sex = "FEMALE"
        
        age_group = "ADULT"
        if re.search(r"\b(pediatric|child|infant|newborn|toddler|boy|girl)\b", text): age_group = "PEDIATRIC"
        
        return {"sex": sex, "age_group": age_group}

    def _infer_encounter_type(self, text: str) -> str:
        if any(k in text for k in ["initial", "presents with", "admission", "new onset"]): return "INITIAL"
        if any(k in text for k in ["subsequent", "follow-up", "f/u", "stable", "resolved"]): return "SUBSEQUENT"
        if any(k in text for k in ["sequela", "late effect"]): return "SEQUELA"
        return "INITIAL"

    def _detect_causal_relationships(self, text: str) -> List[Dict[str, str]]:
        """
        Phase 12 Task 2: Causal Relationship Detection.
        """
        links = []
        lower = text.lower()
        for marker, rel_type in self.causal_markers.items():
            if marker in lower:
                # Find the surrounding context
                pattern = rf"(\w+)\s+{re.escape(marker)}\s+(\w+)"
                matches = re.findall(pattern, lower)
                for m in matches:
                    links.append({"subject": m[0], "relationship": rel_type, "object": m[1]})
        return links

    def _filter_and_score_candidates(self, candidates: List[Dict], note_text: str, code_type: str) -> List[Dict]:
        valid = []
        for cand in candidates:
            code = (cand.get("normed_code") or cand.get("code") or "").upper().strip()
            desc = (cand.get("doc") or cand.get("description", "")).lower()
            
            # Step 1: Reject legacy ICD-9 codes (numeric-only category)
            # ICD-10 always starts with a letter.
            if code_type == "ICD" and code and not code[0].isalpha():
                continue

            # Step 2: Compute Evidence Strength
            entity_conf = cand.get("confidence") or cand.get("score") or 0.5
            score, reason = compute_evidence_strength(
                code, desc, note_text, 
                entity_confidence=entity_conf,
                is_rag_only=(cand.get("source") == "rag")
            )
            
            if score == 0:
                continue
            
            cand["decision_score"] = clamp_score(score)
            # Extend Forensic Trace (Task 1)
            if "forensic" not in cand: cand["forensic"] = {}
            cand["forensic"].update({
                "raw_confidence": round(entity_conf, 3),
                "evidence_strength": round(score, 3),
                "decision_reason": reason
            })
            valid.append(cand)
        
        # Sort and deduplicate
        return sorted(valid, key=lambda x: x["decision_score"], reverse=True)

    def _sequence_diagnoses(self, candidates: List[Dict], decomp: Dict) -> tuple[List[Dict], List[Dict]]:
        if not candidates: return [], []
        principal = []
        secondary = []
        for cand in candidates:
            desc = cand.get("doc", "").lower()
            is_principal_candidate = any(t in desc for t in self.principal_triggers)
            if decomp["acuity"] == "ACUTE" and is_principal_candidate:
                cand["decision_score"] += 0.12
            if is_principal_candidate and not principal:
                principal.append(cand)
            else:
                secondary.append(cand)
        if not principal and candidates:
            principal = [candidates[0]]; secondary = candidates[1:]
        return principal, secondary

    def _collapse_duplicate_families(self, candidates: List[Dict], code_type: str = "ICD") -> List[Dict]:
        """
        Task 8 & 15: Collapses multiple candidates from the same clinical family.
        Retains only the highest-scoring candidate per group.
        """
        if not candidates: return []
        
        # Sort by score first to favor high confidence in the greedy group selection
        sorted_cands = sorted(candidates, key=lambda x: x.get("decision_score", 0.0), reverse=True)
        
        families: Dict[str, Dict] = {}
        for cand in sorted_cands:
            code = (cand.get("normed_code") or cand.get("code") or "").upper().strip()
            if not code: continue
            
            # Family prefix
            if code_type == "ICD":
                group_key = code[:3] # S72.xxx -> S72
            else:
                # CPT is more specific. Only collapse near-identical procedures (e.g. 27130 vs 27131)
                # or keep full code to preserve specificity.
                group_key = code[:4] # 27130 -> 2713
            
            if group_key not in families:
                families[group_key] = cand
            else:
                # Comparison logic: if scores are very close, favor the one with higher SapBERT
                existing = families[group_key]
                existing_score = existing.get("decision_score", 0.0)
                new_score = cand.get("decision_score", 0.0)
                
                if abs(new_score - existing_score) < 0.02:
                    existing_sap = existing.get("forensic", {}).get("sapbert_score", 0.0)
                    new_sap = cand.get("forensic", {}).get("sapbert_score", 0.0)
                    if new_sap > existing_sap:
                        families[group_key] = cand
                # Else: we already sorted by decision_score, so 'existing' is likely higher or equal
                    
        return list(families.values())

    def _suppress_redundant_symptoms(self, secondary: List[Dict], principal: List[Dict]) -> List[Dict]:
        """
        Phase 12 Task 3: Suppress symptom codes when a definitive diagnosis exists.
        Handles partial matches and clinical synonyms.
        """
        if not principal: return secondary
        p_desc = principal[0].get("doc", "").lower()
        
        # Map definitive diagnoses to explained symptoms (expanded for better matching)
        explained_symptoms = {
            "appendicitis": ["abdominal", "nausea", "fever", "vomiting", "pain"],
            "pneumonia": ["cough", "fever", "breath", "dyspnea", "sputum", "chest pain"],
            "fracture": ["pain", "swelling", "deformity", "bruising"],
            "sepsis": ["fever", "tachycardia", "hypotension", "shivering"],
            "respiratory failure": ["dyspnea", "hypoxia", "breath", "oxygen", "cyanosis"],
            "myocardial infarction": ["chest pain", "angina", "sweating", "nausea", "dyspnea"]
        }
        
        suppress_keywords = []
        for diag, symptoms in explained_symptoms.items():
            if diag in p_desc:
                suppress_keywords.extend(symptoms)
        
        final_secondary = []
        for cand in secondary:
            code = cand.get("normed_code", "").upper()
            desc = cand.get("doc", "").lower()
            
            # Suppress R-codes (Symptoms) that are explained by the principal
            # and other non-specific clinical signs
            is_redundant = False
            is_chronic = any(code.startswith(p) for p in CHRONIC_MANAGED_PREFIXES)
            
            if code.startswith("R") or ("unspecified" in desc and not is_chronic):
                if any(sk in desc for sk in suppress_keywords):
                    logger.info(f"SUPPRESS REDUNDANT SYMPTOM: {code} ({desc})")
                    is_redundant = True
            
            if not is_redundant:
                final_secondary.append(cand)
        return final_secondary

    def _detect_contradictions(self, cand: Dict, decomp: Dict, rag_anatomy: Dict) -> List[Dict[str, Any]]:
        """
        Task 10-12: Principal Clinical Contradiction Engine.
        Detects and describes impossible or dangerous clinical mismatches.
        """
        conflicts = []
        doc = cand.get("doc", "").lower()
        meta = cand.get("meta", {})
        
        # 1. Sex Mismatch (Task 10.4)
        sex = decomp["demographics"]["sex"]
        if sex in self.contradiction_rules["SEX_MISMATCH"]:
            for forbidden in self.contradiction_rules["SEX_MISMATCH"][sex]:
                if forbidden in doc:
                    conflicts.append({
                        "type": "SEX_MISMATCH",
                        "severity": "HIGH",
                        "description": f"Patient is {sex}, procedure is {forbidden}"
                    })

        # 2. Anatomy Region Mismatch (Task 10.1)
        q_regions = set(rag_anatomy.get("regions", []))
        for region, forbidden_keywords in self.contradiction_rules["ANATOMY_REGION_MISMATCH"].items():
            if region in q_regions:
                for kw in forbidden_keywords:
                    if kw in doc:
                        conflicts.append({
                            "type": "ANATOMY_REGION_MISMATCH",
                            "severity": "HIGH",
                            "description": f"Note mentions {region}, candidate mentions {kw}"
                        })

        # 3. Approach Mismatch (Task 10.2)
        q_approach = decomp.get("approach", "unknown")
        if q_approach in self.contradiction_rules["APPROACH_MISMATCH"]:
             for forbidden in self.contradiction_rules["APPROACH_MISMATCH"][q_approach]:
                 if forbidden in doc:
                     conflicts.append({
                         "type": "APPROACH_MISMATCH",
                         "severity": "MEDIUM",
                         "description": f"Note approach {q_approach} contradicts {forbidden}"
                     })

        # 4. Age Mismatch (Task 10.4)
        age = decomp["demographics"]["age_group"]
        if age in self.contradiction_rules["AGE_MISMATCH"]:
            for forbidden in self.contradiction_rules["AGE_MISMATCH"][age]:
                if forbidden in doc:
                    conflicts.append({
                        "type": "AGE_MISMATCH",
                        "severity": "HIGH",
                        "description": f"Patient is {age}, code is for {forbidden}"
                    })

        # 5. Laterality Mismatch (Task 10.1)
        q_lat = decomp["laterality"]
        if q_lat != "UNSPECIFIED":
            if ("left" in doc and q_lat == "RIGHT") or ("right" in doc and q_lat == "LEFT"):
                conflicts.append({
                    "type": "LATERALITY_MISMATCH",
                    "severity": "HIGH",
                    "description": f"Note is {q_lat}, code is opposing side"
                })

        return conflicts

    def _calibrate_confidence(self, cand: Dict, contradictions: List[Dict]) -> Dict[str, Any]:
        """
        Task 7-9: Principal Confidence Calibration.
        Computes safe failure levels and explainable thresholds.
        """
        forensic = cand.get("forensic", {})
        sap_score = forensic.get("sapbert_score", 0.5)
        cross_score = forensic.get("cross_encoder", 0.5)
        anatomy_score = forensic.get("consistency", 0.0)
        
        # Normalize anatomy consistency to [0, 1]
        anatomy_norm = clamp_score((anatomy_score + 1.0) / 2.0)
        
        # Task 2 & 6: Weighted fusion
        base_score = (
            (cross_score * self.weights["cross_encoder"]) +
            (sap_score * self.weights["sapbert"]) +
            (anatomy_norm * self.weights["anatomy"])
        )
        
        # Task 11: Contradiction Penalties
        # Severe contradictions override semantic similarity (Task 11)
        penalty = 0.0
        for c in contradictions:
            if c["severity"] == "HIGH": penalty += 0.4
            else: penalty += 0.15
            
        final_conf = clamp_score(base_score - penalty)
        
        # Task 8: Safe Failure Modes
        level = "INSUFFICIENT_EVIDENCE"
        review_required = True
        
        if final_conf > 0.85 and not contradictions:
            level = "HIGH_CONFIDENCE"
            review_required = False
        elif final_conf > 0.65:
            level = "PROBABLE_MATCH"
            review_required = (penalty > 0)
        elif final_conf > 0.45:
            level = "REVIEW_REQUIRED"
            review_required = True
            
        return {
            "score": round(final_conf, 3),
            "level": level,
            "review_required": review_required,
            "reasoning": self._generate_clinical_reasoning(cand, contradictions, final_conf)
        }

    def _generate_clinical_reasoning(self, cand: Dict, contradictions: List[Dict], confidence: float) -> str:
        """
        Task 14: Human-Readable Clinical Reasoning Engine.
        """
        reasons = []
        
        # 1. Address Contradictions
        if contradictions:
            severe = [c["description"] for c in contradictions if c["severity"] == "HIGH"]
            if severe:
                reasons.append(f"Clinical conflict detected: {'; '.join(severe)}.")
            else:
                reasons.append(f"Minor clinical inconsistency: {contradictions[0]['description']}.")
                
        # 2. Address Ontological Alignment (SapBERT)
        sap_score = cand.get("sapbert_score", 0.0)
        if sap_score > 0.85:
            reasons.append("Strong ontological alignment confirmed by SapBERT.")
        elif sap_score < 0.45:
            reasons.append("Ontology validator detected distant anatomical or procedural relation.")
            
        # 3. Address Semantic Match
        if confidence > 0.80:
            reasons.append("High semantic relevance and consistent clinical evidence.")
        elif confidence < 0.50:
            reasons.append("Weak clinical evidence or ambiguous procedure description.")
            
        return " ".join(reasons)

    def _link_procedures(self, cpt_cands: List[Dict], icd_cands: List[Dict]) -> List[Dict]:
        for cpt in cpt_cands:
            cpt_anatomy = str(cpt.get("meta", {}).get("anatomy", "")).lower()
            for icd in icd_cands:
                icd_anatomy = str(icd.get("meta", {}).get("anatomy", "")).lower()
                if cpt_anatomy != "general" and icd_anatomy != "general" and (cpt_anatomy in icd_anatomy or icd_anatomy in cpt_anatomy):
                    cpt["decision_score"] += 0.15
                    cpt["linked_to"] = icd.get("normed_code")
        return sorted(cpt_cands, key=lambda x: x["decision_score"], reverse=True)

    def _resolve_conflicts(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return []
        final = []
        seen_roots = set()
        for cand in candidates:
            code = cand.get("normed_code", "").upper()
            root = code.split(".")[0]
            if root in seen_roots: continue
            final.append(cand)
            seen_roots.add(root)
        return final[:12]

    def _extract_relevant_guidelines(self, guide_cands: List[Dict]) -> List[str]:
        return [g["doc"] for g in guide_cands[:3]]

    def _calculate_overall_confidence(self, icd: List[Dict], cpt: List[Dict]) -> float:
        scores = [c["decision_score"] for c in icd + cpt]
        # Incorporate SapBERT ontological agreement if available
        sap_boost = 0.0
        if icd and "sapbert_score" in icd[0]:
            sap_boost = (icd[0]["sapbert_score"] - 0.5) * 0.1 # Small nudge based on ontology agreement
            
        base_conf = sum(scores) / len(scores) if scores else 0.0
        return clamp_score(base_conf + sap_boost)

    def _enforce_coding_compliance(self, icd: List[Dict], cpt: List[Dict], decomp: Dict, text: str) -> Dict[str, Any]:
        """
        Phase 13: Core Compliance & Billing Logic.
        """
        flags = []
        suppressed = []
        ncci_log = []
        exclusion_log = []
        necessity_map = {}
        modifiers = []

        # 1. NCCI Bundling Logic (Task 1)
        compliant_cpt = []
        for c in cpt:
            desc = c.get("doc", "").lower()
            is_bundled = False
            for major, minors in self.ncci_bundles.items():
                if any(m in desc for m in minors):
                    # Check if major procedure is also present
                    if any(major in other.get("doc", "").lower() for other in cpt if other != c):
                        flags.append(f"NCCI BUNDLE: {c.get('normed_code')} suppressed by major procedure.")
                        ncci_log.append({"code": c.get("normed_code"), "reason": f"Bundled into {major}"})
                        suppressed.append(c)
                        is_bundled = True
                        break
            if not is_bundled:
                compliant_cpt.append(c)

        # 2. Modifier Reasoning (Task 2)
        for c in compliant_cpt:
            code_mods = []
            if decomp["laterality"] == "BILATERAL":
                code_mods.append("50")
                flags.append(f"MODIFIER 50: Bilateral inferred for {c.get('normed_code')}")
            elif decomp["laterality"] in ["LEFT", "RIGHT"]:
                code_mods.append("LT" if decomp["laterality"] == "LEFT" else "RT")

            # Modifier 25 (Separate E/M)
            if any(str(c.get("normed_code")).startswith(("992", "994")) for other in compliant_cpt):
                if not str(c.get("normed_code")).startswith(("992", "994")):
                     code_mods.append("25")
            
            if code_mods:
                modifiers.append({"code": c.get("normed_code"), "modifiers": code_mods})

        # 3. ICD Exclusion Logic (Task 3)
        # Simplified Excludes1: If Sepsis (A41) is present, do not code R65.2 (SIRS)
        final_icd = []
        for i in icd:
            code = i.get("normed_code", "")
            if code.startswith("R65.2") and any(other.get("normed_code", "").startswith("A41") for other in icd):
                exclusion_log.append({"code": code, "reason": "Excludes1: Sepsis overrides SIRS"})
                suppressed.append(i)
                continue
            final_icd.append(i)

        # 4. Diagnosis-Procedure Justification (Task 7)
        for c in compliant_cpt:
            c_desc = c.get("doc", "").lower()
            justified = False
            for i in final_icd:
                i_desc = i.get("doc", "").lower()
                # Basic justification: keyword overlap
                if any(kw in i_desc for kw in ["appendicitis", "fracture", "pneumonia", "sepsis", "infarction"]):
                    if any(kw in c_desc for kw in ["appendectomy", "reduction", "fixation", "debridement", "drainage"]):
                        justified = True
                        necessity_map[c.get("normed_code")] = f"Justified by {i.get('normed_code')}"
                        break
            if not justified:
                flags.append(f"AUDIT WARNING: {c.get('normed_code')} lacks strong diagnosis justification.")

        return {
            "compliant_procedures": compliant_cpt,
            "flags": flags,
            "modifiers": modifiers,
            "suppressed": suppressed,
            "ncci_log": ncci_log,
            "exclusion_log": exclusion_log,
            "necessity_map": necessity_map,
            "compliance_score": 1.0 - (len(flags) * 0.05)
        }

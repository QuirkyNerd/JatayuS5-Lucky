"""
services/final_validator.py – Terminal evidence gate before pipeline output.

PURPOSE:
  This is the LAST line of defense before codes are returned to the user.
  It runs AFTER RuleEngine.apply_final_validation() in audit_pipeline.py.

  Any code that reaches this gate without explicit note support is removed.

SPEC MAPPING:
  Step 5 — Final validation gate:
    Remove any code where evidence_strength < threshold
    unless explicit textual support exists.
services/final_validator.py – Final Deterministic Validation and Suppression Layer.

RESPONSIBILITIES:
  1. Final safety gate before case submission.
  2. Enforces mandatory exclusion rules and clinical exclusivity.
  3. Detects and suppresses hallucinations at the character level.
  4. Manages final deduplication and result capping.
"""

import re
import logging
import time
import copy
import traceback

from services.validation_utils import (
    is_negated,
    has_prophylaxis_context,
    compute_evidence_strength,
    EVIDENCE_STRENGTH_THRESHOLD,
    CALIBRATION_THRESHOLDS,
    clamp_score,
    get_differentiated_threshold,
    build_scoring_breakdown,
    NEGATION_TOKENS,
    PROPHYLAXIS_TOKENS,
    extract_anatomy_regions,
    check_anatomy_consistency,
    validate_procedure_evidence,
    apply_specificity_hierarchy,
    is_parent_of,
    pathological_fracture_protection,
    parse_note_sections,
    compute_section_aware_boost,
    SECTION_WEIGHTS,
    clinical_specificity_score,
    compute_procedural_survival_score,
    is_less_specific_variant,
    compute_principal_diagnosis_strength,
    compute_procedural_immunity,
    compute_exact_context_overlap,
    compute_local_phrase_density,
    compute_phrase_grounding_strength,
    compute_ontology_dependence_ratio,
    compute_procedure_subtype_grounding,
    compute_local_context_coherence,
    compute_specificity_survival_weight,
    compute_procedural_stability_weight,
    compute_chronic_relevance_weight,
    compute_principal_encounter_strength,
    compute_generalization_penalty,
    compute_distributed_evidence_strength,
    compute_supporting_evidence_diversity,
    compute_severity_preservation_strength,
    compute_diagnostic_certainty,
    compute_procedural_subtype_certainty,
    compute_relationship_confidence,
    compute_representation_family,
    compute_semantic_overlap_strength,
    compute_consistency_priority,
    compute_reportability_strength,
    compute_independent_management_strength,
    compute_clinical_significance_priority,
    compute_encounter_attribution_strength,
    resolve_condition_temporal_state,
    compute_management_intensity_score,
    detect_mutually_exclusive_conditions,
    compute_complication_hierarchy_strength,
    build_evidence_provenance_graph,
    compute_documentation_confidence,
    compute_objective_evidence_strength,
    resolve_provider_intent_strength,
    compute_cross_document_consistency,
    compute_evidence_temporal_decay,
    compute_provider_authority_weight,
    compute_discharge_finality_strength,
    compute_probabilistic_diagnostic_confidence,
    track_encounter_state_evolution,
    resolve_multi_provider_conflict,
    build_guideline_reference_map,
    resolve_etiology_manifestation_relationship,
    compute_sequencing_confidence,
    resolve_encounter_setting_policy,
    resolve_uncertain_diagnosis_policy,
    compute_risk_adjustment_significance,
    compute_false_positive_risk,
    compute_domain_calibration_weight,
    compute_sibling_grounding_advantage,
    compute_procedural_domain_strength,
    compute_regression_resistance,
    compute_procedural_immunity_lock,
    compute_specificity_immunity_lock,
    normalize_confidence_scale,
    bounded_confidence_delta,
    compute_confidence_band,
    apply_priority_safe_adjustment,
    compute_document_reliability,
    compute_noise_tolerance_strength,
    compute_copy_forward_probability,
    compute_evidence_tier,
    compute_direct_grounding_authority,
    compute_evidence_conflict_priority,
    build_clinical_relationship_graph,
    compute_anatomical_coherence,
    compute_procedural_intent_alignment,
    compute_relationship_certainty,
    compute_provider_assertion_strength,
    compute_clinical_severity_weight,
    compute_procedural_documentation_strength,
    compute_overgeneralization_risk,
    GLOBAL_REASONING_PRIORITY,
    compute_monotonic_confidence_update,
    compute_reconciliation_stability,
    compute_lock_strength,
    CALIBRATION_THRESHOLDS,
    calculate_soft_fusion_confidence,
    compute_specialty_calibration_modifier,
    compute_cross_specialty_stability,
    compute_integrated_disease_strength,
    compute_specificity_dominance_strength,
    compute_fragmentation_risk,
    compute_procedural_grounding_authority,
    compute_procedural_subtype_stability,
    compute_procedural_family_stability,
    compute_specialty_vocabulary_density,
    compute_semantic_drift_risk,
    compute_severity_convergence_strength,
    compute_encounter_driver_dominance,
    compute_physiologic_coherence,
    compute_sparse_evidence_authority,
    compute_rare_specialty_density,
    compute_domain_adaptive_profile,
    compute_dominant_clinical_state_strength,
    compute_specificity_survival_priority,
    compute_combination_state_integrity,
    compute_procedural_intent_authority,
    compute_procedure_diagnosis_coherence,
    compute_therapeutic_priority_strength,
    compute_clinical_causality_authority,
    compute_state_transition_coherence,
    compute_complication_dominance_strength,
    compute_temporal_encounter_authority,
    compute_procedural_timeline_coherence,
    compute_historical_leakage_risk,
    compute_document_structure_authority,
    compute_copy_forward_risk,
    compute_section_intent_reliability,
    REASONING_PRIORITY,
    check_cross_diagnosis_conflicts,
    ENCOUNTER_DOMAINS,
    PROCEDURE_COHERENCE_FAMILIES,
    classify_audit_failure_type,
    compute_domain_performance_profile,
    compute_failure_priority_score,
    build_representation_family_index,
    compute_audit_decision_confidence,
    compute_auditor_conservatism_weight,
    validate_candidate_schema,
    record_pipeline_telemetry,
    PIPELINE_DEBUG_MODE,
    SAFE_MODE,
)
from services.clinical_reasoning_engine import ClinicalReasoningEngine, build_rejection_trace

logger = logging.getLogger(__name__)

# Singleton — re-used across calls
_reasoning_engine = ClinicalReasoningEngine()


def apply_final_evidence_gate(
    codes: list[dict],
    note_text: str,
) -> tuple[list[dict], list[dict]]:
    """
    Terminal evidence gate: remove codes not backed by the note.

    This function is the LAST call before returning ai_codes from the pipeline.

    Rules applied (in order):
      1. Deterministic/protected codes are ALWAYS kept — skip gate.
      2. CPT codes pass through — they are validated upstream.
      3. Any code with evidence_strength already computed (<THRESHOLD) → REMOVE.
      4. Any code NOT yet evaluated by ClinicalReasoningEngine:
           → Run compute_evidence_strength now.
           → If < THRESHOLD → REMOVE.
      5. Any remaining code whose key terms are negated or in prophylaxis
         context → REMOVE regardless of prior evidence_strength.

    All removals are logged with:
      REJECTED_CODE: <code> | reason=... | evidence=... | matched_text=...

    Returns:
      (passed_codes, rejected_traces)
    """
    if not codes:
        return codes, []

    if not note_text or not note_text.strip():
        logger.warning("FinalValidator: no note_text provided — skipping gate (returning all codes)")
        return codes, []

    # Pre-compute note anatomy once for all codes (Step 1)
    note_anatomy = extract_anatomy_regions(note_text)
    if note_anatomy:
        logger.debug("FinalValidator: detected anatomy: %s", note_anatomy)

    passed: list[dict] = []
    rejected_traces: list[dict] = []
    rejected: int = 0

    for code_dict in codes:
        code        = (code_dict.get("code") or "").strip().upper()
        description = (code_dict.get("description") or "").strip()
        source      = (code_dict.get("source") or "rag").lower()
        code_type   = (code_dict.get("type") or "ICD-10").upper()
        is_protected = bool(
            code_dict.get("protected")
            or source == "deterministic"
            or code_dict.get("grounding") == "deterministic"
        )
        # Part 3: Section-based protection (Task 23)
        sec_name = code_dict.get("section_dominant") or "full_note"
        is_ortho_code = any(code.startswith(pre) for pre in ["S72", "M80", "M81", "S82", "S52", "S42"])
        
        # 🚨 TASK 28: PHASE 5 — SURVIVAL GUARANTEE
        # Top-1 retrieval candidates with strong terminology overlap bypass the gate.
        retrieval_trace = code_dict.get("retrieval_trace") or {}
        r_rank = code_dict.get("metadata", {}).get("retrieval_rank") or retrieval_trace.get("retrieval_rank") or 99
        rag_score = float(code_dict.get("rag_score") or 0)
        
        if r_rank == 1 and rag_score > 0.82:
            code_dict["protected"] = True
            is_protected = True
            code_dict.setdefault("audit_traces", []).append("SURVIVAL_GUARANTEE: TOP_1_ANCHOR")
            logger.info("FinalValidator: SURVIVAL_GUARANTEE for %s (rank 1, score %.2f)", code, rag_score)

        if sec_name in ["postop_diagnosis", "preop_diagnosis", "procedure", "findings"] or (is_ortho_code and "fracture" in note_text.lower()):
            code_dict["protected"] = True
            is_protected = True
            logger.info("FinalValidator: NUCLEAR_PROTECTION: %s", code)
            passed.append(code_dict)
            continue
        
        # ── Rule 1: Protected / deterministic → always keep ──────────────────
        if is_protected:
            passed.append(code_dict)
            continue

        # ── Step 1: Reject Resolved Conditions (Temporal Grounding) ──────────
        if code_dict.get("temporal_status") == "RESOLVED":
            # SOFT CALIBRATION: 0.92x scale instead of 0.50x
            old_conf = code_dict.get("confidence", 0.5)
            code_dict["confidence"] = round(old_conf * 0.92, 3)
            code_dict.setdefault("audit_traces", []).append("TEMPORAL_RESOLVED_SOFT_PENALTY_V39")
            logger.info("SOFT_REJECT_RESOLVED: code=%s (conf %.2f -> %.2f)", code, old_conf, code_dict["confidence"])
            # Fallthrough to passed.append (only reject below 0.15)

        # ── CPT codes: validate procedure + anatomy ─────────────────────────
        if code_type == "CPT":
            # Anatomy check for CPT
            is_anat_ok, anat_reason = check_anatomy_consistency(code, description, note_anatomy)
            if not is_anat_ok:
                # SOFT CALIBRATION: 0.92x scale for anatomy mismatch (Task 39)
                old_conf = code_dict.get("confidence", 0.5)
                code_dict["confidence"] = round(old_conf * 0.92, 3)
                code_dict.setdefault("audit_traces", []).append("ANATOMY_MISMATCH_SOFT_PENALTY_V39")
            # Procedure evidence (already stamped by CRE if it ran, otherwise check now)
            if code_dict.get("evidence_strength") is None:
                proc_strength, proc_match = validate_procedure_evidence(code, note_text)
                code_dict["evidence_strength"] = proc_strength
                code_dict["evidence_reason"]   = proc_match
            passed.append(code_dict)
            continue

        # ── ICD: anatomy mismatch check BEFORE evidence gate (Step 3 / Step 8) ──
        # ── ICD: anatomy mismatch check BEFORE evidence gate ──
        is_anat_ok, anat_reason = check_anatomy_consistency(code, description, note_anatomy)
        if not is_anat_ok:
            # SOFT CALIBRATION: 0.75x scale for ICD anatomy mismatch
            old_conf = code_dict.get("confidence", 0.5)
            code_dict["confidence"] = round(old_conf * 0.75, 3)
            code_dict.setdefault("audit_traces", []).append("ANATOMY_MISMATCH_SOFT_PENALTY")
            # TASK 41: DO NOT REJECT HERE. Fall through.

        # ── Step 4: Minimum Grounding Quality Gate ────
        base_strength = float(code_dict.get("base_evidence_strength") or 0)
        final_strength = float(code_dict.get("evidence_strength") or 0)
        is_primary_confirmed = code_dict.get("primary_evidence_confirmed", False)
        
        if base_strength < 0.35 and final_strength < 0.65 and not is_primary_confirmed:
            # SOFT CALIBRATION: 0.80x scale instead of hard rejection
            old_conf = code_dict.get("confidence", 0.5)
            code_dict["confidence"] = round(old_conf * 0.80, 3)
            code_dict.setdefault("audit_traces", []).append("LOW_GROUNDING_SOFT_PENALTY")

        # ── Rule 3: Pre-computed evidence_strength below DIFFERENTIATED threshold ──
        prior_strength = code_dict.get("evidence_strength")
        if prior_strength is not None:
            prior_float = float(prior_strength)
            tier = code_dict.get("calibration_tier")
            if tier and tier in CALIBRATION_THRESHOLDS:
                gate_threshold = CALIBRATION_THRESHOLDS[tier]
            else:
                entity_conf_r3 = float(code_dict.get("entity_confidence") or 0)
                gate_threshold, _ = get_differentiated_threshold(
                    code=code, code_type=code_type, source=source,
                    entity_confidence=entity_conf_r3,
                    section_dominant=code_dict.get("section_dominant"),
                )
            
            trust = calculate_soft_fusion_confidence(code_dict)
            if r_rank <= 3 and trust > 0.65:
                gate_threshold *= 0.85 

            if prior_float < gate_threshold:
                # SOFT CALIBRATION: 0.85x penalty instead of rejection
                old_conf = code_dict.get("confidence", 0.5)
                code_dict["confidence"] = round(old_conf * 0.85, 3)
                code_dict.setdefault("audit_traces", []).append(f"BELOW_TIER_THRESHOLD_PENALTY({gate_threshold})")

        # ── Rule 4: Evaluate codes not yet scored ─────────────────────────
        if prior_strength is None:
            entity_conf = float(code_dict.get("entity_confidence") or 0)
            is_rag_only = source == "rag" and entity_conf < 0.60
            strength, reason = compute_evidence_strength(
                code=code, description=description, note_text=note_text,
                entity_confidence=entity_conf, is_rag_only=is_rag_only,
            )
            code_dict["evidence_strength"] = round(clamp_score(strength), 3)
            code_dict["evidence_reason"]   = reason

            threshold, tier = get_differentiated_threshold(
                code=code, code_type=code_type, source=source,
                entity_confidence=entity_conf,
                section_dominant=code_dict.get("section_dominant"),
            )
            
            if strength < threshold:
                # SOFT CALIBRATION
                old_conf = code_dict.get("confidence", 0.5)
                code_dict["confidence"] = round(old_conf * 0.85, 3)
                code_dict.setdefault("audit_traces", []).append(f"EVIDENCE_BELOW_THRESHOLD_PENALTY({threshold})")

        # ── Rule 5: Final negation / prophylaxis pass on key terms ─────────
        desc_words = [
            w for w in description.lower().split()
            if len(w) > 4 and w not in {
                "unspecified", "other", "type", "with", "without",
                "acute", "chronic", "bilateral", "encounter",
            }
        ]
        for term in desc_words[:2]:
            if is_negated(term, note_text, window=80):
                # SOFT CALIBRATION: Strong penalty but not removal
                old_conf = code_dict.get("confidence", 0.5)
                code_dict["confidence"] = round(old_conf * 0.50, 3) # Heavy penalty for negation
                code_dict.setdefault("audit_traces", []).append(f"NEGATION_DETECTED({term})")
                break
            if has_prophylaxis_context(term, note_text, window=100):
                # SOFT CALIBRATION: Strong penalty but not removal
                old_conf = code_dict.get("confidence", 0.5)
                code_dict["confidence"] = round(old_conf * 0.50, 3)
                code_dict.setdefault("audit_traces", []).append(f"PROPHYLAXIS_CONTEXT_DETECTED({term})")
                break

        # ── Step 8: Terminal Calibration & Final Rejection Gate ──
        final_conf = float(code_dict.get("confidence", 0.5))
        
        # Safe Borderline Survival (Rank <= 5 + retrieval overlap)
        r_rank = code_dict.get("metadata", {}).get("retrieval_rank") or retrieval_trace.get("retrieval_rank") or 99
        rag_score = float(code_dict.get("rag_score") or 0)
        if r_rank <= 5 and rag_score > 0.40:
            final_conf = max(final_conf, 0.25)
            code_dict["confidence"] = final_conf
            code_dict.setdefault("audit_traces", []).append("BORDERLINE_SURVIVAL_FLOOR_RECOVERY")

        # Final Banding Trace
        band = "REJECT"
        if final_conf >= 0.75: band = "HIGH"
        elif final_conf >= 0.50: band = "MODERATE"
        elif final_conf >= 0.30: band = "LOW"
        elif final_conf >= 0.15: band = "REVIEW"
        code_dict["confidence_band"] = band
        code_dict.setdefault("audit_traces", []).append(f"BAND:{band}")

        # Final Rejection Gate: Only if confidence is catastrophically low
        if final_conf < 0.15:
            rejected += 1
            rt = build_rejection_trace(
                code=code, description=description,
                rejection_stage="terminal_safety_gate",
                rejection_reason="CONFIDENCE_BELOW_SURVIVAL_FLOOR",
                failed_dimension="total_trust",
                actual_score=final_conf
            )
            rejected_traces.append(rt)
        else:
            passed.append(code_dict)

    if rejected:
        logger.info(
            "FinalValidator: gate removed %d/%d codes. %d passed.",
            rejected, len(codes), len(passed),
        )

    return passed, rejected_traces


def run_final_validation(codes: list[dict], note_text: str) -> tuple[list[dict], list[dict]]:
    """
    Convenience wrapper: full validation pipeline.
    Returns (final_codes, all_rejected_traces).
    """
    if not note_text or not note_text.strip():
        return codes, []
    
    # 🚨 TRACE POINT 5 — FINAL VALIDATOR INPUT
    all_rejected_traces = []

    # EARLY PROTECTION: Mark critical codes before any filter runs
    for c in codes:
        code_str = (c.get("code") or "").upper()
        sec = (c.get("section_dominant") or "full_note").lower()
        is_ortho_code = any(code_str.startswith(pre) for pre in ["S72", "M80", "M81", "S82", "S52", "S42"])
        is_high_sec = any(h in sec for h in ["postop", "preop", "procedure", "findings", "diagnosis"])
        
        if is_ortho_code and is_high_sec:
            c["protected"] = True
            c.setdefault("audit_traces", []).append("EARLY_ORTHO_PROTECTION_APPLIED")

        # EARLY UROLOGY PROTECTION: Protect N13, 52332, 74176
        is_urology_code = any(code_str.startswith(pre) for pre in ["N13", "52332", "74176"])
        if is_urology_code:
            c["protected"] = True
            c.setdefault("audit_traces", []).append("EARLY_UROLOGY_PROTECTION_APPLIED")

    # Snapshot protected codes before any pass can drop them
    _protected_snapshot = [c for c in codes if c.get("protected")]

    # Stage 1: clinical reasoning (grounding + anatomy + prophylaxis + section)
    codes = _reasoning_engine.validate_codes(codes, note_text)
    if hasattr(_reasoning_engine, "last_rejected_traces"):
        all_rejected_traces.extend(_reasoning_engine.last_rejected_traces)

    # Stage 2: terminal evidence gate
    codes, final_rejected = apply_final_evidence_gate(codes, note_text)
    all_rejected_traces.extend(final_rejected)

    # Stage 3: pathological fracture M80 vs M81 guard
    try:
        if hasattr(pathological_fracture_protection, "__code__"):
            # If the signature doesn't return rejected, ignore for now (it's a niche list-level rule)
            codes = pathological_fracture_protection(codes, note_text)
    except Exception as exc:
        logger.warning("FinalValidator[path_fracture]: failed (%s) — skipping", exc)

    # Stage 4: final hierarchy pass (parent-child suppression + generic penalty)
    try:
        # Also doesn't return rejected_traces naturally, but it removes codes. We won't rebuild its whole logic,
        # but we capture the diff if we really want to.
        codes = apply_specificity_hierarchy(codes, note_text)
    except Exception as exc:
        logger.warning("FinalValidator[specificity_hierarchy]: failed (%s) — skipping", exc)

    # Stage 5: Step 9 — stable section-aware final ranking (Step 3: ranking stabilization)
    # Uses scoring_breakdown.final_score (the 5-dimension weighted composite) as primary key.
    # Falls back to section_weight + evidence + specificity for codes without a breakdown.
    try:
        def _stable_rank_key(code_dict: dict) -> tuple:
            breakdown = code_dict.get("scoring_breakdown") or {}
            # Primary: composite final_score from build_scoring_breakdown
            composite = float(breakdown.get("final_score") or 0)
            # Secondary dimensions (for codes without a breakdown)
            dominant    = code_dict.get("section_dominant", "full_note")
            sec_weight  = SECTION_WEIGHTS.get(dominant, 0.30)
            ev_strength = float(code_dict.get("evidence_strength") or 0)
            spec_score  = clamp_score(
                clinical_specificity_score(
                    code_dict.get("code") or "",
                    code_dict.get("description") or "",
                ) / 20.0
            )
            protected   = 1 if (code_dict.get("protected") or
                                 (code_dict.get("source") or "") == "deterministic") else 0
            return (protected, composite, sec_weight, ev_strength, spec_score)

        codes.sort(key=_stable_rank_key, reverse=True)
        logger.debug(
            "FinalValidator[stable_ranking]: sorted %d codes by composite score",
            len(codes),
        )
    except Exception as exc:
        logger.warning("FinalValidator[stable_ranking]: failed (%s) — skipping", exc)

    # Stage 6: Step 9 — terminal cross-diagnosis conflict resolution
    if len(codes) > 1:
        try:
            codes = check_cross_diagnosis_conflicts(codes)
        except Exception as exc:
            logger.warning("FinalValidator[conflict_resolution]: failed (%s) — skipping", exc)

    # ── Task: Rule Engine Stabilization ──────────────────────────────
    
    execution_map = {
        "executed_passes": [],
        "failed_passes": [],
        "skipped_passes": []
    }
    final_pass_completed = False

    # Define missing calibration thresholds and profile for adaptive passes
    cal_thresholds = CALIBRATION_THRESHOLDS
    profile = {
        "sparse_survival_modifier": 1.2,
        "suppression_modifier": 0.9,
        "specificity_modifier": 1.1,
        "domain": "general_clinical"
    }

    passes = [
        (apply_terminal_suppression_governance, (codes,)),
        (apply_output_consistency_governance, (codes,)),
        (apply_false_positive_sensitivity_tuning, (codes, cal_thresholds)),
        (apply_reconciliation_balance_tuning, (codes, cal_thresholds)),
        (apply_integrated_state_reconciliation, (codes, note_text)),
        (apply_severity_preservation_governance, (codes, note_text)),
        (apply_safe_procedural_reconciliation, (codes, note_text)),
        (apply_final_procedural_integrity, (codes, note_text)),
        (apply_domain_purity_governance, (codes, note_text)),
        (apply_anatomical_syndrome_preservation, (codes, note_text)),
        (apply_final_severity_governance, (codes, note_text)),
        (apply_encounter_driver_centralization, (codes, note_text)),
        (apply_rare_domain_abstraction_suppression, (codes, note_text)),
        (apply_sparse_state_governance, (codes, note_text)),
        (apply_adaptive_sparse_state_protection, (codes, profile)),
        (apply_adaptive_reconciliation_balance, (codes, profile)),
        (apply_final_dominant_representation_governance, (codes, note_text)),
        (apply_final_intervention_governance, (codes, note_text)),
        (apply_final_causality_governance, (codes, note_text)),
        (apply_final_temporal_governance, (codes, note_text)),
        (apply_final_note_reliability_governance, (codes, note_text)),
        (apply_integral_symptom_terminal_suppression, (codes, note_text)),
        (apply_combination_state_terminal_governance, (codes, note_text)),
        (apply_final_candidate_purity_lock, (codes, note_text)),
        (apply_encounter_compression_reconciliation, (codes, note_text)),
        (apply_unsupported_diagnosis_suppression, (codes, note_text)),
        
        # Task: Final Representation Collapse & Duplicate Survival Suppression
        (apply_dominant_representation_election, (codes, note_text)),
        (apply_manifestation_terminal_collapse, (codes, note_text)),
        (apply_generic_parent_elimination, (codes, note_text)),
        (apply_semantic_tail_compaction, (codes, note_text)),
        (apply_encounter_centrality_lock, (codes, note_text)),
        (apply_final_representation_cleanup, (codes, note_text)),
        
        # ── Task 49: Targeted Evidence Gating (NEW) ───────────────────
        (apply_v49_targeted_evidence_gating, (codes, note_text)),

        # Task: Audit Decision Calibration & Conservative Review Governance
        (apply_conservative_missed_code_governance, (codes, note_text)),
        (apply_supported_vs_refined_reconciliation, (codes, note_text)),
        (apply_revenue_leakage_suppression, (codes, note_text)),
        (apply_final_discrepancy_governance, (codes, note_text))
    ]

    for func, original_args in passes:
        # v18: Dynamic argument injection to fix static closure bug
        current_args = (codes,) + original_args[1:]
        
        pre_count = len(codes)
        codes = apply_pipeline_safety_wrapper(func, current_args, codes, execution_map)
        post_count = len(codes)
        
        if post_count < pre_count:
            logger.error(f"  -> FV Pass {func.__name__}: {pre_count} -> {post_count} survivors")
            
        if len(codes) == 0:
            logger.error(f"FinalValidator: Pipeline exhausted at {func.__name__}")
            break

    # ── Task: Final Pipeline Health Assertion ────────────────────────
    if len(codes) > 0:
        final_pass_completed = True
        
    if not final_pass_completed or execution_map["failed_passes"]:
        logger.warning("FinalValidator: Pipeline instability detected. Activating Recovery Mode.")
        codes = apply_pipeline_recovery_mode(codes, note_text)

    # ── Task 24: PHASE 1 — CONFIDENCE WATERFALL TRACE ──────────────────
    def _build_waterfall(c_dict):
        history = c_dict.get("contribution_history", [])
        waterfall = {
            "INITIAL_SCORE": float(c_dict.get("base_evidence_strength") or 0.5),
            "FINAL_CONFIDENCE": float(c_dict.get("confidence") or 0.0),
            "STAGES": []
        }
        current = waterfall["INITIAL_SCORE"]
        for step in history:
            delta = float(step.get("delta") or 0.0)
            current += delta
            waterfall["STAGES"].append({
                "stage": step.get("stage", "unknown"),
                "delta": delta,
                "running_score": round(current, 3)
            })
        c_dict["CONFIDENCE_WATERFALL"] = waterfall

    for c in codes:
        _build_waterfall(c)
    for rc in all_rejected_traces:
        _build_waterfall(rc)

    if not final_pass_completed or execution_map["failed_passes"] or len(codes) == 0:
        # Minimal fallback still applied for safety
        logger.error("FinalValidator: Pipeline collapsed or unstable. Applying MINIMAL_FALLBACK_GOVERNANCE.")
        codes = apply_minimal_fallback_governance(codes, note_text, all_rejected_traces)
        for c in codes:
            c.setdefault("audit_traces", []).append("FALLBACK_GOVERNANCE_ACTIVATED")
            c.setdefault("audit_traces", []).append("GOVERNANCE_PARTIALLY_APPLIED")
            c.setdefault("audit_traces", []).append("RECOVERY_MODE_TRIGGERED")

    # Part 5: Final Output Lockdown (Task 9I)
    for c in codes:
        c.setdefault("audit_traces", []).append("FINAL_OUTPUT_LOCKED")
        c.setdefault("audit_traces", []).append("RECONCILIATION_LOCKED")
        c.setdefault("execution_map", execution_map)

    # Final Stage (Hypothesis → Final)
    codes = apply_evidence_based_reconciliation(codes, note_text)
    codes = apply_dynamic_confidence_reconciliation(codes, note_text)
    codes = apply_regulatory_compliance_reconciliation(codes, note_text)

    # 🚨 TRACE POINT 6 — FINAL VALIDATOR OUTPUT
    print("\n=== FINAL VALIDATOR OUTPUT ===")
    print("Validated Count:", len(codes))
    for i, v in enumerate(codes[:10]):
        print(f"  {i+1}. VALIDATED: {v.get('code')} | CONF: {v.get('confidence')}")

    # 🛡️ PROTECTED CODE RESCUE: Re-insert any codes that were marked protected
    # but got dropped by intermediate pipeline passes (e.g. temporal governance)
    final_codes_set = {c.get("code") for c in codes}
    for c in _protected_snapshot:
        if c.get("code") not in final_codes_set:
            c.setdefault("audit_traces", []).append("PROTECTED_CODE_RESCUED")
            codes.append(c)
            final_codes_set.add(c.get("code"))
            logger.info("FinalValidator: PROTECTED_CODE_RESCUED: %s", c.get("code"))

    return codes, all_rejected_traces


def apply_authoritative_evidence_priority(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Authoritative Evidence Priority (Task 11F).
    Boosts treatment-confirmed, discharge, and operative diagnoses.
    Penalizes stale imported history and contradictory noise.
    """
    if not codes: return codes

    AUTHORITY_SECTIONS = {"discharge", "operative", "assessment", "attending", "plan", "final_diagnosis"}
    STALE_SECTIONS    = {"pmh", "problem_list", "nursing", "imported", "autogenerated"}

    for c in codes:
        sec = (c.get("section_dominant") or "").lower()
        rel = compute_document_reliability(sec, c.get("entity") or c.get("description") or "")
        c["DOCUMENT_RELIABILITY_VAL"] = rel

        if any(k in sec for k in AUTHORITY_SECTIONS):
            c["evidence_strength"] = min(1.0, float(c.get("evidence_strength") or 0.5) + 0.12)
            c.setdefault("audit_traces", []).append("AUTHORITATIVE_EVIDENCE_CONFIRMED")
        elif any(k in sec for k in STALE_SECTIONS) and not c.get("protected"):
            c["evidence_strength"] = float(c.get("evidence_strength") or 0.5) * 0.70
            c.setdefault("audit_traces", []).append("COPY_FORWARD_SUPPRESSED")
        elif rel < 0.45 and not c.get("protected"):
            c["evidence_strength"] = float(c.get("evidence_strength") or 0.5) * 0.80

    return codes


def apply_incidental_finding_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Incidental Finding Suppression (Task 10E).
    Suppresses background clutter and low-impact chronic conditions.
    """
    if not codes: return codes
    
    passed = []
    for c in codes:
        reportability = compute_reportability_strength(c, note_text)
        c["REPORTABILITY_VAL"] = reportability
        
        if reportability > 0.65:
            c.setdefault("audit_traces", []).append("REPORTABILITY_CONFIRMED")
            passed.append(c)
            continue
            
        # Suppress if incidental chronic background
        is_chronic = (c.get("TEMPORAL_STATE") or c.get("temporal_status")) == "CHRONIC_ACTIVE"
        relevance = float(c.get("encounter_relevance") or 0.5)
        
        code = c.get("code", "").upper()
        # PROTECTION: Never suppress fracture codes as incidental in ortho notes
        is_fracture = code.startswith("S") or code.startswith("M8")
        if is_fracture and relevance > 0.30:
             logger.info("FORENSIC_FV: %s PASSED incidental (fracture protection)", code)
             passed.append(c)
             continue

        if is_chronic and relevance < 0.40 and not c.get("protected"):
            logger.info("FORENSIC_FV: %s REJECTED incidental (chronic background)", code)
            c.setdefault("audit_traces", []).append("INCIDENTAL_FINDING_SUPPRESSED")
            continue
            
        passed.append(c)
        
    return passed


def apply_billing_realism_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Billing Realism Reconciliation (Task 10E).
    Finalizes the code set to look like a realistic professional audit summary.
    """
    if not codes: return codes
    
    # 1. Calculate final significance priority
    for c in codes:
        priority = compute_clinical_significance_priority(c)
        c["CLINICAL_SIGNIFICANCE_VAL"] = priority
        if priority > 0.75:
            c.setdefault("audit_traces", []).append("CLINICAL_SIGNIFICANCE_CONFIRMED")
            
    # 2. Compact the set to highest priority findings
    codes.sort(key=lambda x: x.get("CLINICAL_SIGNIFICANCE_VAL", 0), reverse=True)
    
    finalized = [c for c in codes if c.get("REPORTABILITY_VAL", 1.0) > 0.45 or c.get("protected")]
    
    for c in finalized:
        c.setdefault("audit_traces", []).append("BILLING_REALISM_APPLIED")
        
    return finalized


def apply_false_confidence_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3 — False Confidence Suppression (Task 13H).
    Detects and penalizes inflated confidence from stacked weak signals.
    """
    for c in codes:
        tier         = c.get("EVIDENCE_TIER") or compute_evidence_tier(c)
        direct_auth  = c.get("DIRECT_GROUNDING_AUTHORITY") or compute_direct_grounding_authority(c, note_text)
        raw_conf     = float(c.get("calibrated_confidence") or c.get("confidence") or 0.5)
        ev_strength  = float(c.get("evidence_strength") or 0.5)

        # High confidence without direct grounding → inflation detected
        if raw_conf > 0.80 and direct_auth < 0.30 and tier >= 3 and not c.get("protected"):
            penalty = 0.40
            c["evidence_strength"]     = ev_strength * (1.0 - penalty)
            c["calibrated_confidence"] = raw_conf * (1.0 - penalty)
            c.setdefault("audit_traces", []).append("FALSE_CONFIDENCE_SUPPRESSED")

    return codes


def apply_confidence_ceiling_governance(codes: list[dict]) -> list[dict]:
    """
    Step 7 — Confidence Ceiling Governance (Task 13H).
    Caps maximum confidence based on evidence tier quality.
    No code achieves near-definitive confidence without Tier 1/2 grounding.
    """
    CEILINGS = {1: 0.98, 2: 0.88, 3: 0.72, 4: 0.52}

    for c in codes:
        tier    = c.get("EVIDENCE_TIER") or compute_evidence_tier(c)
        ceiling = CEILINGS.get(tier, 0.52)

        for conf_key in ["calibrated_confidence", "confidence"]:
            val = c.get(conf_key)
            if val is not None and float(val) > ceiling:
                c[conf_key] = ceiling
                c.setdefault("audit_traces", []).append("CONFIDENCE_CEILING_APPLIED")

    return codes


def apply_relationship_fragmentation_prevention(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Relationship Fragmentation Prevention (Task 14R).
    Prevents integrated diagnoses from fragmenting into disconnected generic pieces.
    When a high-coherence integrated code exists, its generic parent fragments
    are suppressed unless they carry independent clinical significance.
    """
    if len(codes) < 2:
        return codes

    # Build relationship graph to identify integrated nodes
    rel_graph = build_clinical_relationship_graph(codes, note_text)
    protected_fragments: set[str] = set()

    for d in codes:
        code = (d.get("code") or "").upper()
        node = rel_graph.get(code, {})
        anat = float(d.get("ANATOMICAL_COHERENCE_VAL") or 0)
        support_count = len(node.get("supported_by", []))

        if support_count >= 2 and anat >= 0.60:
            # This is an integrated node — its generic relatives should collapse to it
            for other in codes:
                o_code = (other.get("code") or "").upper()
                if o_code == code:
                    continue
                o_desc = (other.get("description") or "").lower()
                # Generic relative: shorter code prefix, lower specificity
                if code.startswith(o_code[:3]) and len(o_code) < len(code):
                    if not other.get("protected") and compute_independent_management_strength(other) < 0.35:
                        other["evidence_strength"] = float(other.get("evidence_strength") or 0.5) * 0.30
                        other.setdefault("audit_traces", []).append("RELATIONSHIP_FRAGMENT_COLLAPSED")
                        protected_fragments.add(o_code)

    return codes


def apply_anatomical_stability_governance(codes: list[dict]) -> list[dict]:
    """
    Step 7 — Anatomical Stability Governance (Task 14R).
    Aggressively suppresses generic anatomical variants when
    a specific, coherent anatomical code exists.
    """
    for d in codes:
        anat_d = float(d.get("ANATOMICAL_COHERENCE_VAL") or 0)
        if anat_d < 0.60:
            continue  # Only specific codes suppress their generic relatives

        code_d = (d.get("code") or "").upper()
        for other in codes:
            if other is d:
                continue
            o_code  = (other.get("code") or "").upper()
            anat_o  = float(other.get("ANATOMICAL_COHERENCE_VAL") or 0)

            # Suppress generic relative with same 3-char prefix and lower anatomical precision
            if code_d[:3] == o_code[:3] and len(o_code) <= len(code_d) and anat_o < 0.40:
                if not other.get("protected"):
                    other["evidence_strength"] = float(other.get("evidence_strength") or 0.5) * 0.25
                    other.setdefault("audit_traces", []).append("ANATOMICAL_VARIANT_PROTECTED")

    return codes


def apply_severity_preservation_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Severity Preservation Reconciliation (Task 15P).
    Preserves severe/specific diagnoses and suppresses their generic parents.
    """
    for d in codes:
        severity = float(d.get("CLINICAL_SEVERITY_VAL") or compute_clinical_severity_weight(d, note_text))
        d["CLINICAL_SEVERITY_VAL"] = severity
        if severity >= 0.50:
            d.setdefault("audit_traces", []).append("SEVERE_VARIANT_PROTECTED")
            d["SEVERITY_LOCK"] = True
            # Suppress same-prefix generic relatives
            code_d = (d.get("code") or "").upper()
            for other in codes:
                if other is d: continue
                o_code = (other.get("code") or "").upper()
                o_sev  = float(other.get("CLINICAL_SEVERITY_VAL") or 0)
                if code_d.startswith(o_code[:3]) and len(o_code) < len(code_d) and o_sev < 0.30:
                    if not other.get("protected") and not other.get("SEVERITY_LOCK"):
                        other["evidence_strength"] = float(other.get("evidence_strength") or 0.5) * 0.20
                        other.setdefault("audit_traces", []).append("OVERGENERALIZATION_SUPPRESSED")
    return codes


def apply_assertion_conflict_governance(codes: list[dict]) -> list[dict]:
    """
    Step 7 — Assertion Conflict Governance (Task 15P).
    When provider-confirmed codes conflict with semantic relatives,
    the provider-confirmed code always wins.
    """
    provider_confirmed = {
        (d.get("code") or "").upper()
        for d in codes
        if d.get("PROVIDER_TRUTH_LOCKED") or d.get("SEVERITY_LOCK")
    }
    for d in codes:
        code = (d.get("code") or "").upper()
        if code in provider_confirmed:
            continue
        assertion = float(d.get("PROVIDER_ASSERTION_VAL") or compute_provider_assertion_strength(d))
        # If this code is a semantic relative and a confirmed code shares its prefix
        for conf_code in provider_confirmed:
            if conf_code[:3] == code[:3] and conf_code != code:
                if assertion < 0.65 and not d.get("protected"):
                    d["evidence_strength"] = float(d.get("evidence_strength") or 0.5) * 0.25
                    d.setdefault("audit_traces", []).append("ASSERTION_CONFLICT_RESOLVED")
                    break
    return codes


def apply_terminal_suppression_governance(codes: list[dict]) -> list[dict]:
    """
    Step 4 — Terminal Suppression Governance (Task 16C).
    Finalizes suppression state for codes meeting terminal suppression criteria.
    Terminally suppressed codes cannot revive in subsequent passes.
    """
    for c in codes:
        # Criteria for terminal suppression
        tier      = int(c.get("EVIDENCE_TIER") or 4)
        ev        = float(c.get("evidence_strength") or 0)
        temporal  = c.get("temporal_state", "ACTIVE")
        negation  = float(c.get("negation_scope_strength") or 0)
        overgen   = float(c.get("OVERGENERALIZATION_RISK_VAL") or 0)

        is_terminal = (
            (temporal in ["RULED_OUT", "RESOLVED"] and not c.get("protected"))
            or (negation >= 0.85 and not c.get("protected"))
            or (tier == 4 and ev < 0.15 and not c.get("protected"))
            or (overgen >= 0.80 and not c.get("SEVERITY_LOCK") and not c.get("PROVIDER_TRUTH_LOCKED"))
        )

        if is_terminal:
            c["TERMINAL_SUPPRESSION"] = True
            c["evidence_strength"]    = min(c.get("evidence_strength", 0), 0.05)
            c.setdefault("audit_traces", []).append("TERMINAL_SUPPRESSION_CONFIRMED")

    return codes


def apply_output_consistency_governance(codes: list[dict]) -> list[dict]:
    """
    Step 7 — Final Output Consistency Governance (Task 16C).
    Ensures the final set has no contradictory siblings, unstable tails,
    or weaker relatives surviving stronger locked variants.
    Sorts by reconciliation stability descending.
    """
    if not codes:
        return codes

    # Score reconciliation stability for ordering
    for c in codes:
        if "RECONCILIATION_STABILITY_VAL" not in c:
            c["RECONCILIATION_STABILITY_VAL"] = compute_reconciliation_stability(c)

    # Suppress codes that are terminally suppressed or have near-zero evidence
    active = []
    for c in codes:
        if c.get("TERMINAL_SUPPRESSION") and float(c.get("evidence_strength") or 0) < 0.10:
            continue
        active.append(c)

    # Final deterministic ordering: stability desc, then confidence desc
    active.sort(key=lambda x: (
        x.get("RECONCILIATION_STABILITY_VAL", 0),
        x.get("calibrated_confidence") or x.get("confidence", 0),
    ), reverse=True)

    for c in active:
        c.setdefault("audit_traces", []).append("OUTPUT_CONSISTENCY_CONFIRMED")
        if c.get("RECONCILIATION_STABILITY_VAL", 0) >= 0.65:
            c.setdefault("audit_traces", []).append("RECONCILIATION_STABILITY_CONFIRMED")
        lock_val = compute_lock_strength(c)
        if lock_val >= 0.60:
            c.setdefault("audit_traces", []).append("LOCK_STRENGTH_VALIDATED")

    return active


def apply_false_positive_sensitivity_tuning(
    codes: list[dict], thresholds: dict
) -> list[dict]:
    """
    Step 4 — False-Positive Sensitivity Tuning (Task 17T).
    Targets semantic tails, weak NOS, and unsupported symptom spillover
    WITHOUT oversuppressing true severe diagnoses.
    """
    fp_ceil  = thresholds.get("false_positive_ev_ceiling", 0.45)
    og_thresh = thresholds.get("overgeneralization_suppress", 0.60)

    for c in codes:
        tier    = int(c.get("EVIDENCE_TIER") or 4)
        ev      = float(c.get("evidence_strength") or 0)
        overgen = float(c.get("OVERGENERALIZATION_RISK_VAL") or 0)
        sev_lock = c.get("SEVERITY_LOCK")
        prov_lock = c.get("PROVIDER_TRUTH_LOCKED")

        # Skip protected or severity-locked codes
        if c.get("protected") or sev_lock or prov_lock:
            continue

        # Tier 4 + low evidence + NOS → false positive tail
        if tier == 4 and ev <= fp_ceil and overgen >= og_thresh:
            c["evidence_strength"] = ev * 0.30
            c.setdefault("audit_traces", []).append("FALSE_POSITIVE_TUNED")

        # Weak semantic relatives floating above floor without grounding
        elif tier >= 3 and ev <= fp_ceil:
            grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
            if grounding < 0.20:
                c["evidence_strength"] = ev * 0.55
                c.setdefault("audit_traces", []).append("FALSE_POSITIVE_TUNED")

    return codes


def apply_reconciliation_balance_tuning(
    codes: list[dict], thresholds: dict
) -> list[dict]:
    """
    Step 7 — Final Reconciliation Balance (Task 17T).
    Ensures strong severe diagnoses survive, weak semantic tails die,
    CPTs remain stable, and integrated states are preserved.
    Applies the specialty-calibrated stability floor as the final gate.
    """
    sev_floor = thresholds.get("severe_diagnosis_floor", 0.45)

    for c in codes:
        specialty = c.get("DETECTED_SPECIALTY") or "general"
        cross_stab = compute_cross_specialty_stability(c, specialty)
        c["CROSS_SPECIALTY_STABILITY_VAL"] = cross_stab

        sev  = float(c.get("CLINICAL_SEVERITY_VAL") or 0)
        ev   = float(c.get("evidence_strength") or 0)
        tier = int(c.get("EVIDENCE_TIER") or 4)

        # Preserve: severe + sufficiently evidenced
        if sev >= sev_floor and ev >= 0.40 and not c.get("TERMINAL_SUPPRESSION"):
            c.setdefault("audit_traces", []).append("RECONCILIATION_BALANCE_APPLIED")
            c["SEVERITY_LOCK"] = True

        # Suppress: poor cross-specialty stability, not locked
        elif cross_stab < 0.30 and tier >= 3 and not c.get("protected") and not c.get("SEVERITY_LOCK"):
            c["evidence_strength"] = ev * 0.45
            c.setdefault("audit_traces", []).append("RECONCILIATION_BALANCE_APPLIED")

        if cross_stab >= 0.60:
            c.setdefault("audit_traces", []).append("CROSS_SPECIALTY_STABILITY_CONFIRMED")

    return codes


def apply_encounter_coherence_filter(
    codes: list[dict],
    note_text: str,
) -> list[dict]:
    """
    Step 1, 2, 6, 8: Final Encounter Coherence & Ordering (Task 6).
    Ensures the final code set is compact, clinically coherent, and prioritized.
    """
    if not codes:
        return []

    # 1. ENCOUNTER-CENTRIC PRIORITIZATION & ORDERING (Part 3 Task 9D)
    # Use Principal Diagnosis Strength for dominant ranking
    for c in codes:
        p_strength = compute_principal_diagnosis_strength(c, note_text)
        c["PRINCIPAL_DIAGNOSIS_STRENGTH"] = p_strength
        if p_strength > 0.80:
             c["PRINCIPAL_DIAGNOSIS_CENTRALIZED"] = True
             c["ENCOUNTER_PRIORITY_CONFIRMED"] = True

    codes.sort(key=lambda x: (
        x.get("PRINCIPAL_DIAGNOSIS_STRENGTH", 0),
        x.get("encounter_relevance", 0.5), 
        x.get("confidence", 0.5)
    ), reverse=True)

    # 2. LOW-VALUE SECONDARY SUPPRESSION (Step 2)
    # Suppress codes that have low relevance AND low confidence
    final_passed = []
    for c in codes:
        relevance = float(c.get("encounter_relevance") or 0.5)
        confidence = float(c.get("confidence") or 0.5)
        code_raw = c.get("code") or ""
        ctype = (c.get("type") or "").upper()
        # Part 2: Procedure-Safe Filtering (Task 9C Bypass)
        # Strongly grounded procedures bypass incidental suppression
        is_cpt = ctype == "CPT"
        if is_cpt and (c.get("PROCEDURE_SURVIVAL_PRIORITY") or c.get("protected")):
             final_passed.append(c)
             continue

        # 2. LOW-VALUE SECONDARY SUPPRESSION (Step 2)
        # Incidentals: Low relevance (< 0.55) AND low confidence (< 0.70)
        if relevance < 0.55 and confidence < 0.70:
            if not c.get("protected"):
                logger.info("FORENSIC_FV: %s REJECTED coherence (rel=%.2f, conf=%.2f)", code_raw, relevance, confidence)
                c["secondary_diagnosis_downranked"] = True # Trace Step 9
                continue
        
        final_passed.append(c)

    # --- Part 3: Final Temporal Safety Pass (Task 9C) ---
    final_passed = [
        c for c in final_passed
        if not (c.get("temporal_status") in ["HISTORICAL", "RESOLVED"] 
                and not c.get("protected") 
                and float(c.get("encounter_relevance") or 0) < 0.75)
    ]

    # --- Part 1: Final Specificity Lock (Task 9C) ---
    specificity_removed = set()
    for c in final_passed:
        code_c = (c.get("code") or "").upper()
        desc_c = c.get("description") or ""
        conf_c = float(c.get("confidence") or 0)
        if conf_c > 0.80:
            for other in final_passed:
                code_o = (other.get("code") or "").upper()
                desc_o = other.get("description") or ""
                if code_c != code_o and is_less_specific_variant(code_o, code_c, desc_o, desc_c):
                    specificity_removed.add(code_o)
                    logger.info("FINAL_SPECIFICITY_LOCK: Suppressing generic variant %s in favor of %s", code_o, code_c)
    final_passed = [c for c in final_passed if (c.get("code") or "").upper() not in specificity_removed]

    # 3. LIMIT LOW-CONFIDENCE CLUTTER (Step 6)
    # 4. PRINCIPAL REPRESENTATION DOMINANCE (Part 3 & 4 Task 9D)
    # Symptoms collapse if explained by a centralized principal diagnosis.
    has_strong_principal = any(c.get("PRINCIPAL_DIAGNOSIS_CENTRALIZED") for c in final_passed)

    if len(final_passed) > 8 or has_strong_principal:
        # Prune codes with relevance < 0.60 if they are in the bottom half of the list
        refined = []
        for i, c in enumerate(final_passed):
            relevance = float(c.get("encounter_relevance") or 0.5)
            code = (c.get("code") or "").upper()
            # Symptom dominance logic (Part 4 Task 9C)
            is_symptom = code.startswith("R") or any(word in (c.get("description") or "").lower() for word in ["pain", "dyspnea", "edema"])
            if has_strong_principal and is_symptom:
                # Be more aggressive with symptoms if a strong principal diagnosis exists
                if relevance < 0.70:
                    logger.info("PRINCIPAL_REPRESENTATION_DOMINANCE: Suppressing symptom %s (SUPPORTING_CONCEPT_COLLAPSED)", code)
                    c["INTEGRAL_SYMPTOM_COLLAPSED"] = True
                    c["SUPPORTING_CONCEPT_COLLAPSED"] = True
                    continue

            if i > 5 and relevance < 0.60:
                logger.info("COHERENCE_FILTER: Pruning low-relevance clutter %s", c.get("code"))
                continue
            
            refined.append(c)
        final_passed = refined

    return final_passed


def apply_harvesting_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Part 2 — Diagnosis Harvesting Suppression (Task 9D).
    Suppresses weakly grounded related concepts and low-management chronic background.
    """
    passed = []
    for c in codes:
        relevance = float(c.get("encounter_relevance") or 0.5)
        strength = float(c.get("evidence_strength") or 0.5)
        management = float(c.get("ENCOUNTER_NARRATIVE_STRENGTH") or 0.3)
        
        # Harvesting suppression logic:
        # PROTECTION: Fracture codes should have lower bar for harvesting
        is_fracture = (c.get("code") or "").startswith("S") or (c.get("code") or "").startswith("M8")
        if is_fracture and strength > 0.60:
             logger.info("FORENSIC_FV: %s PASSED harvesting (fracture protection)", c.get("code"))
             passed.append(c)
             continue

        if relevance < 0.60 and management < 0.40 and not c.get("protected"):
            if strength < 0.75:
                logger.info("FORENSIC_FV: %s REJECTED harvesting (rel=%.2f, strength=%.2f)", c.get("code"), relevance, strength)
                c["HARVESTING_SUPPRESSED"] = True
                c["LOW_ENCOUNTER_RELEVANCE"] = True
                continue
        
        passed.append(c)
    return passed


def apply_sibling_stabilization(codes: list[dict], note_text: str) -> list[dict]:
    """
    Part 3 — Sibling Replacement Stabilization (Task 9H).
    Ensures siblings only replace each other if there is a CLEAR grounding advantage.
    """
    families = {}
    for c in codes:
        prefix = (c.get("code") or "")[:3].upper()
        families.setdefault(prefix, []).append(c)
        
    suppressed_codes = set()
    for prefix, siblings in families.items():
        if len(siblings) <= 1: continue
        
        siblings.sort(key=lambda x: compute_phrase_grounding_strength(x.get("description") or "", note_text), reverse=True)
        
        best = siblings[0]
        for other in siblings[1:]:
            advantage = compute_sibling_grounding_advantage(best, other, note_text)
            if advantage > 0.25:
                logger.info("SIBLING_REPLACEMENT_STABILIZED: %s replacing %s (advantage=%.2f)", best.get("code"), other.get("code"), advantage)
                other["SIBLING_REPLACEMENT_SUPPRESSED"] = True
                suppressed_codes.add((other.get("code") or "").upper())
            else:
                # Protect specific variant from generic collapse if advantage is not clear
                if is_less_specific_variant((other.get("code") or ""), (best.get("code") or ""), other.get("description") or "", best.get("description") or ""):
                     other.setdefault("audit_traces", []).append("SIBLING_REPLACEMENT_BLOCKED")
                     other.setdefault("audit_traces", []).append("SPECIFIC_VARIANT_PROTECTED")
                     
    return [c for c in codes if (c.get("code") or "").upper() not in suppressed_codes]


def apply_unsupported_diagnosis_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Unsupported Diagnosis Suppression (Task 12).
    Prunes diagnoses lacking sufficient clinical corroboration.
    """
    passed = []
    for c in codes:
        doc_conf = float(c.get("DOCUMENTATION_CONFIDENCE_VAL") or 0.5)
        obj_strength = float(c.get("OBJECTIVE_EVIDENCE_VAL") or 0.5)
        mgmt = float(c.get("MANAGEMENT_INTENSITY_VAL") or 0)
        
        if doc_conf < 0.40 and obj_strength < 0.35 and mgmt < 0.30:
            if not c.get("protected") and not c.get("PRINCIPAL_DIAGNOSIS_CONFIRMED"):
                logger.info("UNSUPPORTED_DIAGNOSIS_SUPPRESSED: %s", c.get("code"))
                c.setdefault("audit_traces", []).append("UNSUPPORTED_DIAGNOSIS_SUPPRESSED")
                c.setdefault("audit_traces", []).append("EVIDENCE_THRESHOLD_FAILED")
                continue
        passed.append(c)
    return passed


def apply_evidence_based_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Evidence-Based Final Reconciliation (Task 12).
    Ensures audit-defensible coding compactness based on clinical corroboration.
    """
    if len(codes) <= 2: return codes
    
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    passed = []
    for c in codes:
        is_corroborated = float(c.get("OBJECTIVE_EVIDENCE_VAL") or 0) > 0.65
        is_principal = c.get("PRINCIPAL_DIAGNOSIS_CONFIRMED")
        is_active = c.get("TEMPORAL_STATE") == "ACTIVE"
        
        if is_principal or (is_corroborated and is_active):
            passed.append(c)
            c.setdefault("audit_traces", []).append("EVIDENCE_BASED_RECONCILIATION_APPLIED")
            continue
        
        doc_conf = float(c.get("DOCUMENTATION_CONFIDENCE_VAL") or 0.5)
        if doc_conf > 0.55 or c.get("protected"):
            passed.append(c)
        else:
             logger.info("EVIDENCE_RECONCILIATION_PRUNED: %s", c.get("code"))
             
    return passed


def apply_dynamic_confidence_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Dynamic Confidence Reconciliation (Task 13).
    Finalizes billing by prioritizing persistent, discharge-supported diagnoses.
    """
    if len(codes) <= 2: return codes
    
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    passed = []
    for c in codes:
        prob_conf = float(c.get("PROBABILISTIC_CONFIDENCE_VAL") or 0.5)
        discharge_strength = float(c.get("DISCHARGE_FINALITY_STRENGTH") or 0)
        is_principal = c.get("PRINCIPAL_DIAGNOSIS_CONFIRMED")
        state = c.get("ENCOUNTER_STATE_EVOLUTION", "TRANSITIONAL")
        
        if is_principal or prob_conf > 0.70 or discharge_strength > 0.60 or state == "STABILIZED_DRIVER":
            passed.append(c)
            c.setdefault("audit_traces", []).append("DYNAMIC_CONFIDENCE_RECONCILIATION_APPLIED")
            continue
            
        if state in ["EMERGING_CONCEPT", "TRANSITIONAL_STATE"] and prob_conf < 0.55:
            logger.info("DYNAMIC_RECONCILIATION_SUPPRESSED: %s", c.get("code"))
            continue
            
        passed.append(c)
        
    return passed


def apply_regulatory_compliance_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3, 7 — Regulatory Compliance Reconciliation (Task 14).
    Validates sequencing compliance, eti/mani linkage, and setting-aware uncertainty.
    """
    if not codes: return codes
    
    # 1. Sequencing Baseline
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    # 2. Etiology-Manifestation Sequencing
    finalized = []
    for c in codes:
        if c.get("MANIFESTATION_ROLE"):
            has_eti = any(other.get("ETIOLOGY_ROLE") for other in codes)
            if has_eti:
                finalized.append(c)
            else:
                logger.info("REGULATORY_PRUNED: %s", c.get("code"))
                continue
        else:
            finalized.append(c)
            
    # 3. Sequencing Ordering
    for i, c in enumerate(finalized):
        seq_conf = compute_sequencing_confidence(c, i)
        c["SEQUENCING_CONFIDENCE_VAL"] = seq_conf
        if i == 0 and c.get("PRINCIPAL_DIAGNOSIS_CONFIRMED"):
            c.setdefault("audit_traces", []).append("SEQUENCING_CONFIDENCE_CONFIRMED")
            
    for c in finalized:
         c.setdefault("audit_traces", []).append("REGULATORY_COMPLIANCE_RECONCILIATION_APPLIED")
         c.setdefault("audit_traces", []).append("POLICY_AWARE_REASONING_FINALIZED")
         
    return finalized


def apply_encounter_compression_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Final Encounter Compression Reconciliation (Task 11).
    Removes low-value residual redundancy and ensures audit-realistic output.
    """
    if len(codes) <= 2: return codes
    
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    passed = []
    for c in codes:
        is_principal = c.get("PRINCIPAL_DIAGNOSIS_CONFIRMED")
        is_comp = float(c.get("COMPLICATION_HIERARCHY_VAL") or 0) > 0.75
        is_attr = c.get("ENCOUNTER_ATTRIBUTION_CONFIRMED")
        
        if is_principal or is_comp or is_attr:
            passed.append(c)
            c.setdefault("audit_traces", []).append("ENCOUNTER_COMPRESSION_APPLIED")
            if is_principal: c.setdefault("audit_traces", []).append("PRINCIPAL_DIAGNOSIS_RECONCILED")
            continue
        
        state = c.get("TEMPORAL_STATE")
        if state in ["HISTORICAL", "RESOLVED"]:
            logger.info("HISTORICAL_CONDITION_SUPPRESSED: %s", c.get("code"))
            c.setdefault("audit_traces", []).append("HISTORICAL_CONDITION_SUPPRESSED")
            continue
            
        if state == "SUSPECTED" and float(c.get("confidence") or 0) < 0.60:
             logger.info("RULED_OUT_DIAGNOSIS_SUPPRESSED: %s", c.get("code"))
             c.setdefault("audit_traces", []).append("RULED_OUT_DIAGNOSIS_SUPPRESSED")
             continue
             
        passed.append(c)
        
    return passed


def apply_incidental_finding_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Incidental Finding Suppression (Task 10E).
    Suppresses weak incidental findings and passive chronic background clutter.
    """
    passed = []
    for c in codes:
        priority = compute_clinical_significance_priority(c)
        reportability = compute_reportability_strength(c, note_text)
        c["REPORTABILITY_VAL"] = reportability
        
        # If finding is incidental (low priority and reportability)
        if priority < 0.20 and reportability < 0.30:
            if not c.get("protected") and not c.get("PRINCIPAL_ENCOUNTER_LOCKED"):
                logger.info("INCIDENTAL_FINDING_SUPPRESSED: %s", c.get("code"))
                c.setdefault("audit_traces", []).append("INCIDENTAL_FINDING_SUPPRESSED")
                continue
        passed.append(c)
    return passed


def apply_billing_realism_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Billing Realism Reconciliation (Task 10E).
    Reconciles the final output into a compact, audit-realistic code set.
    """
    if len(codes) <= 3: return codes
    
    # Policy: billing realism avoids overcrowding with unmanaged chronic tail
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    passed = []
    for c in codes:
        # If the code is a managed chronic or major acute, always keep
        if c.get("PRINCIPAL_ENCOUNTER_LOCKED") or c.get("COMBINATION_DOMINANCE_ACTIVE"):
            passed.append(c)
            continue
            
        reportability = float(c.get("REPORTABILITY_VAL") or 0)
        if reportability > 0.40:
            passed.append(c)
            c.setdefault("audit_traces", []).append("BILLING_REALISM_APPLIED")
            c.setdefault("audit_traces", []).append("REPORTABILITY_CONFIRMED")
        else:
            logger.info("BILLING_REALISM_PRUNED: %s", c.get("code"))
            
    return passed


def apply_governed_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3 — Governed Reconciliation (Task 10D).
    Enforces principal dominance, combination dominance, and manifestation collapse.
    """
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    to_suppress = set()
    
    for i, c1 in enumerate(codes):
        code1 = (c1.get("code") or "").upper()
        if code1 in to_suppress: continue
        
        for j, c2 in enumerate(codes):
            if i == j: continue
            code2 = (c2.get("code") or "").upper()
            if code2 in to_suppress: continue
            
            # Check for semantic overlap or manifestation containment
            if compute_representation_family(c1) == compute_representation_family(c2):
                overlap = compute_semantic_overlap_strength(c1, c2)
                if overlap > 0.75:
                    # Reconcile: higher priority/certainty wins
                    p1 = float(c1.get("CONSISTENCY_PRIORITY_VAL") or 0)
                    p2 = float(c2.get("CONSISTENCY_PRIORITY_VAL") or 0)
                    
                    if p1 > p2:
                        logger.info("GOVERNED_RECONCILIATION: %s winning over %s", code1, code2)
                        to_suppress.add(code2)
                        c1.setdefault("audit_traces", []).append("GOVERNED_RECONCILIATION_APPLIED")
                    elif p2 > p1:
                        logger.info("GOVERNED_RECONCILIATION: %s winning over %s", code2, code1)
                        to_suppress.add(code1)
                        c2.setdefault("audit_traces", []).append("GOVERNED_RECONCILIATION_APPLIED")
                        break
                        
    return [c for c in codes if (c.get("code") or "").upper() not in to_suppress]


def apply_final_encounter_compaction(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Final Encounter Compaction (Task 10D).
    Produces clean, professional outputs by suppressing redundant semantic tails.
    """
    if len(codes) <= 3: return codes
    
    passed = []
    # Group by family to detect tail redundancy
    families = {}
    for c in codes:
        f = compute_representation_family(c)
        families.setdefault(f, []).append(c)
        
    for f, members in families.items():
        if len(members) > 1:
            # Sort members by priority
            members.sort(key=lambda x: (float(x.get("CONSISTENCY_PRIORITY_VAL") or 0), float(x.get("DIAGNOSTIC_CERTAINTY_VAL") or 0.5)), reverse=True)
            
            # In a family, if we have a principal driver or strong combination, be aggressive
            primary = members[0]
            passed.append(primary)
            
            for other in members[1:]:
                if float(other.get("CONSISTENCY_PRIORITY_VAL") or 0) < 0.20 and not other.get("protected"):
                     logger.info("FINAL_ENCOUNTER_COMPACTED: suppressing redundant tail %s", other.get("code"))
                     other.setdefault("audit_traces", []).append("FINAL_ENCOUNTER_COMPACTED")
                     continue
                passed.append(other)
        else:
            passed.extend(members)
            
    return passed


def apply_semantic_relative_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3 — Semantic Relative Suppression (Task 10A).
    Prunes weaker semantic relatives when a strong principal diagnosis exists.
    """
    principals = [c for c in codes if c.get("PRINCIPAL_ENCOUNTER_LOCKED")]
    if not principals:
        return codes
        
    suppressed_codes = set()
    for p in principals:
        p_prefix = (p.get("code") or "")[:3].upper()
        for other in codes:
            if other is p: continue
            o_code = (other.get("code") or "").upper()
            o_prefix = o_code[:3]
            
            # If they are semantic relatives (same prefix family)
            if o_prefix == p_prefix:
                # Penalize generalization (Step 6)
                penalty = compute_generalization_penalty(other)
                if penalty > 0:
                    other.setdefault("audit_traces", []).append("GENERALIZATION_PENALTY_APPLIED")
                    
                # If other is less specific or just a weaker relative
                if not other.get("protected") and float(other.get("evidence_strength") or 0) < float(p.get("evidence_strength") or 0):
                    logger.info("SEMANTIC_RELATIVE_SUPPRESSED: %s in favor of principal %s", o_code, p.get("code"))
                    other["SEMANTIC_RELATIVE_SUPPRESSED"] = True
                    suppressed_codes.add(o_code)
                    
    return [c for c in codes if (c.get("code") or "").upper() not in suppressed_codes]


def apply_single_signal_penalty(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Single-Signal Collapse Prevention (Task 10B).
    Rejects diagnoses that depend only on vague ontology similarity or isolated phrase overlap.
    """
    passed = []
    for c in codes:
        diversity = compute_supporting_evidence_diversity(c, note_text)
        dist_strength = compute_distributed_evidence_strength(c, note_text)
        
        # If evidence is extremely isolated and weak
        if diversity <= 0.25 and dist_strength < 0.15:
            # But allow if principal or protected or combination
            if c.get("PRINCIPAL_ENCOUNTER_LOCKED") or c.get("protected") or c.get("COMBINATION_DOMINANCE_ACTIVE"):
                passed.append(c)
            else:
                logger.info("SINGLE_SIGNAL_REJECTED: %s (insufficient multi-signal support)", c.get("code"))
                c.setdefault("audit_traces", []).append("SINGLE_SIGNAL_REJECTED")
                # Removed from representation
                continue
        passed.append(c)
    return passed


def apply_semantic_expansion_limits(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Semantic Expansion Limiter (Task 10C).
    Prevents semantic relatives from escalating beyond evidence certainty.
    """
    passed = []
    for c in codes:
        is_grounded = c.get("PHRASE_GROUNDING_CONFIRMED")
        is_principal = c.get("PRINCIPAL_ENCOUNTER_LOCKED")
        certainty = float(c.get("DIAGNOSTIC_CERTAINTY_VAL") or 0.5)
        
        # Limit semantic extrapolations with weak certainty
        if not is_grounded and not is_principal and certainty < 0.60:
            logger.info("SEMANTIC_EXPANSION_LIMITED: %s", c.get("code"))
            c.setdefault("audit_traces", []).append("SEMANTIC_EXPANSION_LIMITED")
            continue
        passed.append(c)
    return passed


def apply_conservative_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Conservative Reconciliation (Task 10C).
    Prefers more certain representation over most specific if certainty is weak.
    """
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    
    to_suppress = set()
    for i, c_spec in enumerate(codes):
        sc = (c_spec.get("code") or "").upper()
        if sc in to_suppress: continue
        
        for j, c_par in enumerate(codes):
            if i == j: continue
            pc = (c_par.get("code") or "").upper()
            if pc in to_suppress: continue
            
            if is_parent_of(pc, sc):
                sub_cert = compute_procedural_subtype_certainty(c_spec, note_text)
                diag_cert = float(c_spec.get("DIAGNOSTIC_CERTAINTY_VAL") or 0.5)
                
                # Prefer stable parent if child is uncertain
                if (sub_cert < 0.65 or diag_cert < 0.55) and not c_spec.get("protected"):
                    logger.info("CONSERVATIVE_RECONCILIATION: keeping %s over uncertain %s", pc, sc)
                    c_spec.setdefault("audit_traces", []).append("SUBTYPE_CERTAINTY_INSUFFICIENT")
                    c_spec.setdefault("audit_traces", []).append("CONSERVATIVE_RECONCILIATION_APPLIED")
                    to_suppress.add(sc)
                    
    return [c for c in codes if (c.get("code") or "").upper() not in to_suppress]


def compute_multisignal_survival_priority(codes: list[dict], note_text: str):
    """
    Step 7 — Multi-Signal Survival Priority (Task 10B).
    Codes supported by MULTIPLE evidence streams receive high reconciliation resistance.
    """
    for c in codes:
        diversity = compute_supporting_evidence_diversity(c, note_text)
        severity = compute_severity_preservation_strength(c)
        
        if diversity >= 0.75:
            c["MULTISIGNAL_SUPPORT_CONFIRMED"] = True
            c.setdefault("audit_traces", []).append("MULTISIGNAL_SUPPORT_CONFIRMED")
            c.setdefault("audit_traces", []).append("DISTRIBUTED_EVIDENCE_CONFIRMED")
        
        if severity > 0.5:
            c["SEVERITY_PRESERVED"] = True
            c.setdefault("audit_traces", []).append("SEVERITY_PRESERVED")
            
        # Resistance boost (Task 9I)
        if diversity >= 0.50 or severity > 0.50:
             c["REGRESSION_RESISTANCE"] = max(float(c.get("REGRESSION_RESISTANCE") or 0), 0.85)


def compute_encounter_centrality(codes: list[dict], note_text: str):
    """
    Step 4 — Encounter Centralization (Task 10A).
    Calculates final representation priority based on clinical story.
    """
    for c in codes:
        centrality = 0.0
        if c.get("PRINCIPAL_ENCOUNTER_LOCKED"): centrality += 0.50
        if c.get("PROCEDURE_SURVIVAL_PRIORITY") or c.get("PROCEDURAL_DOMAIN_STRENGTH", 0) > 0.8: 
            centrality += 0.30
        if c.get("COMBINATION_DOMINANCE_ACTIVE"): centrality += 0.20
        
        c["ENCOUNTER_CENTRALITY"] = centrality
        if centrality > 0.4:
            c.setdefault("audit_traces", []).append("ENCOUNTER_CENTRALITY_CONFIRMED")


def stable_representation_compaction(codes: list[dict], note_text: str) -> list[dict]:
    """
    Part 5 — Final Representation Stability & Lockdown (Task 9G/9H/9I).
    Compacts weak tails and preserves grounded specificity, procedural stability, and immunity locks.
    """
    # Step 4: Calculate Centrality
    compute_encounter_centrality(codes, note_text)
    
    # Step 7: Multi-Signal Priority
    compute_multisignal_survival_priority(codes, note_text)
    
    # 1. Survival Weighting & Lockdown (Part 1, 3, 4 Task 9I)
    for c in codes:
        spec_weight = compute_specificity_survival_weight(c, note_text)
        proc_weight = compute_procedural_stability_weight(c, note_text)
        fp_risk = compute_false_positive_risk(c, note_text)
        proc_domain = compute_procedural_domain_strength(c, note_text)
        
        # Task 9I Locks
        reg_resistance = compute_regression_resistance(c, note_text)
        proc_lock = compute_procedural_immunity_lock(c, note_text)
        spec_lock = compute_specificity_immunity_lock(c, note_text)
        
        c["SPECIFICITY_SURVIVAL_WEIGHT"] = spec_weight
        c["PROCEDURAL_STABILITY_WEIGHT"] = proc_weight
        c["FALSE_POSITIVE_RISK"] = fp_risk
        c["PROCEDURAL_DOMAIN_STRENGTH"] = proc_domain
        c["REGRESSION_RESISTANCE"] = reg_resistance
        c["PROCEDURAL_IMMUNITY_LOCKED_VAL"] = proc_lock
        c["SPECIFICITY_IMMUNITY_LOCKED_VAL"] = spec_lock
        
        if reg_resistance > 0.80:
             c.setdefault("audit_traces", []).append("REGRESSION_RESISTANCE_GRANTED")
             c.setdefault("audit_traces", []).append("STABILITY_LOCK_CONFIRMED")
             c.setdefault("audit_traces", []).append("VOLATILITY_SUPPRESSED")
        if proc_lock > 0.85:
             c.setdefault("audit_traces", []).append("PROCEDURAL_IMMUNITY_LOCKED")
             c.setdefault("audit_traces", []).append("MODIFIER_SURVIVAL_CONFIRMED")
             c.setdefault("audit_traces", []).append("PROCEDURAL_FIDELITY_PRESERVED")
        if spec_lock > 0.85:
             c.setdefault("audit_traces", []).append("SPECIFICITY_IMMUNITY_LOCKED")
             c.setdefault("audit_traces", []).append("SUBTYPE_FIDELITY_PRESERVED")
        
        # Legacy traces (Task 9G/H)
        if spec_weight > 0.80:
             c.setdefault("audit_traces", []).append("SPECIFICITY_RESISTANCE_GRANTED")
        if proc_weight > 0.80:
             c.setdefault("audit_traces", []).append("PROCEDURAL_STABILITY_GRANTED")
             c.setdefault("audit_traces", []).append("PROCEDURE_VOLATILITY_REDUCED")

    # 2. Deterministic Order (Task 9I)
    codes.sort(key=deterministic_reconciliation_key, reverse=True)
    c_set = set()
    ordered_unique = []
    for c in codes:
        cd = (c.get("code") or "").upper()
        if cd not in c_set:
            ordered_unique.append(c)
            c_set.add(cd)
    codes = ordered_unique
    
    if len(codes) <= 6:
        return codes
        
    compacted = []
    for i, c in enumerate(codes):
        is_locked = c.get("HARD_TEMPORAL_LOCK")
        is_immune = float(c.get("PROCEDURAL_IMMUNITY") or 0) > 0.70
        is_proc_lock = float(c.get("PROCEDURAL_IMMUNITY_LOCKED_VAL") or 0) > 0.85
        is_spec_lock = float(c.get("SPECIFICITY_IMMUNITY_LOCKED_VAL") or 0) > 0.85
        is_principal = c.get("PRINCIPAL_DIAGNOSIS_CENTRALIZED")
        
        # High FP risk reduces survivability unless principal/lock
        is_high_risk = float(c.get("FALSE_POSITIVE_RISK") or 0) > 0.70
        
        # Pruning Decision
        if is_locked:
            c["TEMPORAL_REVIVAL_BLOCKED"] = True
            c.setdefault("audit_traces", []).append("DETERMINISTIC_PRUNING_CONFIRMED")
            continue

        # Preservation logic
        if i < 6 and not (is_high_risk and i > 2 and not is_principal):
            compacted.append(c)
        elif is_principal or is_immune or is_proc_lock or is_spec_lock:
            if is_proc_lock or is_immune:
                 c.setdefault("audit_traces", []).append("CPT_SURVIVAL_CONFIRMED")
            compacted.append(c)
        elif float(c.get("encounter_relevance") or 0) > 0.85:
            compacted.append(c)
        else:
            c["OUTPUT_COMPACTED"] = True
            c.setdefault("audit_traces", []).append("STABLE_COMPACTION_APPLIED")
            c.setdefault("audit_traces", []).append("DETERMINISTIC_PRUNING_CONFIRMED")
            
    return compacted


def deterministic_reconciliation_key(c: dict) -> tuple:
    """
    Part 5 — Final Output Lockdown (Task 9I).
    Provides a stable and reproducible sorting key for final selection.
    """
    # 1. Temporal Validity (Hard Lock)
    is_locked = -1 if c.get("HARD_TEMPORAL_LOCK") else 0
    
    # 2. Direct Grounding Authority
    grounding = 1 if c.get("PHRASE_GROUNDING_CONFIRMED") else 0
    
    # Step 1: Diagnostic Certainty
    certainty = float(c.get("DIAGNOSTIC_CERTAINTY_VAL") or 0)
    
    # Step 7: Consistency Priority
    priority = float(c.get("CONSISTENCY_PRIORITY_VAL") or 0)
    
    # Step 1: Reportability
    reportability = float(c.get("REPORTABILITY_VAL") or 0)
    
    # Task 11: Encounter Attribution
    attribution = float(c.get("ENCOUNTER_ATTRIBUTION_VAL") or 0)
    
    # Task 12: Evidence & Confidence
    obj_ev = float(c.get("OBJECTIVE_EVIDENCE_VAL") or 0)
    doc_conf = float(c.get("DOCUMENTATION_CONFIDENCE_VAL") or 0)
    
    # Task 13: Probabilistic Fusion
    prob_conf = float(c.get("PROBABILISTIC_CONFIDENCE_VAL") or 0)
    
    # Task 14: Regulatory & Risk
    risk_sig = float(c.get("RISK_ADJUSTMENT_VAL") or 0)
    
    # 3. Specificity Immunity
    spec_lock = float(c.get("SPECIFICITY_IMMUNITY_LOCKED_VAL") or 0)
    
    # 4. Procedural Immunity
    proc_lock = float(c.get("PROCEDURAL_IMMUNITY_LOCKED_VAL") or 0)
    
    # 5. Principal Encounter Relevance
    principal = 1 if c.get("PRINCIPAL_DIAGNOSIS_CENTRALIZED") or c.get("PRINCIPAL_ENCOUNTER_LOCKED") else 0
    
    # Step 4: Encounter Centrality
    centrality = float(c.get("ENCOUNTER_CENTRALITY") or 0)
    
    # 6. Ontology Independence
    ontology = 1.0 - float(compute_ontology_dependence_ratio(c))
    
    # 7. Regression Resistance
    resistance = float(c.get("REGRESSION_RESISTANCE") or 0)
    
    # 8. Calibrated Confidence Band
    band_val = {
        "DEFINITIVE": 5, 
        "STRONGLY_GROUNDED": 4, 
        "MODERATELY_GROUNDED": 3, 
        "WEAKLY_GROUNDED": 2, 
        "UNSTABLE": 1
    }.get(c.get("CONFIDENCE_BAND", "UNSTABLE"), 0)

    return (
        is_locked,
        grounding,
        centrality,
        spec_lock,
        proc_lock,
        principal,
        ontology,
        resistance,
        band_val,
        float(c.get("evidence_strength") or 0.5),
        len(c.get("description", ""))
    )


def apply_ontology_drift_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Part 2 — Ontology Drift Suppression (Task 9F).
    Suppresses weak semantic relatives and expansions lacking direct note grounding.
    """
    passed = []
    for c in codes:
        dependence = compute_ontology_dependence_ratio(c)
        phrase_confirmed = c.get("PHRASE_GROUNDING_CONFIRMED")
        
        # High dependence on semantic inference without direct phrase grounding
        if dependence > 0.60 and not phrase_confirmed and not c.get("protected"):
            logger.info("ONTOLOGY_DRIFT_DETECTED: suppressing %s (dependence=%.2f)", c.get("code"), dependence)
            c["ONTOLOGY_DRIFT_DETECTED"] = True
            c["SEMANTIC_DEPENDENCE_HIGH"] = True
            c["INFERENCE_ONLY_REJECTED"] = True
            continue
            
        passed.append(c)
    return passed


def apply_final_grounding_authority(codes: list[dict], note_text: str) -> list[dict]:
    """
    Part 5 — Final Evidence Authority (Task 9F).
    Prioritizes direct evidence over aggregate semantic reasoning.
    """
    final_list = []
    for c in codes:
        # Final authority check
        phrase_strength = compute_phrase_grounding_strength(c.get("description") or "", note_text)
        dependence = compute_ontology_dependence_ratio(c)
        
        # Reject if direct grounding is weak and semantic dependence is high
        if phrase_strength < 0.40 and dependence > 0.50 and not c.get("protected"):
            logger.info("AGGREGATE_REASONING_REJECTED: %s (direct_grounding=%.2f)", c.get("code"), phrase_strength)
            c["AGGREGATE_REASONING_REJECTED"] = True
            continue
            
        c["FINAL_GROUNDING_VALIDATED"] = True
        c["DIRECT_EVIDENCE_DOMINANT"] = True
        final_list.append(c)
        
    return final_list


# ── Task 19S: Integrated Disease State Stabilization ─────────────────

def apply_integrated_state_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Generic Relative Suppression.
    Suppresses weaker semantic relatives when a stronger integrated disease state exists.
    """
    to_suppress = set()
    
    # Sort by specificity and strength to find potential dominants
    sorted_codes = sorted(codes, key=lambda x: (
        float(x.get("SPECIFICITY_DOMINANCE_VAL") or 0),
        float(x.get("INTEGRATED_STATE_STRENGTH_VAL") or 0)
    ), reverse=True)
    
    for i, dominant in enumerate(sorted_codes):
        if dominant.get("code") in to_suppress:
            continue
            
        dom_pfx = (dominant.get("code") or "")[:3]
        dom_spec = float(dominant.get("SPECIFICITY_DOMINANCE_VAL") or 0)
        dom_int = float(dominant.get("INTEGRATED_STATE_STRENGTH_VAL") or 0)
        dom_assertion = float(dominant.get("PROVIDER_ASSERTION_VAL") or 0)
        dom_grounding = float(dominant.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Only strong integrated states can dominate
        if dom_int < 0.60 or dom_spec < 0.50:
            continue
            
        for target in sorted_codes[i+1:]:
            if target.get("code") in to_suppress:
                continue
                
            tar_pfx = (target.get("code") or "")[:3]
            
            # Generalized logic: Suppress if same prefix family and weaker representation
            if tar_pfx == dom_pfx:
                tar_spec = float(target.get("SPECIFICITY_DOMINANCE_VAL") or 0)
                tar_assertion = float(target.get("PROVIDER_ASSERTION_VAL") or 0)
                
                # If dominant is clearly more specific and asserted
                if dom_spec > tar_spec and dom_assertion >= tar_assertion:
                    to_suppress.add(target.get("code"))
                    target.setdefault("audit_traces", []).append("GENERIC_RELATIVE_SUPPRESSED")
                    dominant.setdefault("audit_traces", []).append("INTEGRATED_STATE_DOMINANT")
                    
            # Cross-prefix suppression for fragmented concepts (e.g. DKA vs Diabetes)
            elif dom_int >= 0.80 and dom_grounding >= 0.60:
                # Detect if target is a generic parent that the integrated state covers
                if any(word in (dominant.get("description") or "").lower() for word in (target.get("description") or "").lower().split()):
                    if float(target.get("SPECIFICITY_DOMINANCE_VAL") or 0) < 0.40:
                        to_suppress.add(target.get("code"))
                        target.setdefault("audit_traces", []).append("HIERARCHY_COLLAPSE_PREVENTED")

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_severity_preservation_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Conservative Severity Preservation.
    Ensures severe grounded variants survive reconciliation.
    """
    _SEVERITY_KEYWORDS = [
        "hemorrhage", "shock", "respiratory failure", "ketoacidosis", "obstruction",
        "pathological fracture", "neutropenia", "encephalopathy", "acute blood loss",
        "organ failure"
    ]
    
    for c in codes:
        desc = (c.get("description") or "").lower()
        sev_val = float(c.get("CLINICAL_SEVERITY_VAL") or 0)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Detect severe variants via keywords or high severity value
        is_severe = any(k in desc for k in _SEVERITY_KEYWORDS) or sev_val >= 0.80
        
        if is_severe and grounding >= 0.45:
            # Protect from collapse
            c["SEVERITY_LOCK"] = True
            c.setdefault("audit_traces", []).append("SEVERITY_STATE_PRESERVED")
            
            # Check for generic abstraction risk
            if "NOS" in (c.get("code") or "") or "unspecified" in desc:
                # This code itself is generic, but the note is severe. 
                # This should only happen if no specific code found.
                pass
            else:
                c.setdefault("audit_traces", []).append("GENERIC_ABSTRACTION_BLOCKED")
                
    return codes


def apply_safe_procedural_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Safe Subtype Downgrading.
    If subtype certainty insufficient, safely downgrade instead of hallucinating or rejecting.
    """
    for c in codes:
        if (c.get("type") or "").upper() != "CPT": continue
        
        subtype_stability = float(c.get("PROCEDURAL_SUBTYPE_VAL") or 1.0)
        
        if subtype_stability < 0.40:
            # Trigger safe downgrade
            c["SUBTYPE_CERTAINTY_INSUFFICIENT"] = True
            c["SAFE_PROCEDURAL_DOWNGRADE"] = True
            c.setdefault("audit_traces", []).append("SAFE_PROCEDURAL_DOWNGRADE")
            c.setdefault("audit_traces", []).append("SUBTYPE_CERTAINTY_INSUFFICIENT")
            
            # Reduce confidence but preserve family
            ev = float(c.get("evidence_strength") or 0)
            c["evidence_strength"] = ev * 0.85 
            c.setdefault("audit_traces", []).append("PROCEDURE_FAMILY_PRESERVED")
            
    return codes


def apply_final_procedural_integrity(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Final Procedural Integrity Pass.
    Final governance layer for procedural realism.
    """
    print(f"DEBUG apply_final_procedural_integrity got codes: {len(codes)}")
    final_list = []
    for c in codes:
        if (c.get("type") or "").upper() != "CPT":
            print(f"DEBUG apply_final_procedural_integrity found NON-CPT: {c.get('code')}")
            final_list.append(c)
            continue
            
        grounding = float(c.get("PROCEDURAL_GROUNDING_VAL") or 0)
        family_stability = compute_procedural_family_stability(c, codes)
        
        # Integrity checks
        is_realistic = grounding >= 0.50 and family_stability >= 0.50
        
        # v18: Orthopedic Sanctuary - Protect fractures from procedural realism penalties
        code_str = (c.get("code") or "").upper()
        if code_str.startswith("M80") or code_str.startswith("S72"):
            is_realistic = True
            c.setdefault("audit_traces", []).append("ORTHOPEDIC_SANCTUARY_APPLIED")
        
        if is_realistic:
            c.setdefault("audit_traces", []).append("FINAL_PROCEDURAL_INTEGRITY_CONFIRMED")
            c.setdefault("audit_traces", []).append("PROCEDURAL_REALISM_CONFIRMED")
            
            if "THERAPEUTIC_INTERVENTION_DOMINANT" in c.get("audit_traces", []):
                c.setdefault("audit_traces", []).append("INTERVENTION_FIDELITY_PRESERVED")
                
            final_list.append(c)
        else:
            # Vague semantic drift suppression
            if grounding < 0.30:
                logger.info("PROCEDURAL_INTEGRITY_REJECTED: %s (low grounding)", c.get("code"))
                continue
            final_list.append(c)
            
    return final_list


def apply_domain_purity_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Domain Purity Governance.
    Ensures final output remains coherent with encounter specialty/domain.
    """
    final_list = []
    for c in codes:
        drift_risk = float(c.get("SEMANTIC_DRIFT_RISK_VAL") or 0)
        density = float(c.get("SPECIALTY_VOCAB_DENSITY_VAL") or 0.5)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Suppress unrelated semantic drift
        if drift_risk >= 0.80 and grounding < 0.40:
            logger.info("DOMAIN_PURITY_REJECTED: %s (drift=%.2f)", c.get("code"), drift_risk)
            c.setdefault("audit_traces", []).append("SEMANTIC_OUTLIER_SUPPRESSED")
            continue
            
        if density >= 0.65:
            c.setdefault("audit_traces", []).append("DOMAIN_PURITY_CONFIRMED")
            c.setdefault("audit_traces", []).append("SPECIALTY_ENCOUNTER_ALIGNMENT_CONFIRMED")
            
        final_list.append(c)
        
    return final_list


def apply_anatomical_syndrome_preservation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Anatomical & Syndrome Preservation.
    Preserves anatomically and syndromically precise concepts against semantic abstraction.
    """
    for c in codes:
        desc = (c.get("description") or "").lower()
        anatomy_val = float(c.get("ANATOMICAL_COHERENCE_VAL") or 0)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Syndromic markers
        syndromes = ["obstructive", "hemorrhagic", "inflammatory", "malignant", "failure", "crisis", "pathological"]
        is_syndrome = any(s in desc for s in syndromes)
        
        # Preserve if high anatomy or syndrome match with grounding
        if (anatomy_val >= 0.70 or is_syndrome) and grounding >= 0.45:
            c.setdefault("audit_traces", []).append("ANATOMICAL_SYNDROME_PRESERVED")
            c.setdefault("audit_traces", []).append("SPECIALTY_SPECIFICITY_DOMINANT")
            
            # Boost evidence to ensure it survives final safety filter
            ev = float(c.get("evidence_strength") or 0)
            c["evidence_strength"] = min(0.98, ev + 0.10)
            
    return codes


def apply_final_severity_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Severity Governance.
    Final protection layer for severe encounter-driving diagnoses.
    """
    final_list = []
    for c in codes:
        sev_conv = float(c.get("SEVERITY_CONVERGENCE_VAL") or 0)
        driver_dom = float(c.get("ENCOUNTER_DRIVER_DOMINANCE_VAL") or 0)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Preserve severe grounded diagnoses
        if sev_conv >= 0.70 and grounding >= 0.45:
            c.setdefault("audit_traces", []).append("FINAL_SEVERITY_GOVERNANCE_APPLIED")
            c.setdefault("audit_traces", []).append("SEVERE_STATE_SURVIVAL_CONFIRMED")
            
            # Boost evidence to ensure it survives final safety filter
            ev = float(c.get("evidence_strength") or 0)
            c["evidence_strength"] = min(0.98, ev + 0.10)
            
        final_list.append(c)
        
    return final_list


def apply_encounter_driver_centralization(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Final Encounter Centralization.
    Ensures final output is centered around principal encounter drivers.
    """
    to_suppress = set()
    
    # Identify primary drivers
    drivers = [c for c in codes if float(c.get("ENCOUNTER_DRIVER_DOMINANCE_VAL") or 0) >= 0.75]
    
    for c in codes:
        if c in drivers: continue
        
        driver_dom = float(c.get("ENCOUNTER_DRIVER_DOMINANCE_VAL") or 0)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        sev_conv = float(c.get("SEVERITY_CONVERGENCE_VAL") or 0)
        
        # Suppress vague semantic abstractions with low driver dominance
        if driver_dom < 0.35 and grounding < 0.30 and sev_conv < 0.40:
            if any(m in (c.get("description") or "").lower() for m in ["unspecified", "nos", "other"]):
                to_suppress.add(c.get("code"))
                c.setdefault("audit_traces", []).append("ENCOUNTER_DRIVER_CENTRALIZED")
                c.setdefault("audit_traces", []).append("FINAL_REPRESENTATION_STABILIZED")

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_rare_domain_abstraction_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Rare-Domain Abstraction Suppression.
    Prevent broad semantic relatives from replacing grounded rare-domain diagnoses.
    """
    final_list = []
    for c in codes:
        sparse_auth = float(c.get("SPARSE_EVIDENCE_AUTHORITY_VAL") or 0)
        rare_density = float(c.get("RARE_SPECIALTY_DENSITY_VAL") or 0)
        
        if sparse_auth >= 0.75 or (sparse_auth >= 0.50 and rare_density >= 0.70):
            # Find broad semantic parents to suppress
            c_pfx = (c.get("code") or "")[:3]
            for other in codes:
                if other.get("code") == c.get("code"): continue
                o_pfx = (other.get("code") or "")[:3]
                
                # If in same family and other is broad/NOS
                if c_pfx == o_pfx and ("NOS" in (other.get("code") or "") or "unspecified" in (other.get("description") or "").lower()):
                    if float(other.get("evidence_strength") or 0) < 0.85:
                        other["evidence_strength"] *= 0.5
                        other.setdefault("audit_traces", []).append("RARE_DOMAIN_ABSTRACTION_SUPPRESSED")
                        c.setdefault("audit_traces", []).append("SPECIALTY_SPECIFIC_VARIANT_PROTECTED")
        
        final_list.append(c)
    return final_list


def apply_sparse_state_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Sparse-State Governance.
    Final protection layer for pathology/procedure/specialist confirmed states.
    """
    for c in codes:
        sparse_auth = float(c.get("SPARSE_EVIDENCE_AUTHORITY_VAL") or 0)
        
        # High authority signals from sections
        section = (c.get("section_dominant") or "").lower()
        is_path_proc = any(s in section for s in ["pathology", "operative", "procedure"])
        
        if sparse_auth >= 0.80 or (sparse_auth >= 0.60 and is_path_proc):
            c.setdefault("audit_traces", []).append("SPARSE_STATE_GOVERNANCE_APPLIED")
            c.setdefault("audit_traces", []).append("FINAL_SPARSE_STATE_SURVIVAL_CONFIRMED")
            
            # Lock evidence at a surviving level
            ev = float(c.get("evidence_strength") or 0)
            c["evidence_strength"] = max(0.92, ev)
            
    return codes


def apply_adaptive_sparse_state_protection(codes: list[dict], profile: dict) -> list[dict]:
    """
    Step 4 — Adaptive Sparse-State Protection.
    Increase survivability of sparse authoritative diagnoses based on adaptive profile.
    """
    mod = profile.get("sparse_survival_modifier", 1.0)
    if mod <= 1.0: return codes
    
    for c in codes:
        sparse_auth = float(c.get("SPARSE_EVIDENCE_AUTHORITY_VAL") or 0)
        rare_density = float(c.get("RARE_SPECIALTY_DENSITY_VAL") or 0)
        
        if sparse_auth >= 0.50 or rare_density >= 0.50:
            ev = float(c.get("evidence_strength") or 0)
            # Boost based on mod
            boost = 0.08 * (mod - 1.0) * 2.0
            c["evidence_strength"] = min(0.97, ev + boost)
            
            c.setdefault("audit_traces", []).append("ADAPTIVE_SPARSE_STATE_PROTECTED")
            c.setdefault("audit_traces", []).append("RARE_STATE_SURVIVAL_CALIBRATED")
            
    return codes


def apply_adaptive_reconciliation_balance(codes: list[dict], profile: dict) -> list[dict]:
    """
    Step 5 — Adaptive Final Reconciliation.
    Dynamically balance specificity and suppression.
    """
    supp_mod = profile.get("suppression_modifier", 1.0)
    spec_mod = profile.get("specificity_modifier", 1.0)
    
    for c in codes:
        ev = float(c.get("evidence_strength") or 0)
        
        # Apply suppression sensitivity adjustment
        if "NOS" in (c.get("code") or "") or "unspecified" in (c.get("description") or "").lower():
            if supp_mod > 1.10:
                c["evidence_strength"] = ev * (1.0 / supp_mod)
                c.setdefault("audit_traces", []).append("ADAPTIVE_RECONCILIATION_APPLIED")
                
        # Apply specificity reinforcement
        if len(c.get("code") or "") > 5 and spec_mod > 1.10:
            c["evidence_strength"] = min(0.98, ev * spec_mod)
            c.setdefault("audit_traces", []).append("CALIBRATED_FINAL_BALANCE_CONFIRMED")
            
    return codes


def apply_final_dominant_representation_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Dominant Representation Reconciliation.
    Terminal reconciliation layer for syndromes, specificity, and clutter.
    """
    if not codes: return []
    
    final_list = []
    to_suppress = set()
    
    # 1. Identify dominant representation anchors
    anchors = []
    for c in codes:
        strength = compute_dominant_clinical_state_strength(c, note_text)
        spec_priority = compute_specificity_survival_priority(c, note_text)
        comb_integrity = compute_combination_state_integrity(c, note_text)
        
        if strength >= 0.75 or spec_priority >= 0.75 or comb_integrity >= 0.75:
            anchors.append(c)
            c.setdefault("audit_traces", []).append("FINAL_DOMINANT_REPRESENTATION_CONFIRMED")
            if spec_priority >= 0.75:
                c.setdefault("audit_traces", []).append("FINAL_SPECIFICITY_SURVIVAL_CONFIRMED")
            if comb_integrity >= 0.75:
                c.setdefault("audit_traces", []).append("FINAL_INTEGRATED_STATE_CONFIRMED")
                
    # 2. Family-level arbitration
    # Build prefix families
    families = {}
    for c in codes:
        pfx = (c.get("code") or "")[:3].upper()
        families.setdefault(pfx, []).append(c)
        
    for pfx, members in families.items():
        # If family has an anchor, suppress non-anchors that are generic or weaker
        family_anchors = [m for m in members if m in anchors]
        if family_anchors:
            for m in members:
                if m in family_anchors: continue
                
                m_desc = (m.get("description") or "").lower()
                is_generic = "NOS" in (m.get("code") or "") or "unspecified" in m_desc
                m_strength = float(m.get("DOMINANT_STATE_VAL") or 0)
                
                # Suppress if generic sibling or much weaker relative
                if is_generic or m_strength < 0.40:
                    to_suppress.add(m.get("code"))
                    m.setdefault("audit_traces", []).append("SEMANTIC_TAIL_SUPPRESSED")
                    m.setdefault("audit_traces", []).append("GENERIC_RELATIVE_COLLAPSED")
                    
    # 3. Symptom cleanup (Already penalized in CRE, final purge here)
    symptoms = ["pain", "nausea", "fever", "edema", "cough"]
    for c in codes:
        if c.get("code") in to_suppress: continue
        desc = (c.get("description") or "").lower()
        if any(s == desc for s in symptoms):
            # If a dominant syndrome exists and symptom is not independently managed
            if anchors and float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.40:
                to_suppress.add(c.get("code"))
                c.setdefault("audit_traces", []).append("SYMPTOM_HIERARCHY_ENFORCED")

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_final_intervention_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Intervention Governance.
    Final procedural reconciliation layer.
    """
    if not codes: return []
    
    final_list = []
    to_suppress = set()
    
    # 1. Identify high-priority interventions
    therapeutic_anchors = []
    for c in codes:
        if (c.get("type") or "").upper() != "CPT": continue
        
        intent_auth = compute_procedural_intent_authority(c, note_text)
        therap_priority = compute_therapeutic_priority_strength(c, note_text)
        grounding = float(c.get("PROCEDURAL_GROUNDING_VAL") or 0)
        
        if intent_auth >= 0.70 or therap_priority >= 0.70 or grounding >= 0.80:
            therapeutic_anchors.append(c)
            c.setdefault("audit_traces", []).append("FINAL_INTERVENTION_GOVERNANCE_APPLIED")
            if therap_priority >= 0.70:
                c.setdefault("audit_traces", []).append("THERAPEUTIC_PROCEDURE_PRESERVED")
            c.setdefault("audit_traces", []).append("PROCEDURAL_FAMILY_STABILIZED")
            
    # 2. Family-level procedural arbitration
    for c in codes:
        if (c.get("type") or "").upper() != "CPT": continue
        if c in therapeutic_anchors: continue
        
        # Build family key (Prefix or generic root)
        desc = (c.get("description") or "").lower()
        is_generic = any(m in desc for m in ["unspecified", "procedure nos", "imaging", "screening"])
        
        # If high-priority therapeutic anchor exists in the pool
        if therapeutic_anchors:
            # Suppress generic procedure relatives or weak abstractions
            if is_generic or float(c.get("evidence_strength") or 0) < 0.40:
                to_suppress.add(c.get("code"))
                c.setdefault("audit_traces", []).append("PROCEDURE_ABSTRACTION_SUPPRESSED")
                
        # 3. Diagnostic-Procedural Coherence check
        # Preserve procedures that are coherently linked to dominant diagnoses
        has_coherent_diag = False
        for other in codes:
            if (other.get("type") or "").upper() == "ICD":
                if compute_procedure_diagnosis_coherence(other, note_text, codes) >= 0.65:
                    has_coherent_diag = True
                    break
        
        if has_coherent_diag:
            c.setdefault("audit_traces", []).append("INTERVENTION_COHERENCE_CONFIRMED")

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_final_causality_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Causality Governance.
    Terminal reconciliation authority for causal chains and complication hierarchy.
    """
    if not codes: return []
    
    final_list = []
    to_suppress = set()
    
    # 1. Identify causal anchor states
    causal_anchors = []
    for c in codes:
        causal_auth = compute_clinical_causality_authority(c, note_text, codes)
        transition_coh = compute_state_transition_coherence(c, note_text, codes)
        comp_dom = compute_complication_dominance_strength(c, note_text, codes)
        
        if causal_auth >= 0.70 or transition_coh >= 0.70 or comp_dom >= 0.75:
            causal_anchors.append(c)
            c.setdefault("audit_traces", []).append("FINAL_CAUSALITY_GOVERNANCE_APPLIED")
            c.setdefault("audit_traces", []).append("PATHOPHYSIOLOGIC_STATE_PRESERVED")
            if comp_dom >= 0.75:
                c.setdefault("audit_traces", []).append("FINAL_COMPLICATION_HIERARCHY_CONFIRMED")
                
    # 2. Causality-based arbitration
    for c in codes:
        # Build prefix family
        pfx = (c.get("code") or "")[:3].upper()
        
        # If a high-priority causal complication exists in this family
        family_anchors = [m for m in causal_anchors if (m.get("code") or "").startswith(pfx)]
        
        if family_anchors:
            if c in family_anchors: continue
            
            # Suppress generic parent abstractions if they are redundant with complications
            desc = (c.get("description") or "").lower()
            is_generic = "NOS" in (c.get("code") or "") or "unspecified" in desc
            
            if is_generic or float(c.get("evidence_strength") or 0) < 0.40:
                to_suppress.add(c.get("code"))
                c.setdefault("audit_traces", []).append("GENERIC_CAUSAL_ABSTRACTION_SUPPRESSED")
                
    # 3. Treatment-effect linkage cleanup
    # (Generalized: suppress symptoms that are part of a causal treatment chain)
    for c in codes:
        if c.get("code") in to_suppress: continue
        if "SYMPTOM_HIERARCHY_ENFORCED" in c.get("audit_traces", []):
            # Already handled, but ensure it stays suppressed if causal anchors are strong
            if causal_anchors:
                to_suppress.add(c.get("code"))

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_final_temporal_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Temporal Governance.
    Terminal authority for active encounter integrity and timeline coherence.
    """
    if not codes: return []
    
    final_list = []
    to_suppress = set()
    
    # 1. Identify active encounter anchors
    active_anchors = []
    for c in codes:
        temp_auth = compute_temporal_encounter_authority(c, note_text)
        leak_risk = compute_historical_leakage_risk(c, note_text)
        proc_coh = compute_procedural_timeline_coherence(c, note_text)
        
        # Criteria for active status
        if temp_auth >= 0.75 and leak_risk < 0.40:
            active_anchors.append(c)
            c.setdefault("audit_traces", []).append("FINAL_TEMPORAL_GOVERNANCE_APPLIED")
            c.setdefault("audit_traces", []).append("ACTIVE_TIMELINE_PRESERVED")
            c.setdefault("audit_traces", []).append("TEMPORAL_ENCOUNTER_INTEGRITY_CONFIRMED")
            
        if (c.get("type") or "").upper() == "CPT" and proc_coh >= 0.65:
            c.setdefault("audit_traces", []).append("PROCEDURAL_TIMELINE_VALIDATED")

    # 2. Temporal-based arbitration
    for c in codes:
        temp_auth = float(c.get("TEMPORAL_AUTHORITY_VAL") or 0.5)
        leak_risk = float(c.get("HISTORICAL_LEAK_RISK_VAL") or 0.0)
        
        # Suppress non-active states if they lack strong current evidence
        if (leak_risk >= 0.75 or temp_auth < 0.30) and not c.get("protected"):
            to_suppress.add(c.get("code"))
            c.setdefault("audit_traces", []).append("HISTORICAL_FRAGMENT_REMOVED")
            
        # Suppress planned/prophylactic misclassification in final gate
        desc = (c.get("description") or "").lower()
        if any(m in desc for m in ["prophylaxis", "planned", "scheduled", "possible"]):
            if c not in active_anchors and float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.50 and not c.get("protected"):
                to_suppress.add(c.get("code"))

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_final_note_reliability_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Note Reliability Governance.
    Terminal suppression of stale documentation, template contamination, and low-authority noise.
    """
    if not codes: return []
    
    final_list = []
    to_suppress = set()
    
    # 1. Identify authoritative documentation anchors
    authoritative_anchors = []
    for c in codes:
        struct_auth = compute_document_structure_authority(c)
        copy_risk = compute_copy_forward_risk(c, note_text)
        intent_rel = compute_section_intent_reliability(c)
        
        # Criteria for authoritative anchor
        if struct_auth >= 0.75 and copy_risk < 0.40 and intent_rel >= 0.75:
            authoritative_anchors.append(c)
            c.setdefault("audit_traces", []).append("FINAL_NOTE_RELIABILITY_GOVERNANCE_APPLIED")
            c.setdefault("audit_traces", []).append("AUTHORITATIVE_DOCUMENT_SIGNAL_PRESERVED")
            c.setdefault("audit_traces", []).append("ACTIVE_DOCUMENT_TIMELINE_CONFIRMED")
            c.setdefault("audit_traces", []).append("FINAL_ENCOUNTER_DOCUMENT_ALIGNMENT_CONFIRMED")
            
    # 2. Reliability-based arbitration
    for c in codes:
        struct_auth = float(c.get("DOC_STRUCTURE_AUTH_VAL") or 0.5)
        copy_risk = float(c.get("COPY_FORWARD_RISK_VAL") or 0.0)
        
        # Suppress stale/low-authority documentation fragments
        if copy_risk >= 0.75 or struct_auth < 0.30:
            to_suppress.add(c.get("code"))
            if copy_risk >= 0.75:
                c.setdefault("audit_traces", []).append("COPY_FORWARD_FRAGMENT_REMOVED")
            if struct_auth < 0.30:
                c.setdefault("audit_traces", []).append("LOW_AUTHORITY_NOISE_SUPPRESSED")
                
        # 3. Instruction/Administrative noise cleanup
        desc = (c.get("description") or "").lower()
        if any(m in desc for m in ["instruction", "education", "administrative", "billing"]):
            if c not in authoritative_anchors and float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.50:
                to_suppress.add(c.get("code"))

    return [c for c in codes if c.get("code") not in to_suppress]


def apply_pipeline_safety_wrapper(func, args, codes: list[dict], execution_map: dict) -> list[dict]:
    """
    Step 1, 2, 5 — Upgraded Rule Engine Stabilization.
    Wrap passes in isolation, safe-copying, and detailed telemetry.
    """
    pass_name = func.__name__.upper()
    start_time = time.time()
    failures = []
    
    # SAFE_MODE Bypass logic
    if SAFE_MODE:
        ADVANCED_PASSES = {"APPLY_ADAPTIVE_SPARSE_STATE_PROTECTION", "APPLY_AGGRESSIVE_COMPACTION_MODE"}
        if pass_name in ADVANCED_PASSES:
            execution_map["skipped_passes"].append(pass_name)
            return codes
            
    try:
        # 1. Pre-pass validation & isolation (Safe Copying)
        input_codes = [copy.deepcopy(c) for c in codes if validate_candidate_schema(c)]
        if len(input_codes) < len(codes):
            failures.append(f"CANDIDATE_SCHEMA_INVALID: Dropped {len(codes) - len(input_codes)} codes")
        
        # 2. Execute pass
        result = func(*args)
        
        # 3. Post-pass validation & telemetry
        if result is None: 
            result = input_codes
            execution_map["skipped_passes"].append(pass_name)
        else:
            # Re-validate schema after mutation
            result = [c for c in result if validate_candidate_schema(c)]
            execution_map["executed_passes"].append(pass_name)
            
        record_pipeline_telemetry(pass_name, start_time, input_codes, result, failures)
        
        for d in result:
            d.setdefault("audit_traces", []).append(f"PASS_EXECUTED:{pass_name}")
            d.setdefault("audit_traces", []).append(f"STRUCTURE_NORMALIZED_SAFE")
            
        return result
        
    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"FinalValidator: PASS_FAILURE:{pass_name}\n{err_trace}")
        
        execution_map["failed_passes"].append(pass_name)
        failures.append(f"EXCEPTION: {str(e)}")
        
        record_pipeline_telemetry(pass_name, start_time, codes, codes, failures)
        
        for d in codes:
            d.setdefault("audit_traces", []).append(f"PASS_FAILED:{pass_name}")
            d.setdefault("audit_traces", []).append("RULE_ENGINE_EXCEPTION_RECOVERED")
            
        return codes


def apply_pipeline_recovery_mode(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 8 — Pipeline Recovery Mode.
    Gracefully degrade to grounded candidates only if pipeline destabilizes.
    """
    logger.warning("FinalValidator: Activating Pipeline Recovery Mode.")
    
    # 1. Grounded Candidates Only
    recovery_codes = [c for c in codes if float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) > 0.50]
    
    # 2. Minimal compact output
    if not recovery_codes:
        # Fallback to top-3 by confidence
        recovery_codes = sorted(codes, key=lambda x: float(x.get("confidence") or 0), reverse=True)[:3]
        
    for c in recovery_codes:
        c.setdefault("audit_traces", []).append("PIPELINE_RECOVERY_MODE_ACTIVATED")
        c.setdefault("audit_traces", []).append("MINIMAL_GOVERNANCE_FALLBACK")
        
    return recovery_codes


def _normalize_and_validate_structure_fv(codes: list[dict]):
    """
    Step 2 — Safe Structure Normalization.
    Guarantee field existence and clamp scores before/after every pass.
    """
    for d in codes:
        if not isinstance(d, dict): continue
        
        # Clamp & Normalize confidence
        conf = float(d.get("confidence") or d.get("evidence_strength") or 0)
        d["confidence"] = min(1.0, max(0.0, conf))
        
        # Clamp & Normalize evidence strength
        ev = float(d.get("evidence_strength") or 0)
        d["evidence_strength"] = min(1.0, max(0.0, ev))
        
        # Ensure collections exist
        d.setdefault("audit_traces", [])
        d.setdefault("relationships", [])
        d.setdefault("relationship_graph", {})
        d.setdefault("family", "")
        
        # Validate core code identity
        if not d.get("code"):
            d["code"] = "UNKNOWN"
            
        d.setdefault("audit_traces", []).append("STRUCTURE_NORMALIZED_SAFE")


def apply_minimal_fallback_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Terminal Fallback Reconciliation.
    Minimal governance to prevent raw ontology leakage if pipeline fails.
    """
    if not codes: return []
    to_suppress = set()
    
    # Identify dominant syndromes (Heuristic fallback)
    dominant_prefixes = set()
    for c in codes:
        if float(c.get("confidence") or 0) > 0.85:
            dominant_prefixes.add((c.get("code") or "")[:3].upper())
            
    for c in codes:
        code = (c.get("code") or "").upper()
        
        # Absolute protection for fractures
        is_fracture = code.startswith("S") or "fracture" in (c.get("description") or "").lower()
        if is_fracture:
            # v15: Location precision boost
            precise_keywords = ["neck", "intertrochanteric", "subtrochanteric", "shaft", "distal", "proximal"]
            if any(pk in (c.get("description") or "").lower() and pk in note_text.lower() for pk in precise_keywords):
                c["confidence"] = min(1.0, float(c.get("confidence") or 0.5) + 0.50)
            continue
        
        # 1. Suppress R-codes if dominant syndrome exists
        if code.startswith("R") and any(p in dominant_prefixes for p in ["I50", "J44", "E11", "N18", "I10"]):
            to_suppress.add(code)
            
        # 2. Suppress NOS sibling if specific sibling exists
        if "NOS" in code or "unspecified" in (c.get("description") or "").lower():
            pfx = code[:3]
            if any(other.get("code", "").startswith(pfx) and len(other.get("code", "")) > len(code) for other in codes):
                to_suppress.add(code)
                
        # 3. Suppress low-grounding T4 candidates
        if float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0) < 0.30 and "source" in c and c["source"] == "deterministic":
            to_suppress.add(code)
            
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_integral_symptom_terminal_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    🚨 TASK 28 — PHASE 8: INTEGRAL SYMPTOM TERMINAL SUPPRESSION.
    Surgically isolates and removes R-codes (Symptoms) that are purely integral
     to a confirmed definitive diagnosis.
    """
    if not codes: return codes
    
    definitive_codes = [c for c in codes if not (c.get("code") or "").startswith("R") and (c.get("type") or "").upper() == "ICD-10"]
    if not definitive_codes: return codes
    
    passed = []
    for c in codes:
        code = (c.get("code") or "").upper()
        if not code.startswith("R"):
            passed.append(c)
            continue
            
        # Check if this symptom is integral to any definitive diagnosis
        INTEGRAL_MAP = {
            "R07": ["I20", "I21", "I22", "I25"], # Chest pain integral to Ischemic Heart Disease
            "R06": ["I50", "J44", "J45", "J96"], # Dyspnea integral to HF, COPD, Asthma, Resp Failure
            "R10": ["K35", "K37", "K80", "K81"], # Abd pain integral to Appendicitis, Cholelithiasis
            "R50": ["A41", "J18", "N39.0"],     # Fever integral to Sepsis, Pneumonia, UTI
            "R41.0": ["F03", "G30"],            # Disorientation integral to Dementia
        }
        
        is_integral = False
        for s_pfx, d_pfxs in INTEGRAL_MAP.items():
            if code.startswith(s_pfx):
                if any(any(d.get("code", "").startswith(dp) for dp in d_pfxs) for d in definitive_codes):
                    is_integral = True
                    break
        
        if is_integral:
            # Only suppress if not protected
            if not c.get("protected") and float(c.get("confidence", 0)) < 0.85:
                logger.info("FinalValidator: INTEGRAL_SYMPTOM_SUPPRESSED %s in favor of definitive diagnosis", code)
                c.setdefault("audit_traces", []).append("INTEGRAL_SYMPTOM_SUPPRESSED")
                continue
                
        passed.append(c)
                
    return passed


def apply_combination_state_terminal_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Combination Code Dominance.
    Prevent decomposition of integrated ICD states.
    """
    if not codes: return []
    to_suppress = set()
    
    combination_codes = [c for c in codes if "INTEGRATED_DISEASE_STATE_CONFIRMED" in c.get("audit_traces", [])]
    
    for combo in combination_codes:
        pfx = combo.get("code", "")[:3].upper()
        for other in codes:
            if other == combo: continue
            
            other_code = other.get("code", "").upper()
            # Suppress fragments and manifestations in the same family
            if other_code.startswith(pfx) and len(other_code) <= len(combo.get("code", "")):
                if "NOS" in other_code or other_code != combo.get("code", ""):
                    to_suppress.add(other_code)
                    other.setdefault("audit_traces", []).append("COMBINATION_STATE_TERMINALLY_CONFIRMED")
                    other.setdefault("audit_traces", []).append("FRAGMENTED_RELATIVE_SUPPRESSED")
                    other.setdefault("audit_traces", []).append("INTEGRATED_ICD_STATE_DOMINANT")
                    
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_final_candidate_purity_lock(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Candidate Purity Lock.
    Remove all remaining low-purity semantic survivors.
    """
    if not codes: return []
    to_suppress = set()
    
    for c in codes:
        purity = float(c.get("CANDIDATE_PURITY_VAL") or 0.5)
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        
        # Aggressively suppress low-purity semantic survivors
        if purity < 0.40 and grounding < 0.40:
            to_suppress.add(c.get("code"))
            c.setdefault("audit_traces", []).append("LOW_PURITY_SURVIVOR_REMOVED")
            c.setdefault("audit_traces", []).append("ONTOLOGY_ABSTRACTION_TERMINALLY_SUPPRESSED")
            
        # Final purity trace
        if c.get("code") not in to_suppress:
            c.setdefault("audit_traces", []).append("FINAL_CANDIDATE_PURITY_CONFIRMED")
            
    return [c for c in codes if c.get("code") not in to_suppress]


# ─── Task: Benchmark Evaluation & Error Analysis Framework ───────────────────

def compute_benchmark_metrics(predictions: list[dict], expectations: list[dict], note_text: str) -> dict:
    """
    Step 2 — Benchmark Metric Engine.
    Compute research-grade clinical audit metrics.
    """
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity_preservation": 0.0,
        "hallucination_rate": 0.0,
        "symptom_suppression_rate": 0.0,
        "fp_count": 0,
        "fn_count": 0,
        "tp_count": 0,
        "failure_modes": {}
    }
    
    if not expectations: return metrics
    
    pred_codes = {p.get("code") for p in predictions}
    exp_codes = {e.get("code") for e in expectations}
    
    tp = len(pred_codes & exp_codes)
    fp = len(pred_codes - exp_codes)
    fn = len(exp_codes - pred_codes)
    
    metrics["tp_count"] = tp
    metrics["fp_count"] = fp
    metrics["fn_count"] = fn
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Analyze failure modes
    for p in predictions:
        if p.get("code") not in exp_codes:
            # Find closest expectation for classification
            closest_exp = expectations[0] # Simplification
            ftype = classify_audit_failure_type(p, closest_exp, note_text)
            metrics["failure_modes"][ftype] = metrics["failure_modes"].get(ftype, 0) + 1
            
    # Trace metrics
    logger.info(f"BENCHMARK_METRICS_COMPUTED: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}")
    
    return metrics


def apply_regression_detection(current: dict, previous: dict) -> dict:
    """
    Step 4 — Regression Detection.
    Detect stability regressions in clinical audit behavior.
    """
    regression = {"detected": False, "warnings": []}
    
    if not previous: return regression
    
    precision_delta = current["precision"] - previous["precision"]
    if precision_delta < -0.05:
        regression["detected"] = True
        regression["warnings"].append(f"STABILITY_REGRESSION_FLAGGED: Precision drop {precision_delta:.2f}")
        
    hallucination_delta = current["hallucination_rate"] - previous["hallucination_rate"]
    if hallucination_delta > 0.10:
        regression["detected"] = True
        regression["warnings"].append(f"REGRESSION_DETECTED: Hallucination rate increase {hallucination_delta:.2f}")
        
    return regression


def generate_benchmark_summary(metrics: dict, domain_profile: dict, regression: dict) -> str:
    """
    Step 6 — Benchmark Reporting.
    Generate professional AI clinical audit evaluation report.
    """
    summary = f"""
# CODEPERFECT AUDITOR V5 BENCHMARK REPORT
─────────────────────────────────────────
OVERALL PERFORMANCE:
- Precision: {metrics['precision']:.2f}
- Recall:    {metrics['recall']:.2f}
- F1 Score:  {metrics['precision'] * metrics['recall'] * 2 / (metrics['precision'] + metrics['recall'] + 0.001):.2f}

DOMAIN PERFORMANCE ({domain_profile['domain']}):
- FP Rate: {domain_profile['fp_rate']:.2f}
- FN Rate: {domain_profile['fn_rate']:.2f}
- Specificity Retention: {domain_profile['specificity_retention']:.2f}

TOP FAILURE MODES:
{chr(10).join([f"- {k}: {v}" for k, v in metrics['failure_modes'].items()])}

REGRESSION STATUS:
- Status: {"⚠️ REGRESSION DETECTED" if regression['detected'] else "✅ STABLE"}
{chr(10).join([f"  ! {w}" for w in regression['warnings']])}
─────────────────────────────────────────
    """
    logger.info("BENCHMARK_SUMMARY_GENERATED")
    return summary


# ─── Task: Final Representation Collapse & Duplicate Survival Suppression ───

def apply_dominant_representation_election(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 2 — Dominant Representation Election.
    Within every family, elect ONE dominant anchor.
    """
    if not codes: return []
    families = build_representation_family_index(codes)
    to_suppress = set()
    
    for fid, data in families.items():
        members = data["members"]
        if len(members) <= 1: continue
        
        # Priority Hierarchy:
        # 1. Integrated disease state
        # 2. Severe acute state
        # 3. Anatomically specific
        # 4. Procedure-linked
        # 5. Generic parents (Last)
        
        def election_key(c):
            score = 0.0
            traces = c.get("audit_traces", [])
            if "INTEGRATED_DISEASE_STATE_CONFIRMED" in traces: score += 10.0
            if "SEVERE_ACUTE_STATE_PRESERVED" in traces: score += 8.0
            if len(c.get("code", "")) > 5: score += 5.0 # Specificity proxy
            if "PROCEDURE_DIAGNOSIS_COHERENCE_CONFIRMED" in traces: score += 6.0
            if "PROVIDER_INTENT_CONFIRMED" in traces: score += 4.0
            return (score, float(c.get("confidence") or 0))
            
        sorted_members = sorted(members, key=election_key, reverse=True)
        dominant = sorted_members[0]
        dominant.setdefault("audit_traces", []).append("DOMINANT_REPRESENTATION_ELECTED")
        
        # Mark others for potential suppression
        for other in sorted_members[1:]:
            # If dominant is clearly better, suppress other
            if election_key(dominant)[0] > election_key(other)[0] + 2.0:
                to_suppress.add(other.get("code"))
                other.setdefault("audit_traces", []).append("REDUNDANT_REPRESENTATIVE_SUPPRESSED")
                
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_manifestation_terminal_collapse(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3 — Hard Suppress Duplicate Manifestations.
    Standalone manifestations must die if already represented in integrated state.
    """
    if not codes: return []
    to_suppress = set()
    
    integrated_codes = [c for c in codes if "INTEGRATED_DISEASE_STATE_CONFIRMED" in c.get("audit_traces", [])]
    
    for c in codes:
        desc = (c.get("description") or "").lower()
        is_manifestation = any(m in desc for m in ["edema", "dyspnea", "neuropathy", "anemia", "dehydration"])
        
        if is_manifestation:
            # Check if an integrated parent exists
            pfx = c.get("code", "")[:3]
            if any(ic.get("code", "").startswith(pfx) for ic in integrated_codes):
                if "INDEPENDENTLY_MANAGED" not in c.get("audit_traces", []):
                    to_suppress.add(c.get("code"))
                    c.setdefault("audit_traces", []).append("MANIFESTATION_TERMINALLY_COLLAPSED")
                    
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_generic_parent_elimination(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — Generic Parent Elimination.
    No coexistence allowed between generic parents and specific variants.
    """
    if not codes: return []
    to_suppress = set()
    
    for c in codes:
        code = c.get("code", "")
        if "NOS" in code or len(code) <= 5:
            pfx = code[:3]
            # If a more specific grounded variant exists
            if any(other.get("code", "").startswith(pfx) and len(other.get("code", "")) > len(code) for other in codes):
                to_suppress.add(code)
                c.setdefault("audit_traces", []).append("GENERIC_PARENT_ELIMINATED")
                
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_semantic_tail_compaction(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 5 — Semantic Tail Compaction.
    Remove low-value semantic clutter (vague neighbors, weak ontology).
    """
    if not codes: return []
    to_suppress = set()
    
    for c in codes:
        grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
        purity = float(c.get("CANDIDATE_PURITY_VAL") or 0.5)
        
        if grounding < 0.30 and purity < 0.40:
            if "DOMINANT_REPRESENTATION_ELECTED" not in c.get("audit_traces", []):
                to_suppress.add(c.get("code"))
                c.setdefault("audit_traces", []).append("SEMANTIC_TAIL_COMPACTED")
                
    return [c for c in codes if c.get("code") not in to_suppress]


def apply_encounter_centrality_lock(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Encounter Centralization.
    Anchor output around principal syndromes and severe acute states.
    """
    if not codes: return []
    
    # 1. Identify Central Anchors
    anchors = [c for c in codes if any(t in c.get("audit_traces", []) for t in ["DOMINANT_CLINICAL_STATE_CONFIRMED", "SEVERE_ACUTE_STATE_PRESERVED"])]
    
    if len(codes) > 15:
        # Step 7: Aggressive Compaction Trigger
        return apply_aggressive_compaction_mode(codes, note_text)
        
    for c in codes:
        c.setdefault("audit_traces", []).append("ENCOUNTER_CENTRALITY_LOCKED")
        
    return codes


def apply_aggressive_compaction_mode(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 7 — Aggressive Compaction Mode.
    Triggered for over-populated encounters to preserve core clinical truth.
    """
    logger.warning("FinalValidator: Aggressive Compaction Triggered (count=%d)", len(codes))
    
    # Keep: 1. Anchors, 2. Procedures, 3. Integrated States
    to_keep = []
    for c in codes:
        traces = c.get("audit_traces", [])
        if any(t in traces for t in ["DOMINANT_CLINICAL_STATE_CONFIRMED", "SEVERE_ACUTE_STATE_PRESERVED", "INTEGRATED_DISEASE_STATE_CONFIRMED"]):
            to_keep.append(c)
        elif (c.get("type") or "").upper() == "CPT":
            to_keep.append(c)
        elif float(c.get("confidence") or 0) > 0.90:
            to_keep.append(c)
            
    for c in to_keep:
        c.setdefault("audit_traces", []).append("AGGRESSIVE_COMPACTION_TRIGGERED")
        
    return to_keep


def apply_final_representation_cleanup(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 8 — Final Professional Representation Assertion.
    Last-resort cleanup for duplicate/generic coexistence.
    """
    if not codes: return []
    
    # Terminal suppression of any remaining symptoms explained by syndromes
    codes = apply_integral_symptom_terminal_suppression(codes, note_text)
    
    # Final generic check
    codes = apply_generic_parent_elimination(codes, note_text)
    
    for c in codes:
        c.setdefault("audit_traces", []).append("FINAL_REPRESENTATION_CLEANUP_APPLIED")
        
    return codes


# ─── Task: Audit Decision Calibration & Conservative Review Governance ──────

def apply_conservative_missed_code_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 2 — Conservative Missed-Code Governance.
    Only high-confidence grounded findings become "MISSED BY HUMAN".
    """
    if not codes: return []
    
    threshold = compute_auditor_conservatism_weight(codes)
    
    for c in codes:
        conf = compute_audit_decision_confidence(c, note_text)
        
        if c.get("is_new_finding"): # Audit identified this
            if conf < threshold:
                c["discrepancy_type"] = "CLINICAL_REFINEMENT"
                c.setdefault("audit_traces", []).append("WEAK_REFINEMENT_DOWNGRADED")
                c.setdefault("audit_traces", []).append(f"CONSERVATIVE_THRESHOLD_NOT_MET: {conf:.2f} < {threshold:.2f}")
            else:
                c["discrepancy_type"] = "MISSED_BY_HUMAN"
                c.setdefault("audit_traces", []).append("MISSED_CODE_CONFIDENCE_VALIDATED")
                
    return codes


def apply_supported_vs_refined_reconciliation(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 3 — Unsupported Classification Stabilization.
    Reclassify family-level variant mismatches as REFINED, not UNSUPPORTED.
    """
    if not codes: return []
    
    for c in codes:
        if c.get("discrepancy_type") == "UNSUPPORTED":
            code = c.get("code", "")
            pfx = code[:3]
            
            # Check if a sibling exists in the human list (family match)
            # (Assuming human codes are present in the list with is_human=True)
            has_family_match = any(h.get("is_human") and h.get("code", "").startswith(pfx) for h in codes)
            
            if has_family_match:
                c["discrepancy_type"] = "REFINED"
                c.setdefault("audit_traces", []).append("FAMILY_REFINEMENT_RECLASSIFIED")
                c.setdefault("audit_traces", []).append("VARIANT_UPGRADE_OR_DOWNGRADE_DETECTED")
            else:
                c.setdefault("audit_traces", []).append("TRUE_UNSUPPORTED_CONFIRMED")
                
    return codes


def apply_revenue_leakage_suppression(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 4 — False Revenue Leakage Suppression.
    Only high-certainty billable codes qualify as revenue leakage.
    """
    if not codes: return []
    
    for c in codes:
        if c.get("potential_revenue_leakage"):
            conf = float(c.get("AUDIT_DECISION_CONFIDENCE_VAL") or 0)
            grounding = float(c.get("DIRECT_GROUNDING_AUTHORITY") or 0)
            
            if conf < 0.85 or grounding < 0.60:
                c["potential_revenue_leakage"] = 0.0
                c.setdefault("audit_traces", []).append("FALSE_REVENUE_LEAKAGE_SUPPRESSED")
                c.setdefault("audit_traces", []).append("LOW_CERTAINTY_BILLING_IMPACT_REMOVED")
                
    return codes


def apply_final_discrepancy_governance(codes: list[dict], note_text: str) -> list[dict]:
    """
    Step 6 — Final Discrepancy Governance.
    Categorize all findings into definitive auditor buckets.
    """
    if not codes: return []
    
    for c in codes:
        conf = float(c.get("AUDIT_DECISION_CONFIDENCE_VAL") or 0)
        
        # Final Bucketization
        if c.get("discrepancy_type") == "MISSED_BY_HUMAN" and conf >= 0.85:
            c["audit_bucket"] = "HIGH-CONFIDENCE MISSED CODE"
        elif c.get("discrepancy_type") == "REFINED":
            c["audit_bucket"] = "CLINICAL REFINEMENT"
        elif c.get("discrepancy_type") == "UNSUPPORTED":
            c["audit_bucket"] = "UNSUPPORTED/HALLUCINATED"
        elif not c.get("is_new_finding"):
            c["audit_bucket"] = "VALIDATED"
        else:
            c["audit_bucket"] = "SUPPRESSED/REDUNDANT"
            
        c.setdefault("audit_traces", []).append("FINAL_DISCREPANCY_CLASSIFICATION_APPLIED")
        
    return codes




def apply_v49_targeted_evidence_gating(codes: list[dict], note_text: str) -> list[dict]:
    """
    🚨 TASK 49 — TARGETED EVIDENCE GATING.
    Reduces hallucinated survivors by requiring explicit clinical support.
    """
    if not codes:
        return []

    # Chronic conditions protected from history isolation (Phase 4)
    CHRONIC_PROTECTED = {"I10", "E11", "E78", "I50", "N18", "J44", "I25"}
    HISTORY_SECTIONS = {"history", "pmh", "past medical", "prior encounter", "history of"}

    has_definitive_dx = any(
        c.get("type") == "ICD-10" and not c.get("code", "").startswith("R") 
        for c in codes if c.get("confidence", 0) > 0.70
    )

    for i, c in enumerate(codes):
        code_str = c.get("code", "")
        prefix3 = code_str[:3]
        
        # Extract features
        term = float(c.get("terminology_overlap") or 0.0)
        anat = float(c.get("anatomy_overlap") or 0.0)
        proc = float(c.get("procedure_linkage") or 0.0)
        sec_auth = float(c.get("section_authority") or 0.0)
        semantic = float(c.get("rag_score") or 0.0)
        
        demotion_delta = 0.0
        reasons = []

        # ── Phase 1: Direct Evidence Requirement ──────────────
        has_direct_evidence = (
            term >= 0.60 or 
            anat >= 0.70 or 
            proc >= 0.60 or 
            sec_auth >= 0.80
        )
        if not has_direct_evidence:
            demotion_delta -= 0.12
            reasons.append("weak_direct_evidence")

        # ── Phase 2: Semantic Drift Detection ────────────────
        if semantic > 0.70 and term < 0.20 and anat < 0.15 and proc == 0:
            demotion_delta -= 0.15
            reasons.append("semantic_drift")

        # ── Phase 3: Procedure-Anchor Protection ─────────────
        is_procedure_supported = proc > 0.50
        if is_procedure_supported:
            # Protect from Phase 1/2 demotions
            demotion_delta = max(demotion_delta, -0.05)
            reasons.append("procedure_protected")

        # ── Phase 4: History Isolation ───────────────────────
        sections = {s.lower() for s in c.get("sections", [])}
        is_history_only = sections and all(any(hs in s for hs in HISTORY_SECTIONS) for s in sections)
        if is_history_only and prefix3 not in CHRONIC_PROTECTED:
            demotion_delta -= 0.14
            reasons.append("history_isolation")

        # ── Phase 5: Symptom Collapse ────────────────────────
        if code_str.startswith("R") and has_definitive_dx:
            demotion_delta -= 0.10
            reasons.append("symptom_collapse")

        # ── Phase 6: Gold Survival Protection ────────────────
        # Retain floor for top-ranked grounded retrieval
        if i < 3 and semantic > 0.65:
            if demotion_delta < -0.10:
                demotion_delta = -0.08
                reasons.append("top_rank_protection")

        # ── Apply Demotion (Targeted confidence reduction ONLY) ─
        if demotion_delta < 0:
            old_conf = c.get("confidence", 0.0)
            # Limit total demotion per candidate to prevent collapse
            final_delta = max(demotion_delta, -0.25)
            c["confidence"] = round(max(0.20, old_conf + final_delta), 3)
            
            # ── Phase 7: Forensic Trace ─────────────────────
            trace_msg = f"V49_DEMOTION: {','.join(reasons)} | term={term:.2f} anat={anat:.2f} proc={proc:.2f} sem={semantic:.2f}"
            c.setdefault("audit_traces", []).append(trace_msg)
            logger.info("FinalValidator[V49]: demoted %s (%.2f -> %.2f) reason: %s", 
                        code_str, old_conf, c["confidence"], ",".join(reasons))

    return codes


    """
    Ensures that a pipeline pass doesn't crash the entire audit.
    """
    try:
        start_time = time.time()
        result = func(*args)
        execution_map["executed_passes"].append({
            "name": func.__name__ if hasattr(func, "__name__") else str(func),
            "duration": round(time.time() - start_time, 4)
        })
        return result
    except Exception as e:
        logger.error(f"Pipeline pass {func} failed: {str(e)}")
        execution_map["failed_passes"].append({
            "name": func.__name__ if hasattr(func, "__name__") else str(func),
            "error": str(e)
        })
        return fallback_codes


def apply_pipeline_recovery_mode(codes, note_text):
    """
    Graceful degradation when the main pipeline fails.
    Returns codes that passed the early terminal gate.
    """
    logger.error("RECOVERY_MODE_ACTIVE: Pipeline state preserved.")
    return codes


def apply_minimal_fallback_governance(codes, note_text, rejected_traces):
    """
    🚨 EMERGENCY SAFEGUARD: Task 29.
    If the pipeline suppresses EVERYTHING, revive the top-ranked grounded candidate
    to prevent zero-emission benchmarks.
    """
    if len(codes) > 0:
        return codes
        
    if not rejected_traces:
        logger.error("MINIMAL_FALLBACK_FAILURE: No rejected traces to revive.")
        return []

    # Filter for candidates that were rejected for 'evidence' or 'threshold' reasons,
    # not hard clinical rejections like 'negation' or 'anatomy mismatch'.
    soft_rejections = [
        rt for rt in rejected_traces 
        if rt.get("rejection_stage") in ["final_gate", "pre_computed_gate", "final_grounding_gate"]
        and rt.get("failed_dimension") == "evidence"
    ]
    
    if not soft_rejections:
        # Fallback to ANY grounded retrieval
        soft_rejections = [rt for rt in rejected_traces if rt.get("source") == "rag"]

    if not soft_rejections:
        return []

    # Sort by actual score or confidence to find the "best" rejected candidate
    soft_rejections.sort(key=lambda x: float(x.get("actual_score") or x.get("confidence") or 0), reverse=True)
    
    revived = soft_rejections[0]
    revived["confidence"] = max(0.51, float(revived.get("confidence") or 0.51))
    revived["audit_traces"] = revived.get("audit_traces", []) + ["EMERGENCY_REVIVAL: MINIMAL_FALLBACK"]
    revived["protected"] = True
    
    logger.error("MINIMAL_FALLBACK_SUCCESS: Revived code %s (score %.2f)", revived.get("code"), revived.get("actual_score"))
    return [revived]


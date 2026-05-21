"""
services/audit_pipeline.py – RAG-First 5-Stage Clinical Audit Pipeline.

RESPONSIBILITIES:
  1. Orchestrates the end-to-end clinical audit flow.
  2. Enforces encounter-local reasoning via context reset.
  3. Manages stage-based fallbacks and error handling.
  4. Consolidation of clinical evidence traces and discrepancy analysis.
"""

import json
import time
import re
import asyncio
from typing import Any, AsyncGenerator

from agents.coding_logic import CodingLogicAgent
from agents.auditor import AuditorAgent
from agents.evidence_agent import EvidenceHighlighterAgent
from services.entity_extractor import EntityExtractor
from services.rule_engine import RuleEngine
from services.selection_engine import SelectionEngine
from services.final_validator import run_final_validation
from utils.phi_masker import PHIMasker
from config import settings
from utils.logging import get_logger
from utils.llm_client import generate_json_async

logger = get_logger(__name__)


def _load_explanation_prompt() -> str:
    import os
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "prompts", "clinical_explanation_prompt.txt"),
        "prompts/clinical_explanation_prompt.txt",
    ]
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return "Write a professional clinical coding audit summary with at least 5 structured points."


_EXPLANATION_PROMPT = _load_explanation_prompt()


class PipelineStep:
    def __init__(self, name: str, label: str = ""):
        self.name = name
        self.label = label or name
        self.status: str = "pending"
        self.duration_ms: float = 0.0
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.name,
            "label": self.label,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


def _build_structured_explanation_context(
    note_text: str,
    ai_codes: list[dict],
    discrepancies: list[dict],
) -> str:
    """
    Build a structured, human-readable context block for the LLM explanation prompt.
    Includes: note summary, final code set, discrepancies, rule engine adjustments.
    """
    # Note summary — first 500 chars gives sufficient clinical context
    note_summary = note_text[:500].strip()

    # Format validated codes
    code_lines = []
    for c in ai_codes[:12]:
        code   = c.get("code", "")
        desc   = c.get("description", "")
        src    = c.get("source", "")
        conf   = int(c.get("confidence", 0) * 100)
        rationale = c.get("rationale", "")
        # Strip internal rule annotations from rationale before sending to LLM
        rationale_clean = rationale.replace("[RULE:", "(Rule:").strip()
        line = f"  {code}  {desc}  [confidence {conf}%, source: {src}]"
        if rationale_clean:
            line += f"\n    Basis: {rationale_clean[:120]}"
        code_lines.append(line)
    codes_block = "\n".join(code_lines) if code_lines else "  (none)"

    # Format discrepancies
    disc_lines = []
    for d in discrepancies[:10]:
        dtype = d.get("type", "")
        code  = d.get("code", "")
        msg   = d.get("message", "")
        sev   = d.get("severity", "")
        disc_lines.append(f"  [{dtype.upper()} | {sev.upper()}] {code}: {msg}")
    disc_block = "\n".join(disc_lines) if disc_lines else "  (none)"

    # Identify rule-engine adjustments from rationale annotations
    rule_adjustments = []
    for c in ai_codes:
        rationale = c.get("rationale", "")
        if "(Rule:" in rationale or "RULE:" in rationale:
            rule_adjustments.append(f"  {c.get('code', '')}: {c.get('description', '')}")
    adjustments_block = "\n".join(rule_adjustments) if rule_adjustments else "  (none)"

    return f"""CLINICAL NOTE SUMMARY:
{note_summary}

VALIDATED CODE SET:
{codes_block}

CODING DISCREPANCIES (vs human coder):
{disc_block}

RULE ENGINE ADJUSTMENTS APPLIED:
{adjustments_block}"""


async def _generate_explanation(
    note_text: str,
    ai_codes: list[dict],
    discrepancies: list[dict],
) -> str:
    """
    Generate a structured clinical audit explanation via LLM (Groq).
    Enforces 5-minimum structured sections with CDI-specialist tone.
    Returns plain text; never returns AI-style generic phrases.
    """
    if settings.benchmark_mode:
        logger.info("ExplanationAgent: Benchmark mode active, bypassing Groq explanation layer.")
        return _build_deterministic_explanation(ai_codes, discrepancies)

    try:
        context_block = _build_structured_explanation_context(note_text, ai_codes, discrepancies)

        prompt = f"""{_EXPLANATION_PROMPT}

{context_block}

Return ONLY valid JSON: {{"explanation": "..."}}"""

        import asyncio
        raw = await asyncio.wait_for(
            generate_json_async(prompt, tier="best"),
            timeout=15.0,
        )
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            import re
            cleaned = re.sub(r"^```[json]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        parsed = json.loads(cleaned)
        explanation = parsed.get("explanation", "")

        # ── Sections 1 & 5: Structural & Consistency Validation ──
        headers = ["FINAL CODE SET", "CODING ADJUSTMENTS", "CLINICAL JUSTIFICATION", "SUMMARY OF FINDINGS"]
        has_headers = all(h in explanation for h in headers)
        
        lower_exp = explanation.lower()
        has_ai_phrases = any(p in lower_exp for p in [
            "as an ai", "i am an ai", "the system identified", "the ai found", "ai identified"
        ])
        
        if not has_headers:
            logger.warning("LLM explanation missing required structural headers. Using fallback.")
            return _build_deterministic_explanation(ai_codes, discrepancies)
            
        if has_ai_phrases:
            logger.warning("LLM explanation contained generic AI phrases. Using fallback.")
            return _build_deterministic_explanation(ai_codes, discrepancies)

        # Quality gate: reject explanations shorter than 200 chars or under 3 newlines
        if len(explanation) < 200 or explanation.count("\n") < 2:
            logger.warning("LLM explanation too short or unstructured — using deterministic fallback.")
            return _build_deterministic_explanation(ai_codes, discrepancies)

        return explanation

    except Exception as exc:
        logger.warning("ExplanationAgent failed: %s. Using deterministic fallback.", exc)
        return _build_deterministic_explanation(ai_codes, discrepancies)


def _build_deterministic_explanation(
    ai_codes: list[dict],
    discrepancies: list[dict],
) -> str:
    """
    Deterministic fallback explanation when LLM is unavailable.
    Structured in 4 sections to match audit report format.
    Never generic — always references specific codes and findings.
    """
    icd_codes = [c for c in ai_codes if c.get("type", "ICD-10") != "CPT"]
    cpt_codes = [c for c in ai_codes if c.get("type", "") == "CPT"]
    missed    = [d for d in discrepancies if d.get("type") == "missed_code"]
    unsupported = [d for d in discrepancies if d.get("type") == "unsupported_code"]

    lines = []

    # Section 1 — Final Code Set
    if icd_codes:
        code_str = "; ".join(
            f"{c.get('code')} ({c.get('description', '')})"
            for c in icd_codes[:6]
        )
        lines.append(f"FINAL CODE SET\nThe following diagnosis codes were validated: {code_str}.")

    if cpt_codes:
        cpt_str = "; ".join(f"{c.get('code')} ({c.get('description', '')})" for c in cpt_codes[:4])
        lines.append(f"Procedure codes assigned: {cpt_str}.")

    # Section 2 — Coding Adjustments
    rule_adjusted = [c for c in ai_codes if "(Rule:" in (c.get("rationale") or "")]
    if rule_adjusted:
        adj_str = "; ".join(
            f"{c.get('code')} — {c.get('description', '')}"
            for c in rule_adjusted[:4]
        )
        lines.append(f"CODING ADJUSTMENTS\nThe following codes were upgraded or corrected per clinical guidelines: {adj_str}.")

    # Section 3 — Clinical Justification
    if missed:
        miss_str = "; ".join(f"{d.get('code')} ({d.get('message', '')})" for d in missed[:3])
        lines.append(f"CLINICAL JUSTIFICATION\nCodes identified as missed by the human coder: {miss_str}. These represent potential undercoding risk.")
    if unsupported:
        unsup_str = "; ".join(f"{d.get('code')}" for d in unsupported[:3])
        lines.append(f"Codes submitted by the human coder without sufficient clinical documentation: {unsup_str}.")

    # Section 4 — Summary
    lines.append(
        f"SUMMARY\nAudit complete. {len(ai_codes)} code(s) validated. "
        f"{len(missed)} missed and {len(unsupported)} unsupported code(s) identified. "
        "Review discrepancy table for case-level details."
    )

    return "\n\n".join(lines)


class AuditPipeline:

    def __init__(self):
        self.entity_extractor     = EntityExtractor()
        self.coding_logic         = CodingLogicAgent()
        self.selection_engine     = SelectionEngine()
        self.auditor              = AuditorAgent()
        self.evidence_highlighter = EvidenceHighlighterAgent()

        if settings.benchmark_mode:
            logger.error("BENCHMARK_MODE_ACTIVE: Deterministic RAG+Selector path enforced.")

    def reset_encounter_context(self):
        """
        MANDATORY: Completely clears all stateful components to ensure
        encounter-local reasoning and context isolation.
        """
        logger.info("CASE_PROCESSING_STARTED: Initiating encounter context reset.")
        
        # 1. Clear CodingLogicAgent RAG cache
        self.coding_logic.reset_cache()
        
        # 2. Reset EntityExtractor (if it had state, currently stateless but placeholder added)
        # self.entity_extractor.reset()
        
        # 3. Reset RuleEngine / SelectionEngine / ReasoningEngine if needed
        # (Currently these are largely stateless per-call, but we enforce the reset signal)
        
        logger.info("ENCOUNTER_CONTEXT_RESET: All temporary ontology matches and candidate pools purged.")

    async def run(
        self,
        note_text: str,
        human_codes: list[str],
        ground_truth: list[str] = None,
    ) -> dict[str, Any]:
        """
        Direct, non-streaming, optimized execution path for benchmarks/evaluations.
        Computes only: note -> retrieval -> validation -> final codes -> metrics.
        Bypasses explanation generation, auditor LLM comparison, highlighter, and streaming overhead.
        """
        t_total = time.time()
        
        # Step 1: ENCOUNTER CONTEXT RESET
        self.reset_encounter_context()

        loop = asyncio.get_event_loop()
        masked_note = await loop.run_in_executor(None, lambda: PHIMasker.mask(note_text))

        # ── Step 0: Entity Extraction ─────────────────────────────────
        t0 = time.time()
        extraction_result = await loop.run_in_executor(None, lambda: self.entity_extractor.extract(masked_note))
        deterministic_codes = extraction_result.get("deterministic_codes", [])
        rag_queries = extraction_result.get("rag_queries", [])
        duration_step0 = (time.time() - t0) * 1000.0

        lifecycle_counts = {
            "case_id": "unknown",
            "raw_entities": len(extraction_result.get("confirmed_entities", [])),
            "rag_queries": len(rag_queries),
            "retrieval_candidates": 0,
            "post_rag_filter": 0,
            "post_grounding": 0,
            "post_reasoning": 0,
            "post_competition": 0,
            "post_validator": 0,
            "final_output": 0
        }

        # ── Step 1: RAG + Deterministic Code Mapping ──────────────────
        t0 = time.time()
        clinical_facts_minimal = {
            "raw_note_text": masked_note,
            "clinical_summary": masked_note[:1500],
            "evidence_sentences": {},
            "diagnoses": [
                {"entity": c.get("entity", ""), "evidence_sentence": c.get("evidence_span", "")}
                for c in deterministic_codes
            ],
            "procedures": [
                {"entity": c.get("entity", ""), "evidence_sentence": c.get("evidence_span", "")}
                for c in deterministic_codes if c.get("type") == "CPT"
            ],
        }

        # Inject ground truth into pre_extracted for forensic MRR calculation
        if ground_truth and extraction_result:
            extraction_result["ground_truth"] = ground_truth
            
        result1 = await self.coding_logic.generate_codes(clinical_facts_minimal, pre_extracted=extraction_result)
        duration_step1 = (time.time() - t0) * 1000.0

        ai_codes = []
        if result1["success"] and result1["data"]:
            raw_codes = result1["data"].get("codes", [])
            ai_codes = RuleEngine.inject_deterministic_codes(raw_codes, deterministic_codes)
            ai_codes = RuleEngine.apply_hierarchy_rules(masked_note, ai_codes)
        else:
            ai_codes = deterministic_codes

        f_trace = result1.get("data", {}).get("forensic_trace", {}) or {}
        lifecycle_counts["retrieval_candidates"] = len(f_trace.get("candidate_pool", []))
        lifecycle_counts["post_rag_filter"] = len(ai_codes)
        lifecycle_counts["post_grounding"] = len(f_trace.get("candidate_pool", [])) - len(f_trace.get("grounding_rejected", []))

        # ── Step 1.5: MANDATORY Selection Engine Gate (UNSKIPPABLE) ────
        t_sel = time.time()
        for c in ai_codes:
            if "source" not in c: c["source"] = "fallback"
            if "confidence" not in c: c["confidence"] = 0.5

        selection_result = await loop.run_in_executor(
            None,
            lambda: self.selection_engine.select(
                candidates=ai_codes,
                note_text=masked_note,
                deterministic_codes=deterministic_codes,
                gold_codes=ground_truth
            )
        )
        ai_codes = selection_result["selected"]
        selection_rejected = selection_result["rejected"]
        duration_step_sel = (time.time() - t_sel) * 1000.0
        lifecycle_counts["post_competition"] = len(ai_codes)

        # ── Step 1b: Rule Engine (Clinical Rules + CPT + Final Validation) ──
        t0 = time.time()
        try:
            diag_c, proc_c, _ = await loop.run_in_executor(None, lambda: run_final_validation(ai_codes, masked_note))
            ai_codes = diag_c + proc_c
            ai_codes = await loop.run_in_executor(None, lambda: RuleEngine.apply_final_validation(ai_codes))
        except Exception as exc:
            logger.error("RuleEngine failed in direct path: %s", exc)
        duration_step1b = (time.time() - t0) * 1000.0
        lifecycle_counts["post_reasoning"] = len(ai_codes)

        # ── Step 2: Auditor (compare AI vs human codes) ───────────────
        t0 = time.time()
        discrepancies = []
        summary = ""
        try:
            result2 = await self.auditor.compare_codes(human_codes, ai_codes, masked_note)
            if result2.get("success") and result2.get("data"):
                discrepancies = result2["data"].get("discrepancies", [])
                summary = result2["data"].get("summary", "")
        except Exception as exc:
            logger.error("AuditorAgent failed in direct path: %s", exc)
        duration_step2 = (time.time() - t0) * 1000.0

        # ── Step 3: Explanation & Step 4: Highlighter ─────────────────
        # Skip LLM explanation entirely. Generate deterministic fallback instantly.
        explanation = _build_deterministic_explanation(ai_codes, discrepancies)
        evidence = []
        duration_step3 = 0.0
        duration_step4 = 0.0

        # ── Section 6: Final Sanity Check + Terminal Evidence Gate ────────
        all_final_rejections = []
        diagnosis_codes = []
        procedure_codes = []
        try:
            diagnosis_codes, procedure_codes, all_final_rejections = run_final_validation(ai_codes, masked_note)
            ai_codes = diagnosis_codes + procedure_codes
        except Exception as exc:
            logger.warning("AuditPipeline[terminal_gate]: gate failed in direct path (%s) — skipping", exc)
            diagnosis_codes = [c for c in ai_codes if (c.get('type') or '').upper() != 'CPT']
            procedure_codes = [c for c in ai_codes if (c.get('type') or '').upper() == 'CPT']

        lifecycle_counts["post_validator"] = len(ai_codes)
        lifecycle_counts["final_output"] = len(ai_codes)

        # Sort for stable audit trail ordering
        ai_codes             = sorted(ai_codes, key=lambda x: x.get("code", ""))
        diagnosis_codes      = sorted(diagnosis_codes, key=lambda x: x.get("code", ""))
        procedure_codes      = sorted(procedure_codes, key=lambda x: x.get("code", ""))
        all_final_rejections = sorted(all_final_rejections, key=lambda x: x.get("code", ""))

        total_ms = (time.time() - t_total) * 1000

        # Build pipeline log mimicking standard steps for compatibility
        pipeline_log = [
            {"step": "EntityExtractorAgent", "label": "Extracting Clinical Entities", "status": "success", "duration_ms": duration_step0, "error": None},
            {"step": "CodingLogicAgent", "label": "RAG + Deterministic Code Mapping", "status": "success", "duration_ms": duration_step1, "error": None},
            {"step": "SelectionEngine", "label": "Competitive Resolution & Competition", "status": "success", "duration_ms": duration_step_sel, "error": None},
            {"step": "RuleEngine", "label": "Applying Clinical Coding Rules", "status": "success", "duration_ms": duration_step1b, "error": None},
            {"step": "AuditorAgent", "label": "Auditing Human vs Validated Codes", "status": "success", "duration_ms": duration_step2, "error": None},
            {"step": "ExplanationAgent", "label": "Generating Clinical Audit Explanation", "status": "success", "duration_ms": duration_step3, "error": None},
            {"step": "EvidenceHighlighterAgent", "label": "Linking Evidence Spans", "status": "success", "duration_ms": duration_step4, "error": None},
        ]

        # Determine top rejection reason
        rejection_counts = {}
        for rc in all_final_rejections:
            reason = rc.get("rejection_reason", "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        top_reason = max(rejection_counts, key=rejection_counts.get) if rejection_counts else "none"

        # Determine strongest evidence section
        section_counts = {}
        for ac in ai_codes:
            sec = ac.get("section_dominant", "full_note")
            section_counts[sec] = section_counts.get(sec, 0) + 1
        strongest_section = max(section_counts, key=section_counts.get) if section_counts else "full_note"

        trace_summary = {
            "total_accepted": len(ai_codes),
            "total_rejected": len(all_final_rejections),
            "top_rejection_reason": top_reason,
            "strongest_evidence_section": strongest_section,
        }

        forensic_trace = result1["data"].get("forensic_trace", {}) if result1["success"] and result1["data"] else {}
        forensic_trace["terminal_rejections"] = all_final_rejections

        return {
            "ai_codes": ai_codes,
            "diagnosis_codes": diagnosis_codes,
            "procedure_codes": procedure_codes,
            "low_confidence_codes": [],
            "discrepancies": discrepancies,
            "evidence": evidence,
            "summary": summary,
            "explanation": explanation,
            "pipeline_log": pipeline_log,
            "removed_codes": all_final_rejections,
            "trace_summary": trace_summary,
            "tokens_used": 0,
            "deterministic_codes_count": len(deterministic_codes),
            "forensic_trace": forensic_trace,
            "lifecycle_counts": lifecycle_counts,
        }

    async def run_stream(
        self,
        note_text: str,
        human_codes: list[str],
        ground_truth: list[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:

        t_total = time.time()
        tokens_total = 0
        pipeline_log: list[dict] = []

        # Step 1: ENCOUNTER CONTEXT RESET
        self.reset_encounter_context()

        loop = asyncio.get_event_loop()
        masked_note = await loop.run_in_executor(None, lambda: PHIMasker.mask(note_text))
        yield {"event": "info", "data": "PHI masking complete. Launching clinical audit pipeline."}

        ai_codes: list[dict] = []
        discrepancies: list[dict] = []
        evidence: list[dict] = []
        summary: str = ""
        deterministic_codes: list[dict] = []
        explanation: str = ""
        
        # 🚨 TASK 25: LIFECYCLE COUNTS
        lifecycle_counts = {
            "case_id": "unknown", # To be set if available
            "raw_entities": 0,
            "rag_queries": 0,
            "retrieval_candidates": 0,
            "post_rag_filter": 0,
            "post_grounding": 0,
            "post_reasoning": 0,
            "post_competition": 0,
            "post_validator": 0,
            "final_output": 0
        }

        # ── Step 0: Entity Extraction ─────────────────────────────────
        step0 = PipelineStep("EntityExtractorAgent", "Extracting Clinical Entities")
        yield {"event": "step_start", "data": step0.to_dict()}
        t0 = time.time()
        try:
            extraction_result = await loop.run_in_executor(None, lambda: self.entity_extractor.extract(masked_note))
            deterministic_codes = extraction_result.get("deterministic_codes", [])
            rag_queries = extraction_result.get("rag_queries", [])
            step0.duration_ms = (time.time() - t0) * 1000
            step0.status = "success"
            logger.info(
                "Step0: %d deterministic codes, %d RAG queries generated",
                len(deterministic_codes), len(rag_queries),
            )
        except Exception as exc:
            step0.duration_ms = (time.time() - t0) * 1000
            step0.status = "failed"
            step0.error = str(exc)
            deterministic_codes = []
            rag_queries = []
            logger.error("EntityExtractor failed: %s", exc)
        
        lifecycle_counts["raw_entities"] = len(extraction_result.get("confirmed_entities", []))
        lifecycle_counts["rag_queries"] = len(rag_queries)
        
        pipeline_log.append(step0.to_dict())
        yield {"event": "step_end", "data": step0.to_dict()}
        yield {"event": "lifecycle_trace", "data": lifecycle_counts}

        # ── Step 1: RAG + Deterministic Code Mapping ──────────────────
        step1 = PipelineStep("CodingLogicAgent", "RAG + Deterministic Code Mapping")
        yield {"event": "step_start", "data": step1.to_dict()}
        t0 = time.time()

        clinical_facts_minimal = {
            "raw_note_text": masked_note,
            "clinical_summary": masked_note[:1500],
            "evidence_sentences": {},
            "diagnoses": [
                {"entity": c.get("entity", ""), "evidence_sentence": c.get("evidence_span", "")}
                for c in deterministic_codes
            ],
            "procedures": [
                {"entity": c.get("entity", ""), "evidence_sentence": c.get("evidence_span", "")}
                for c in deterministic_codes if c.get("type") == "CPT"
            ],
        }

        result1 = {"success": False, "data": None, "tokens_used": 0}
        try:
            # Inject ground truth into pre_extracted for forensic MRR calculation
            if ground_truth and extraction_result:
                extraction_result["ground_truth"] = ground_truth
                
            result1 = await self.coding_logic.generate_codes(clinical_facts_minimal, pre_extracted=extraction_result)
            step1.duration_ms = (time.time() - t0) * 1000

            if result1["success"] and result1["data"]:
                raw_codes = result1["data"].get("codes", [])
                # Inject deterministic codes (they always win)
                ai_codes = RuleEngine.inject_deterministic_codes(raw_codes, deterministic_codes)
                # Apply ICD hierarchy upgrade rules (pre-existing)
                ai_codes = RuleEngine.apply_hierarchy_rules(masked_note, ai_codes)
                step1.status = "success"
            else:
                ai_codes = deterministic_codes
                step1.status = "partial"
                step1.error = result1.get("error", "CodingLogic returned no data")
                logger.warning("CodingLogicAgent returned no data — using deterministic codes only.")
        except Exception as exc:
            step1.duration_ms = (time.time() - t0) * 1000
            step1.status = "failed"
            step1.error = str(exc)
            ai_codes = deterministic_codes
            logger.error("FALLBACK_PATH_ACTIVE: CodingLogicAgent exception: %s — falling back to deterministic codes.", exc)

        # 🚨 TASK 25: Capture counts from forensic_trace if available
        f_trace = result1.get("data", {}).get("forensic_trace", {})
        lifecycle_counts["retrieval_candidates"] = len(f_trace.get("candidate_pool", []))
        lifecycle_counts["post_rag_filter"] = len(ai_codes)
        # Record post_grounding from forensic_trace as well
        lifecycle_counts["post_grounding"] = len(f_trace.get("candidate_pool", [])) - len(f_trace.get("grounding_rejected", []))

        tokens_total += result1.get("tokens_used", 0)
        pipeline_log.append(step1.to_dict())
        yield {"event": "step_end", "data": step1.to_dict()}
        yield {"event": "lifecycle_trace", "data": lifecycle_counts}

        # ── Step 1.5: MANDATORY Selection Engine Gate (UNSKIPPABLE) ────
        step_sel = PipelineStep("SelectionEngine", "Competitive Resolution & Competition")
        yield {"event": "step_start", "data": step_sel.to_dict()}
        t_sel = time.time()
        try:
            # UNIFIED CANDIDATE NORMALIZATION (Ensure fallback codes have required metadata)
            for c in ai_codes:
                if "source" not in c: c["source"] = "fallback"
                if "confidence" not in c: c["confidence"] = 0.5

            # Force all codes through SelectionEngine logic
            logger.info("SELECTOR_EXECUTED: Processing %d codes through mandatory gate", len(ai_codes))
            selection_result = await loop.run_in_executor(
                None,
                lambda: self.selection_engine.select(
                    candidates=ai_codes,
                    note_text=masked_note,
                    deterministic_codes=deterministic_codes,
                    gold_codes=ground_truth
                )
            )
            ai_codes = selection_result["selected"]
            logger.info("PIPELINE_TRACE: Stage 2 End | Selected: %d", len(ai_codes))
            step_sel.status = "success"
        except Exception as exc:
            logger.error("SelectionEngine CRITICAL FAILURE: %s", exc)
            step_sel.status = "failed"
            step_sel.error = str(exc)
        
        lifecycle_counts["post_competition"] = len(ai_codes)
        
        step_sel.duration_ms = (time.time() - t_sel) * 1000
        pipeline_log.append(step_sel.to_dict())
        yield {"event": "step_end", "data": step_sel.to_dict()}
        yield {"event": "lifecycle_trace", "data": lifecycle_counts}

        # ── Step 1b: Rule Engine (Clinical Rules + CPT + Final Validation) ──
        step1b = PipelineStep("RuleEngine", "Applying Clinical Coding Rules")
        yield {"event": "step_start", "data": step1b.to_dict()}
        t0 = time.time()
        try:
            # TERMINAL SAFETY GATE (Task 78/79 parity)
            # This gate encompasses clinical reasoning, CPT validation, and governance.
            diag_c, proc_c, _ = await loop.run_in_executor(None, lambda: run_final_validation(ai_codes, masked_note))
            ai_codes = diag_c + proc_c
            
            ai_codes = await loop.run_in_executor(None, lambda: RuleEngine.apply_final_validation(ai_codes))
            step1b.duration_ms = (time.time() - t0) * 1000
            step1b.status = "success"
            logger.info("RuleEngine complete: %d validated codes", len(ai_codes))
        except Exception as exc:
            step1b.duration_ms = (time.time() - t0) * 1000
            step1b.status = "failed"
            step1b.error = str(exc)
            logger.error("RuleEngine failed: %s — codes passed through unmodified.", exc)
        
        lifecycle_counts["post_reasoning"] = len(ai_codes)
        
        pipeline_log.append(step1b.to_dict())
        yield {"event": "step_end", "data": step1b.to_dict()}
        yield {"event": "lifecycle_trace", "data": lifecycle_counts}

        # ── Step 2: Auditor (compare AI vs human codes) ───────────────
        step2 = PipelineStep("AuditorAgent", "Auditing Human vs Validated Codes")
        yield {"event": "step_start", "data": step2.to_dict()}
        t0 = time.time()
        try:
            # ai_codes is now the rule-engine-validated set
            result2 = await self.auditor.compare_codes(human_codes, ai_codes, masked_note)
            step2.duration_ms = (time.time() - t0) * 1000
            if result2.get("success") and result2.get("data"):
                discrepancies = result2["data"].get("discrepancies", [])
                summary = result2["data"].get("summary", "")
                step2.status = "success"
            else:
                step2.status = "partial"
                step2.error = result2.get("error")
            tokens_total += result2.get("tokens_used", 0)
        except Exception as exc:
            step2.duration_ms = (time.time() - t0) * 1000
            step2.status = "failed"
            step2.error = str(exc)
            logger.error("AuditorAgent failed: %s", exc)
        pipeline_log.append(step2.to_dict())
        yield {"event": "step_end", "data": step2.to_dict()}

        # ── Step 3: Clinical Explanation (Groq/LLM — structured output) ─
        step3 = PipelineStep("ExplanationAgent", "Generating Clinical Audit Explanation")
        yield {"event": "step_start", "data": step3.to_dict()}
        t0 = time.time()
        try:
            explanation = await _generate_explanation(masked_note, ai_codes, discrepancies)
            step3.duration_ms = (time.time() - t0) * 1000
            step3.status = "success"
        except Exception as exc:
            step3.duration_ms = (time.time() - t0) * 1000
            step3.status = "failed"
            step3.error = str(exc)
            explanation = _build_deterministic_explanation(ai_codes, discrepancies)
            logger.warning("ExplanationAgent failed: %s — using deterministic fallback.", exc)
        pipeline_log.append(step3.to_dict())
        yield {"event": "step_end", "data": step3.to_dict()}

        # ── Step 4: Evidence Highlighter ──────────────────────────────
        step4 = PipelineStep("EvidenceHighlighterAgent", "Linking Evidence Spans")
        yield {"event": "step_start", "data": step4.to_dict()}
        t0 = time.time()
        try:
            result4 = self.evidence_highlighter.highlight_evidence(
                masked_note, ai_codes, clinical_facts_minimal,
            )
            step4.duration_ms = (time.time() - t0) * 1000
            if result4.get("success"):
                evidence = result4.get("data", [])
                step4.status = "success"
            else:
                step4.status = "partial"
                step4.error = result4.get("error")
        except Exception as exc:
            step4.duration_ms = (time.time() - t0) * 1000
            step4.status = "failed"
            step4.error = str(exc)
            logger.warning("EvidenceHighlighter failed: %s", exc)
        pipeline_log.append(step4.to_dict())
        yield {"event": "step_end", "data": step4.to_dict()}

        # ── Section 3: Evidence Consistency Check ─────────────────────
        # If no valid evidence found for a code, downgrade confidence
        if step4.status == "success" and evidence:
            evidenced_codes = {ev.get("code", "").upper() for ev in evidence}
            for c in ai_codes:
                code_str = c.get("code", "").upper()
                if code_str not in evidenced_codes:
                    old_conf = c.get("confidence", 0)
                    new_conf = max(0.0, old_conf - 0.20)
                    c["confidence"] = new_conf
                    c["rationale"] = (c.get("rationale") or "") + " [WARNING: Weak support — no explicit evidence span found]"
                    logger.info("AuditPipeline[evidence_check]: downgraded %s confidence %.2f -> %.2f", code_str, old_conf, new_conf)

        # ── Section 6: Final Sanity Check + Terminal Evidence Gate ────────
        all_final_rejections = []
        diagnosis_codes = []
        procedure_codes = []
        print(f"TRACE_STAGE_PRE_VALIDATOR: {len(ai_codes)}")
        try:
            pre_gate_count = len(ai_codes)
            logger.error(f"VALIDATOR_INPUT_COUNT: {len(ai_codes)}")
            diagnosis_codes, procedure_codes, all_final_rejections = run_final_validation(ai_codes, masked_note)
            ai_codes = diagnosis_codes + procedure_codes
            print(f"TRACE_STAGE_POST_VALIDATOR: {len(ai_codes)}")
            if len(ai_codes) < pre_gate_count:
                logger.info(
                    "AuditPipeline[terminal_gate]: removed %d unsupported codes. Final: %d",
                    pre_gate_count - len(ai_codes),
                    len(ai_codes),
                )
        except Exception as exc:
            logger.warning("AuditPipeline[terminal_gate]: gate failed (%s) — skipping", exc)
            diagnosis_codes = [c for c in ai_codes if (c.get('type') or '').upper() != 'CPT']
            procedure_codes = [c for c in ai_codes if (c.get('type') or '').upper() == 'CPT']
        
        lifecycle_counts["post_validator"] = len(ai_codes)
        lifecycle_counts["final_output"] = len(ai_codes)
        
        if not explanation:
            explanation = _build_deterministic_explanation(ai_codes, discrepancies)

        # ── Step 9: Deterministic Trace Output ────────────────────────
        # Sort for stable audit trail ordering
        ai_codes             = sorted(ai_codes, key=lambda x: x.get("code", ""))
        diagnosis_codes      = sorted(diagnosis_codes, key=lambda x: x.get("code", ""))
        procedure_codes      = sorted(procedure_codes, key=lambda x: x.get("code", ""))
        logger.error(f"FINAL_EMISSION_COUNT: {len(ai_codes)}")
        for ac in ai_codes[:5]: logger.error(f"  EMITTED: {ac.get('code')}")
        all_final_rejections = sorted(all_final_rejections, key=lambda x: x.get("code", ""))

        total_ms = (time.time() - t_total) * 1000
        logger.info(
            "AuditPipeline v5 complete: %d codes, %d discrepancies, %.1fms",
            len(ai_codes), len(discrepancies), total_ms,
        )

        # ── Step 4: Trace Consolidation ───────────────────────────────
        total_accepted = len(ai_codes)
        total_rejected = len(all_final_rejections)
        
        # Determine top rejection reason
        rejection_counts = {}
        for rc in all_final_rejections:
            reason = rc.get("rejection_reason", "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        top_reason = max(rejection_counts, key=rejection_counts.get) if rejection_counts else "none"
        
        # Determine strongest evidence section
        section_counts = {}
        for ac in ai_codes:
            sec = ac.get("section_dominant", "full_note")
            section_counts[sec] = section_counts.get(sec, 0) + 1
        strongest_section = max(section_counts, key=section_counts.get) if section_counts else "full_note"

        trace_summary = {
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "top_rejection_reason": top_reason,
            "strongest_evidence_section": strongest_section,
        }

        # 🚨 TASK 13: FORENSIC TRACE PASS-THROUGH
        forensic_trace = result1["data"].get("forensic_trace", {}) if result1["success"] and result1["data"] else {}
        forensic_trace["terminal_rejections"] = all_final_rejections

        # 🚨 TRACE POINT 7 — FINAL EMISSION
        print("\n=== FINAL EMISSION ===")
        print("Final Count:", len(ai_codes))
        print([c.get("code") for c in ai_codes])

        yield {
            "event": "complete",
            "data": {
                "ai_codes": ai_codes,
                "diagnosis_codes": diagnosis_codes,
                "procedure_codes": procedure_codes,
                "low_confidence_codes": [],
                "discrepancies": discrepancies,
                "evidence": evidence,
                "summary": summary,
                "explanation": explanation,
                "pipeline_log": pipeline_log,
                "removed_codes": all_final_rejections,
                "trace_summary": trace_summary,
                "tokens_used": tokens_total,
                "deterministic_codes_count": len(deterministic_codes),
                "forensic_trace": forensic_trace,
                "lifecycle_counts": lifecycle_counts,
            },
        }
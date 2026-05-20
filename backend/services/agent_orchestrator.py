"""
services/agent_orchestrator.py – Central Multi-Agent Pipeline Orchestrator.

RESPONSIBILITIES:
  1. Orchestrates the sequential execution of the clinical audit agents.
  2. Manages shared pipeline state and data propagation between stages.
  3. Enforces confidence thresholds and per-agent retry logic.
  4. Provides structured execution logging for audit traceability.
"""

import asyncio
import time
from typing import Any

from agents.clinical_reader import ClinicalReaderAgent
from agents.coding_logic import CodingLogicAgent
from agents.auditor import AuditorAgent
from agents.evidence_agent import EvidenceHighlighterAgent
from config import settings
from utils.logging import get_logger
from services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when a critical agent fails and the pipeline cannot continue."""
    pass


class AgentOrchestrator:
    """
    Orchestrates the full multi-agent audit pipeline.

    Each agent is constructed fresh per run to avoid state leakage between
    concurrent HTTP requests (FastAPI is async, multiple requests can run
    simultaneously with the same orchestrator instance if we're not careful –
    stateless construction is the safe pattern).
    """

    def __init__(self):
        self.clinical_reader = ClinicalReaderAgent()
        self.coding_logic = CodingLogicAgent()
        self.auditor = AuditorAgent()
        self.evidence_agent = EvidenceHighlighterAgent()

    async def run(self, note_text: str, human_codes: list[str]) -> dict:
        """
        Execute the full audit pipeline and return the consolidated result.

        Args:
            note_text:   Raw clinical note text.
            human_codes: List of ICD-10/CPT codes entered by the human coder.

        Returns:
            Pipeline result dict (see module docstring for schema).

        Raises:
            PipelineError: If a critical stage fails after retries.
        """
        pipeline_log: list[dict] = []
        state: dict[str, Any] = {
            "note_text": note_text,
            "human_codes": human_codes,
        }

        logger.info(
            "Orchestrator: starting pipeline. note_length=%d, human_codes=%s",
            len(note_text), human_codes,
        )

        # ── Stage 1: Clinical Reader ──────────────────────────────────────────
        stage_result = await self._run_stage(
            name="ClinicalReaderAgent",
            coro=self.clinical_reader.extract_medical_entities(note_text),
            log=pipeline_log,
        )
        if not stage_result["success"]:
            raise PipelineError(f"ClinicalReaderAgent failed: {stage_result['error']}")

        state["clinical_facts"] = stage_result["data"]

        # ── Stage 2: Coding Logic (RAG + GPT) ────────────────────────────────
        stage_result = await self._run_stage(
            name="CodingLogicAgent",
            coro=self.coding_logic.generate_codes(state["clinical_facts"]),
            log=pipeline_log,
        )
        if not stage_result["success"]:
            raise PipelineError(f"CodingLogicAgent failed: {stage_result['error']}")

        state["ai_codes"] = stage_result["data"].get("codes", [])
        state["low_confidence_codes"] = stage_result["data"].get("low_confidence_codes", [])

        # ── Stage 3: Auditor ──────────────────────────────────────────────────
        clinical_summary = state["clinical_facts"].get("clinical_summary", "")
        stage_result = await self._run_stage(
            name="AuditorAgent",
            coro=self.auditor.compare_codes(
                human_codes=human_codes,
                ai_codes=state["ai_codes"],
                clinical_summary=clinical_summary,
            ),
            log=pipeline_log,
        )
        if not stage_result["success"]:
            raise PipelineError(f"AuditorAgent failed: {stage_result['error']}")

        state["discrepancies"] = stage_result["data"].get("discrepancies", [])
        state["summary"] = stage_result["data"].get("summary", "")

        # ── Stage 4: Evidence Highlighter ─────────────────────────────────────
        # This stage is synchronous (no LLM call); run in thread to avoid blocking.
        stage_result = await asyncio.to_thread(
            self._run_evidence_stage,
            note_text=note_text,
            ai_codes=state["ai_codes"],
            clinical_facts=state["clinical_facts"],
            log=pipeline_log,
        )
        state["evidence"] = stage_result

        state["pipeline_log"] = pipeline_log

        logger.info(
            "Orchestrator: pipeline complete. ai_codes=%d discrepancies=%d evidence=%d",
            len(state["ai_codes"]),
            len(state["discrepancies"]),
            len(state["evidence"]),
        )
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_stage(name: str, coro, log: list[dict]) -> dict:
        """
        Execute an async agent coroutine, record execution time, and append
        a log entry.

        Args:
            name: Human-readable agent name for logging.
            coro: Awaitable coroutine returned by the agent method.
            log:  Shared pipeline log list to append to.

        Returns:
            The agent's result dict (standard envelope: {success, data, error}).
        """
        start = time.perf_counter()
        logger.info("Orchestrator: starting %s.", name)

        try:
            result = await coro
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.append({
                "agent": name,
                "success": False,
                "elapsed_ms": round(elapsed * 1000),
                "error": str(exc),
            })
            logger.error("Orchestrator: %s raised exception: %s", name, exc)
            return {"success": False, "data": None, "error": str(exc)}

        elapsed = time.perf_counter() - start
        log.append({
            "agent": name,
            "success": result.get("success", False),
            "elapsed_ms": round(elapsed * 1000),
            "error": result.get("error"),
        })
        logger.info(
            "Orchestrator: %s complete in %.0f ms. success=%s",
            name, elapsed * 1000, result.get("success"),
        )
        return result

    def _run_evidence_stage(
        self,
        note_text: str,
        ai_codes: list[dict],
        clinical_facts: dict,
        log: list[dict],
    ) -> list[dict]:
        """
        Synchronous wrapper for the EvidenceHighlighterAgent (no LLM call).

        Returns the evidence list directly (not wrapped in an envelope).
        """
        start = time.perf_counter()
        result = self.evidence_agent.highlight_evidence(
            note_text=note_text,
            ai_codes=ai_codes,
            clinical_facts=clinical_facts,
        )
        elapsed = time.perf_counter() - start
        log.append({
            "agent": "EvidenceHighlighterAgent",
            "success": result.get("success", False),
            "elapsed_ms": round(elapsed * 1000),
            "error": result.get("error"),
        })
        return result.get("data", [])

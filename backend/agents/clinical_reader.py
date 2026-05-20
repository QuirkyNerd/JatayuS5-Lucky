"""
agents/clinical_reader.py – Clinical Reader Agent.

Reads a free-text clinical note and extracts structured medical entities
using the centralised async LLM client (Groq).

Returns:
  {
    "diagnoses":         [{"entity": "...", "evidence_sentence": "..."}],
    "procedures":        [{"entity": "...", "evidence_sentence": "..."}],
    "comorbidities":     [{"entity": "...", "evidence_sentence": "..."}],
    "medications":       [{"entity": "...", "evidence_sentence": "..."}],
    "clinical_summary":  "...",
    "evidence_sentences": {"entity_name": "verbatim sentence", ...}
  }
"""

import json
import os

try:
    # When running from project root (development)
    from config import settings
    from utils.logging import get_logger, set_request_context
    from utils.llm_client import generate_json_async
except ImportError:
    # When running from backend directory (Docker/production)
    from config import settings
    from utils.logging import get_logger, set_request_context
    from utils.gemini_client import generate_json_async

logger = get_logger(__name__)

_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "clinical_reader_prompt.txt"
)


def _load_system_prompt() -> str:
    try:
        with open(_PROMPT_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s — using inline fallback.", _PROMPT_PATH)
        return (
            "You are a medical NLP system. Extract diagnoses, procedures, "
            "comorbidities, and medications from the clinical note. "
            "Return ONLY valid JSON with keys: diagnoses, procedures, comorbidities, "
            "medications (each a list of {entity, evidence_sentence}), and clinical_summary."
        )


def _build_result(success: bool, data=None, error: str | None = None) -> dict:
    return {"success": success, "data": data, "error": error, "tokens_used": 0}


class ClinicalReaderAgent:
    """
    Agent 1: Clinical Reader.

    Uses Groq (via centralised async LLM client with JSON mode) to extract
    structured medical facts from unstructured clinical documentation.
    """

    def __init__(self):
        self.system_prompt = _load_system_prompt()
        logger.info("ClinicalReaderAgent: initialised.")

    async def extract_medical_entities(self, note_text: str) -> dict:
        """
        Extract structured medical entities from a clinical note.

        Args:
            note_text: Raw clinical note text.

        Returns:
            Standard result envelope with structured clinical facts.
        """
        set_request_context(agent_name="ClinicalReaderAgent")
        logger.info("ClinicalReaderAgent: processing note (%d chars).", len(note_text))

        # Build full prompt: system instructions + note
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"CLINICAL NOTE:\n{note_text}\n\n"
            "Return JSON only."
        )

        last_error: str | None = None
        for attempt in range(settings.agent_max_retries + 1):
            try:
                logger.info("ClinicalReaderAgent: attempt %d.", attempt + 1)

                # Async REST call — Tier 2 (fast) for extraction
                raw = await generate_json_async(full_prompt, tier="fast")

                if not raw or not raw.strip():
                    raise ValueError("Empty response from LLM")

                parsed = json.loads(raw)

                # Normalise structure: ensure required keys exist
                diagnoses    = parsed.get("diagnoses", [])
                procedures   = parsed.get("procedures", [])
                comorbidities = parsed.get("comorbidities", [])
                medications  = parsed.get("medications", [])
                summary      = parsed.get("clinical_summary", "")

                # Build convenience evidence map: entity → verbatim sentence
                evidence_sentences: dict[str, str] = {}
                for group in (diagnoses, procedures, comorbidities, medications):
                    for item in group:
                        entity = item.get("entity", "")
                        sent   = item.get("evidence_sentence", "")
                        if entity and sent:
                            evidence_sentences[entity] = sent

                result = {
                    "diagnoses":          diagnoses,
                    "procedures":         procedures,
                    "comorbidities":      comorbidities,
                    "medications":        medications,
                    "clinical_summary":   summary,
                    "evidence_sentences": evidence_sentences,
                }

                logger.info(
                    "ClinicalReaderAgent: extracted %d diagnoses, %d procedures, %d comorbidities.",
                    len(diagnoses), len(procedures), len(comorbidities),
                )
                return _build_result(success=True, data=result)

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error on attempt {attempt + 1}: {e}"
                logger.warning("ClinicalReaderAgent: %s", last_error)

            except Exception as e:
                last_error = str(e)
                logger.error("ClinicalReaderAgent: error on attempt %d: %s", attempt + 1, e)

        return _build_result(success=False, error=last_error)
"""
agents/auditor.py – Auditor Agent.

Compares human-entered billing codes against AI-suggested codes using
Gemini for nuanced clinical reasoning, with a deterministic set-based
fallback if the LLM call fails.

Classifications:
  - correct_code     : present in both human and AI lists
  - missed_code      : in AI list but absent from human list (revenue leakage risk)
  - unsupported_code : in human list but absent from AI list (up-coding risk)
"""

import json
import os

try:
    # When running from project root (development)
    from config import settings
    from utils.logging import get_logger, set_request_context
    from utils.code_normalizer import normalize_code
    from utils.llm_client import generate_json_async
except ImportError:
    # When running from backend directory (Docker/production)
    from config import settings
    from utils.logging import get_logger, set_request_context
    from utils.code_normalizer import normalize_code
    from utils.gemini_client import generate_json_async

logger = get_logger(__name__)

_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "auditor_prompt.txt"
)


def _load_prompt() -> str:
    try:
        with open(_PROMPT_PATH, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s – using inline fallback.", _PROMPT_PATH)
        return (
            "You are a senior medical coding auditor. "
            "Compare human-entered codes against AI-suggested codes. "
            "For each code, classify as: correct_code, missed_code, or unsupported_code. "
            "Return a JSON object with keys: discrepancies (list) and summary (string). "
            "Each discrepancy must have: code, type, message, severity (high/medium/low)."
        )


def _build_result(success: bool, data=None, error: str | None = None, tokens: int = 0) -> dict:
    return {"success": success, "data": data, "error": error, "tokens_used": tokens}


def _deterministic_compare(
    human_codes: list[str],
    ai_codes: list[dict],
) -> list[dict]:
    """
    Deterministic fallback: classify all codes using set operations.
    """
    human_norm = {normalize_code(c) for c in human_codes}
    ai_map = {normalize_code(c["code"]): c for c in ai_codes}

    discrepancies: list[dict] = []

    for code in sorted(human_norm & set(ai_map)):
        ai_entry = ai_map[code]
        discrepancies.append({
            "code": code,
            "type": "correct_code",
            "message": (
                f"{code} was correctly included by the human coder. "
                f"AI confidence: {ai_entry.get('confidence', 0):.0%}."
            ),
            "severity": "low",
        })

    for code in sorted(set(ai_map) - human_norm):
        ai_entry = ai_map[code]
        conf = ai_entry.get("confidence", 0.80)
        discrepancies.append({
            "code": code,
            "type": "missed_code",
            "message": (
                f"{code} ({ai_entry.get('description', '')}) was identified by AI "
                f"but not submitted by the human coder (confidence {conf:.0%}). "
                "This may indicate under-coding and revenue leakage."
            ),
            "severity": "high" if conf >= 0.85 else "medium",
        })

    for code in sorted(human_norm - set(ai_map)):
        discrepancies.append({
            "code": code,
            "type": "unsupported_code",
            "message": (
                f"{code} was submitted by the human coder but was not identified "
                "by the AI. Verify that clinical documentation fully supports this code."
            ),
            "severity": "medium",
        })

    return discrepancies


class AuditorAgent:
    """
    Agent 3: Auditor.

    Uses Gemini (via fallback client) to provide clinically nuanced code comparison
    with severity scoring. Falls back to deterministic set logic if the LLM fails.
    """

    def __init__(self):
        self.model_name = settings.groq_model_primary
        self.system_prompt = _load_prompt()  # ✅ keep prompt

    def _clean_json_response(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    async def compare_codes(
        self,
        human_codes: list[str],
        ai_codes: list[dict],
        clinical_summary: str = "",
    ) -> dict:
        """
        Compare human-entered codes against AI-suggested codes.
        """
        set_request_context(agent_name="AuditorAgent")
        logger.info(
            "AuditorAgent: comparing %d human codes vs %d AI codes.",
            len(human_codes), len(ai_codes),
        )

        # Normalise inputs before comparison
        norm_human = [normalize_code(c) for c in human_codes]

        if settings.benchmark_mode:
            logger.info("AuditorAgent: Benchmark mode active, bypassing Groq comparison layer.")
            discrepancies = _deterministic_compare(norm_human, ai_codes)
            summary = _build_summary(discrepancies)
            return _build_result(
                success=True,
                data={"discrepancies": discrepancies, "summary": summary}
            )

        # Grounding section
        grounding_section = ""
        if clinical_summary:
            grounding_section = f"CLINICAL SUMMARY GROUNDING:\n{clinical_summary}"

        # Safe f-string
        full_prompt = f"""
{self.system_prompt}

HUMAN-ENTERED CODES:
{norm_human}

AI-SUGGESTED CODES:
{json.dumps(ai_codes, indent=2)}

{grounding_section}

IMPORTANT:
- Return ONLY valid JSON
- Do NOT include markdown
- Do NOT include explanations
- Start response with '{{' and end with '}}'
"""

        # ✅ ADD THIS LINE (STRICT JSON enforcement)
        full_prompt += "\n\nReturn STRICT JSON only."

        try:
            # Await async Groq call — Tier 1 (best) for clinical comparison
            response_text = await generate_json_async(full_prompt, tier="best")

            raw = self._clean_json_response(response_text)
            parsed = json.loads(raw)

            discrepancies = parsed.get("discrepancies", [])
            summary = parsed.get("summary", "")

            tokens_used = 0  # not available in fallback

            logger.info("AuditorAgent: Gemini returned %d discrepancies.", len(discrepancies))

            return _build_result(
                success=True,
                data={"discrepancies": discrepancies, "summary": summary},
                tokens=tokens_used
            )

        except Exception as e:
            logger.warning(
                "AuditorAgent: Groq failed (%s) – using deterministic fallback.", e
            )

            discrepancies = _deterministic_compare(norm_human, ai_codes)
            summary = _build_summary(discrepancies)

            return _build_result(
                success=True,
                data={"discrepancies": discrepancies, "summary": summary}
            )


def _build_summary(discrepancies: list[dict]) -> str:
    counts = {"correct_code": 0, "missed_code": 0, "unsupported_code": 0}

    for d in discrepancies:
        counts[d.get("type", "correct_code")] += 1

    parts = []

    if counts["correct_code"]:
        parts.append(f"{counts['correct_code']} code(s) correctly submitted.")

    if counts["missed_code"]:
        parts.append(
            f"{counts['missed_code']} code(s) missed by human coder (potential revenue leakage)."
        )

    if counts["unsupported_code"]:
        parts.append(
            f"{counts['unsupported_code']} code(s) unsupported by AI retrieval (verify documentation)."
        )

    return " | ".join(parts) or "No discrepancies found."
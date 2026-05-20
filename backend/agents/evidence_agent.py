"""
agents/evidence_agent.py – Evidence Highlighter Agent (v2).

Maps each billing code back to the exact sentence(s) in the clinical note
that support it, using the pre-built SentenceIndexer.

For each code the agent:
  1. Looks up the entity's verbatim evidence sentence from the clinical reader.
  2. Uses SentenceIndexer.find_best_match() to locate that sentence in the index.
  3. Returns sentence_id, start_char, end_char for frontend highlighting.

v2 changes:
  - Contextual validation for description-based fallback (Strategy 4)
  - Threshold raised to 0.38 to reduce false positives
  - Procedural-only sentences are rejected for diagnosis code mappings
"""

from typing import Any
try:
    from utils.logging import get_logger
    from utils.sentence_indexer import SentenceIndexer
except ImportError:
    from utils.logging import get_logger
    from utils.sentence_indexer import SentenceIndexer

# Clinical indicator words — a matched sentence must contain at least one
# of these before being accepted as valid evidence for a diagnosis code.
_CLINICAL_EVIDENCE_WORDS = frozenset([
    "diagnosed", "diagnosis", "history", "presents", "documented", "confirmed",
    "elevated", "decreased", "positive", "found", "noted", "exhibited",
    "culture", "organism", "infection", "sepsis", "fever", "pain",
    "nephropathy", "neuropathy", "retinopathy", "failure", "chronic", "acute",
    "hba1c", "creatinine", "wbc", "glucose", "blood", "bilateral",
    "hypertension", "diabetes", "ckd", "esrd", "pneumonia", "bacteremia",
])

# Purely procedural words — if a sentence contains ONLY these and no clinical
# indicators, it should not be used as evidence for a diagnosis code.
_PROCEDURAL_ONLY_WORDS = frozenset([
    "inserted", "placed", "catheter", "foley", "iv access", "central line",
    "intubated", "intubation", "ventilated", "secured", "positioned",
    "connected", "attached", "administered", "given", "infused",
])


def _is_clinically_valid_sentence(sentence_text: str, code: str) -> bool:
    """
    Returns True if the sentence is valid evidence for a diagnosis code.
    CPT codes bypass this check (procedures can be evidenced by procedural text).
    ICD-10 codes require at least one clinical indicator word in the sentence.
    """
    if not sentence_text:
        return False
    # CPT codes — procedural text is valid evidence
    if code and (code.isdigit() or (len(code) == 5 and code[0].isdigit())):
        return True
    lower = sentence_text.lower()
    has_clinical = any(w in lower for w in _CLINICAL_EVIDENCE_WORDS)
    return has_clinical

logger = get_logger(__name__)


def _build_result(success: bool, data: Any = None, error: str | None = None) -> dict:
    """Standard agent result envelope."""
    return {"success": success, "data": data, "error": error}


class EvidenceHighlighterAgent:
    """
    Agent 4: Evidence Highlighter.

    Responsibility: For every AI-suggested code, locate the exact sentence
    (with character offsets) in the original clinical note that justifies it.

    Uses SentenceIndexer for reliable, position-stable retrieval.
    """

    def highlight_evidence(
        self,
        note_text: str,
        ai_codes: list[dict],
        clinical_facts: dict,
    ) -> dict:
        """
        Map billing codes to exact evidence sentences.
        """
        logger.info(
            "EvidenceHighlighterAgent: mapping %d codes to evidence.", len(ai_codes)
        )

        try:
            if not note_text:
                logger.warning("Empty note_text received.")
                return _build_result(success=False, error="Empty clinical note")

            indexer = SentenceIndexer(note_text)

            evidence_sentences: dict[str, str] = clinical_facts.get("evidence_sentences", {}) or {}

            # Build reverse map: code → entity names
            code_entity_map: dict[str, list[str]] = {}

            for category in ("diagnoses", "comorbidities", "procedures", "medications"):
                for item in clinical_facts.get(category, []) or []:
                    entity = item.get("entity", "")
                    if not entity:
                        continue

                    for code_obj in ai_codes:
                        code = code_obj.get("code", "")
                        if code:
                            code_entity_map.setdefault(code, []).append(entity)

            evidence_list = []

            for code_obj in ai_codes:
                code = code_obj.get("code", "")
                description = code_obj.get("description", "")
                rationale = code_obj.get("rationale", "")

                best_span = None

                # Strategy 1: entity → verbatim sentence
                for entity in code_entity_map.get(code, []):
                    verbatim = evidence_sentences.get(entity, "")
                    if verbatim:
                        span = indexer.find_best_match(verbatim)
                        if span:
                            best_span = span
                            break

                # Strategy 2: evidence_span from deterministic/RAG layer
                evidence_span = code_obj.get("evidence_span", "")
                if not best_span and evidence_span:
                    best_span = indexer.find_best_match(evidence_span)

                # Strategy 3: rationale
                if not best_span and rationale:
                    best_span = indexer.find_best_match(rationale)

                # Strategy 4: description-based match with contextual validation
                # Threshold raised to 0.38 to reduce false positives.
                # Only accept if the matched sentence contains clinical indicators.
                if not best_span and description:
                    candidate_span = indexer.find_best_match(description, threshold=0.38)
                    if candidate_span and _is_clinically_valid_sentence(candidate_span.text, code):
                        best_span = candidate_span
                    elif candidate_span:
                        logger.debug(
                            "Evidence[%s]: description match rejected — procedural-only sentence", code
                        )

                # Strategy 5: fallback to first sentence (only if no other option)
                if not best_span and indexer.sentences:
                    first = indexer.sentences[0]
                    if _is_clinically_valid_sentence(first.text, code):
                        best_span = first

                if best_span:
                    evidence_list.append({
                        "code": code,
                        "sentence_id": best_span.sentence_id,
                        "sentence_text": best_span.text,
                        "start_char": best_span.start_char,
                        "end_char": best_span.end_char,
                    })
                else:
                    logger.warning(f"No evidence found for code: {code}")

            logger.info(
                "EvidenceHighlighterAgent: evidence mapped for %d codes.", len(evidence_list)
            )

            return _build_result(success=True, data=evidence_list)

        except Exception as e:
            logger.error("EvidenceHighlighterAgent error: %s", e)
            return _build_result(success=False, error=str(e))
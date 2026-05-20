"""
utils/sentence_indexer.py – Reliable sentence-based span indexer.

Instead of running a fragile substring search at query time, this module
pre-processes the clinical note once and builds an index mapping each
sentence to its exact character span in the original text.

Agents reference sentences by their stable `sentence_id` (0-indexed).
This makes evidence highlighting far more reliable than ad-hoc string
search, because it works even when the model paraphrases slightly.

Usage:
    indexer = SentenceIndexer(note_text)
    sentences = indexer.sentences  # List[SentenceSpan]
    span = indexer.get_span(2)     # character offsets for sentence #2
    best_id = indexer.find_best_match("hypertension")  # fuzzy match
"""

import re
from dataclasses import dataclass
from typing import Optional
from difflib import SequenceMatcher


@dataclass
class SentenceSpan:
    """Represents a single indexed sentence with character positions."""
    sentence_id: int
    text: str
    start_char: int
    end_char: int


class SentenceIndexer:
    """
    Splits a clinical note into sentences and indexes their character
    positions in the original string.

    Args:
        text: The full clinical note text.
    """

    # Regex: split after sentence-ending punctuation followed by whitespace,
    # or at paragraph breaks (two+ newlines).
    _SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|(?:\r?\n){1,}")

    def __init__(self, text: str):
        self._original_text = text
        self.sentences: list[SentenceSpan] = self._build_index(text)

    def _build_index(self, text: str) -> list[SentenceSpan]:
        """
        Tokenise text into sentences and record their character offsets.

        Returns:
            List of SentenceSpan objects in document order.
        """
        spans: list[SentenceSpan] = []
        sentence_id = 0
        cursor = 0

        # Split text and walk through to find exact offsets in the original.
        raw_parts = self._SPLIT_PATTERN.split(text)

        for part in raw_parts:
            stripped = part.strip()
            if not stripped:
                cursor += len(part)
                # Account for the separator that was consumed
                cursor = text.find(stripped, cursor) if stripped else cursor
                continue

            # Find this sentence in the original text starting from cursor.
            start = text.find(stripped, cursor)
            if start == -1:
                # Fallback: search from the beginning (shouldn't happen)
                start = text.find(stripped)
            end = start + len(stripped)

            spans.append(SentenceSpan(
                sentence_id=sentence_id,
                text=stripped,
                start_char=start,
                end_char=end,
            ))
            sentence_id += 1
            cursor = end

        return spans

    def get_span(self, sentence_id: int) -> Optional[SentenceSpan]:
        """
        Retrieve a SentenceSpan by its ID.

        Args:
            sentence_id: 0-indexed sentence identifier.

        Returns:
            SentenceSpan or None if out of range.
        """
        if 0 <= sentence_id < len(self.sentences):
            return self.sentences[sentence_id]
        return None

    def find_best_match(self, query: str, threshold: float = 0.40) -> Optional[SentenceSpan]:
        """
        Find the sentence that best matches the query string using
        SequenceMatcher similarity ratio.

        Args:
            query:     Text to look for (e.g. a clinical fact extracted by LLM).
            threshold: Minimum similarity ratio to accept (0–1).

        Returns:
            Best-matching SentenceSpan or None if nothing exceeds threshold.
        """
        best_ratio = 0.0
        best_span: Optional[SentenceSpan] = None

        q_lower = query.lower()

        for span in self.sentences:
            # Exact substring check first (fast path)
            if q_lower in span.text.lower():
                return span

            ratio = SequenceMatcher(
                None, q_lower, span.text.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = span

        if best_ratio >= threshold:
            return best_span
        return None

    def all_sentences_text(self) -> list[str]:
        """Return list of all sentence texts in index order."""
        return [s.text for s in self.sentences]

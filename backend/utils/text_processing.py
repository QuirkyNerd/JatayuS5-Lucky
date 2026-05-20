"""
utils/text_processing.py – Text normalization and span utilities.

Provides helpers used by agents for:
  - whitespace normalization
  - finding a substring in the original text and returning its character span
"""

import re
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize clinical note text:
    - Collapse multiple whitespace into single space.
    - Strip leading / trailing whitespace.

    Args:
        text: Raw clinical note string.

    Returns:
        Cleaned string.
    """
    return re.sub(r"\s+", " ", text).strip()


def find_span(text: str, substring: str) -> Optional[tuple[int, int]]:
    """
    Find the first occurrence of ``substring`` in ``text`` and return
    the character positions as a (start, end) tuple.

    The search is case-insensitive and ignores internal whitespace
    differences (collapses multi-space to single space).

    Args:
        text:      The haystack string (e.g. the original clinical note).
        substring: The needle to locate.

    Returns:
        (start_char, end_char) tuple or None if not found.
    """
    if not substring:
        return None

    # Normalise both sides for a resilient match
    norm_text = normalize_text(text)
    norm_sub = normalize_text(substring)

    start = norm_text.lower().find(norm_sub.lower())
    if start == -1:
        return None
    return (start, start + len(norm_sub))


def extract_sentences_simple(text: str) -> list[str]:
    """
    Split text into sentences using a simple regex heuristic.

    Suitable for clinical notes which often use period-terminated sentences
    or newline-separated entries.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentence strings.
    """
    # Split on period + space/newline, or plain newline
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if s.strip()]

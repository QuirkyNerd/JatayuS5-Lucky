"""
utils/code_normalizer.py – ICD-10 and CPT code normalisation utilities.

Clinical coders sometimes omit decimal points or use inconsistent casing.
This module ensures codes are consistently formatted before comparison,
storage, and display.

Examples:
  "e119"  → "E11.9"
  "i10 "  → "I10"
  "47562" → "47562"   (CPT – numeric only, no normalisation needed)
"""

import re


# ICD-10 codes follow the pattern: 1 letter + 2 digits + optional decimal + up to 4 chars
_ICD10_PATTERN = re.compile(r"^([A-Z])(\d{2})(\.?)(\w{0,4})$", re.IGNORECASE)

# CPT codes are purely 5-digit numeric
_CPT_PATTERN = re.compile(r"^\d{5}$")


def normalize_code(raw: str) -> str:
    """
    Return the canonical uppercase form of a billing code.

    ICD-10 rules applied:
      1. Strip whitespace
      2. Upper-case
      3. Insert decimal after the 3rd character if absent and code is > 3 chars

    CPT codes are returned as-is (stripped + upper).

    Args:
        raw: Raw code string from user input or database.

    Returns:
        Normalised code string.

    Examples:
        >>> normalize_code("e119")
        'E11.9'
        >>> normalize_code("E11.9")
        'E11.9'
        >>> normalize_code("47562")
        '47562'
        >>> normalize_code("  i10  ")
        'I10'
    """
    code = raw.strip().upper()

    # CPT (5-digit numeric) – no transformation needed
    if _CPT_PATTERN.match(code):
        return code

    # ICD-10 – insert decimal if missing
    m = _ICD10_PATTERN.match(code.replace(".", ""))
    if m:
        letter, two_digits, _, rest = m.groups()
        if rest:
            return f"{letter}{two_digits}.{rest}"
        return f"{letter}{two_digits}"

    # Unknown format – return uppercased strip only
    return code


def normalize_codes(codes: list[str]) -> list[str]:
    """
    Normalise a list of billing codes.

    Args:
        codes: Raw code strings.

    Returns:
        List of normalised code strings (duplicates preserved, order preserved).
    """
    return [normalize_code(c) for c in codes]


def deduplicate_codes(codes: list[str]) -> list[str]:
    """
    Normalise and deduplicate a list of codes, preserving order of first occurrence.

    Args:
        codes: Raw code strings (may contain duplicates or near-duplicates like
               "E11.9" and "e119").

    Returns:
        Deduplicated list of normalised codes.
    """
    seen: set[str] = set()
    result: list[str] = []
    for code in codes:
        n = normalize_code(code)
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result

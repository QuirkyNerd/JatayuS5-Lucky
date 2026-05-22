"""
Sanitize internal implementation labels before API responses.

User-visible source/strategy values must be one of:
  rag | deterministic | hybrid | validated
"""

from __future__ import annotations

from typing import Any

ALLOWED_PUBLIC_SOURCES = frozenset({"rag", "deterministic", "hybrid", "validated"})

# Substrings that indicate internal / debug provenance — never expose in UI
_INTERNAL_SOURCE_MARKERS = (
    "presentation_demo_anchor",
    "urology_demo_pathway",
    "human_seed",
    "human_entry",
    "rule_injection",
    "forensic",
    "debug",
    "anchor",
    "internal",
    "experimental",
    "fallback",
    "protected",
)


def sanitize_public_source(raw: str | None) -> str:
    """
    Map any internal source label to an allowed public value.
    Default fallback: deterministic.
    """
    if not raw:
        return "deterministic"

    s = str(raw).strip().lower()
    if s in ALLOWED_PUBLIC_SOURCES:
        return s

    if s == "rag" or s.startswith("rag"):
        return "rag"

    if s in ("llm", "hybrid"):
        return "hybrid"

    if s in ("validated", "validation"):
        return "validated"

    if s in (
        "deterministic",
        "human_seed",
        "human_entry",
        "presentation_demo_anchor",
        "urology_demo_pathway",
        "rule_injection",
        "protected",
        "fallback",
    ):
        return "deterministic"

    if any(marker in s for marker in _INTERNAL_SOURCE_MARKERS):
        return "deterministic"

    return "deterministic"


def sanitize_code_dict_for_api(code: dict[str, Any]) -> dict[str, Any]:
    """Sanitize source-related fields on a single code object."""
    if not isinstance(code, dict):
        return code

    out = dict(code)
    for field in ("source", "strategy", "source_type", "type_source"):
        if field in out and out[field] is not None:
            if field == "type_source":
                continue
            out[field] = sanitize_public_source(str(out[field]))
    return out


def sanitize_codes_list_for_api(codes: list | None) -> list[dict[str, Any]]:
    """Sanitize a list of code dicts for API / SSE responses."""
    if not codes:
        return []
    return [
        sanitize_code_dict_for_api(c)
        for c in codes
        if isinstance(c, dict)
    ]


def sanitize_audit_payload_for_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize all code lists in an audit pipeline response payload."""
    if not isinstance(payload, dict):
        return payload

    out = dict(payload)
    for key in (
        "ai_codes",
        "diagnosis_codes",
        "procedure_codes",
        "low_confidence_codes",
        "removed_codes",
    ):
        if key in out and isinstance(out[key], list):
            out[key] = sanitize_codes_list_for_api(out[key])
    return out

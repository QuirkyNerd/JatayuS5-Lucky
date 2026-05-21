"""Canonical case lifecycle statuses (API + DB)."""

CANONICAL_STATUSES = ("draft", "submitted", "in_review", "approved", "rejected")

# Legacy / UI aliases → canonical
STATUS_ALIASES = {
    "under_review": "in_review",
    "in review": "in_review",
    "in-review": "in_review",
    "pending": "submitted",
    "review": "in_review",
    "reviewed": "in_review",
}


def normalize_status(raw: str) -> str:
    if not raw or not str(raw).strip():
        raise ValueError("Status is required.")
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if key in STATUS_ALIASES:
        key = STATUS_ALIASES[key]
    if key not in CANONICAL_STATUSES:
        allowed = ", ".join(CANONICAL_STATUSES)
        raise ValueError(f"Invalid status '{raw}'. Allowed values: {allowed}")
    return key


def status_for_response(stored: str | None) -> str:
    """Normalize DB value for API responses."""
    if not stored:
        return "draft"
    s = stored.strip().lower()
    if s == "under_review":
        return "in_review"
    return s

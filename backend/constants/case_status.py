"""Canonical case lifecycle statuses (API + DB)."""
from enum import Enum


class CaseStatus(str, Enum):
    draft = "draft"
    submitted = "submitted"
    in_review = "in_review"
    approved = "approved"
    rejected = "rejected"


CANONICAL_STATUSES = tuple(s.value for s in CaseStatus)

# Legacy / UI aliases → canonical value
STATUS_ALIASES = {
    "under_review": CaseStatus.in_review.value,
    "in review": CaseStatus.in_review.value,
    "in-review": CaseStatus.in_review.value,
    "pending": CaseStatus.submitted.value,
    "review": CaseStatus.in_review.value,
    "reviewed": CaseStatus.in_review.value,
}

# Role → allowed target statuses (PATCH /cases/{id}/status)
ROLE_ALLOWED_STATUSES: dict[str, frozenset[str]] = {
    "ADMIN": frozenset(CANONICAL_STATUSES),
    "REVIEWER": frozenset({
        CaseStatus.in_review.value,
        CaseStatus.approved.value,
        CaseStatus.rejected.value,
    }),
    "CODER": frozenset(),  # coders use POST /submit only
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
    if not stored:
        return CaseStatus.draft.value
    s = stored.strip().lower()
    if s == "under_review":
        return CaseStatus.in_review.value
    if s in CANONICAL_STATUSES:
        return s
    return normalize_status(s) if s else CaseStatus.draft.value


def assert_role_may_set(role: str, new_status: str) -> None:
    allowed = ROLE_ALLOWED_STATUSES.get(role, frozenset())
    if new_status not in allowed:
        if role == "REVIEWER":
            raise ValueError(
                "Reviewers may only set status to: in_review, approved, rejected."
            )
        if role == "CODER":
            raise ValueError("Coders cannot update case status via this endpoint.")
        raise ValueError(f"Role {role} cannot set status to '{new_status}'.")

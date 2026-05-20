"""
schemas/audit.py – Pydantic request/response schemas for the /audit endpoint.

Separating schemas from routes keeps route logic clean and makes schemas
reusable across multiple endpoints and tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    """Payload for POST /api/v1/audit."""

    note_text: str = Field(
        ...,
        min_length=10,
        description="Free-text clinical note (discharge summary, surgical note, etc.)",
    )
    human_codes: list[str] = Field(
        ...,
        description='ICD-10 / CPT codes entered by the human coder, e.g. ["I10", "E11.9"]',
    )
    case_id: int | None = Field(None, description="Optional ID of an existing case to re-audit/update")


# ── Code entries ──────────────────────────────────────────────────────────────

class CodeEntry(BaseModel):
    """A single AI-suggested billing code with metadata."""

    code: str = Field(..., description="Normalised billing code (e.g. E11.9)")
    description: str = Field(..., description="Human-readable code description")
    type: str = Field(default="ICD-10", description="Code system: ICD-10 or CPT")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence score")
    rationale: str | None = Field(None, description="Brief reason for code selection")


# ── Discrepancy ───────────────────────────────────────────────────────────────

class Discrepancy(BaseModel):
    """A classified difference between human-entered and AI-suggested codes."""

    code: str
    type: str = Field(
        ...,
        description="One of: missed_code | unsupported_code | correct_code",
    )
    message: str = Field(..., description="Plain-language explanation of the discrepancy")
    severity: str = Field(
        default="medium",
        description="Risk level: high | medium | low",
    )


# ── Evidence ──────────────────────────────────────────────────────────────────

class Evidence(BaseModel):
    """Maps an AI-suggested code to an exact sentence span in the clinical note."""

    code: str
    sentence_id: int
    sentence_text: str
    start_char: int
    end_char: int


# ── Outbound ──────────────────────────────────────────────────────────────────

class AuditResponse(BaseModel):
    """Structured response from the full audit pipeline."""

    audit_id: int | None = Field(
        None, description="Database ID of the stored audit result"
    )
    note_hash: str = Field(
        ..., description="SHA-256 digest of the normalised note (for integrity tracking)"
    )
    cache_hit: bool = Field(
        default=False, description="True when result was served from the in-memory cache"
    )
    request_id: str = Field(
        ..., description="UUID for end-to-end log tracing"
    )
    ai_codes: list[CodeEntry] = Field(default_factory=list)
    low_confidence_codes: list[CodeEntry] = Field(
        default_factory=list,
        description="Codes below MIN_CODE_CONFIDENCE; require human review before submission",
    )
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str = Field(default="", description="Plain-language audit summary")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of audit completion")


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Payload for POST /api/v1/feedback."""
    
    note_hash: str = Field(..., description="Hash of the document to associate feedback")
    ai_code: str = Field(..., description="The AI suggested code")
    decision: str = Field(..., description="'accepted' or 'rejected'")


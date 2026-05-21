"""
api/case_routes.py – Case Management endpoints for CodePerfectAuditor.

Endpoints:
  GET  /cases           – list cases (filtered, paginated, tenant-isolated)
  GET  /cases/{id}      – single case detail
  PATCH /cases/{id}     – update status (reviewer/admin only)
  DELETE /cases/{id}    – soft delete (admin only)
"""

import json
from datetime import datetime

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

try:
    # When running from project root (development)
    from backend.database.db import get_db
    from backend.database.models import Case, User
    from backend.security.auth import get_current_user, require_admin, require_reviewer
    from backend.utils.logging import get_logger
except ImportError:
    # When running from backend directory (Docker/production)
    from database.db import get_db
    from database.models import Case, User
    from security.auth import get_current_user, require_admin, require_reviewer
    from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cases", tags=["cases"])


try:
    from backend.constants.case_status import (
        CANONICAL_STATUSES,
        CaseStatus,
        assert_role_may_set,
        normalize_status,
        status_for_response,
    )
except ImportError:
    from constants.case_status import (
        CANONICAL_STATUSES,
        CaseStatus,
        assert_role_may_set,
        normalize_status,
        status_for_response,
    )


class CaseStatusUpdate(BaseModel):
    status: CaseStatus | None = None
    priority: str | None = None
    feedback: str = ""
    comment: str = ""  # legacy alias for feedback
    rejection_reason: str = ""
    correction_notes: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, CaseStatus):
            return v
        if isinstance(v, Enum):
            return CaseStatus(normalize_status(v.value))
        normalized = normalize_status(str(v))
        return CaseStatus(normalized)


class RejectPayload(BaseModel):
    justification: str = ""
    feedback: str = ""
    rejection_reason: str = ""
    correction_notes: str = ""
    review_confidence: float = 1.0


class AssignPayload(BaseModel):
    reviewer_id: int


class UpdateCodesPayload(BaseModel):
    final_codes: list = []
    justification: str = ""


async def _load_case(case_id: int, current_user: User, db: AsyncSession) -> Case:
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if case.is_demo != current_user.is_demo:
        raise HTTPException(status_code=403, detail="Forbidden: Environment mismatch.")
    if current_user.role == "CODER" and case.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return case


def _compose_feedback(payload: CaseStatusUpdate | RejectPayload) -> str:
    parts = []
    if getattr(payload, "rejection_reason", None):
        parts.append(f"Rejection reason: {payload.rejection_reason}")
    if getattr(payload, "correction_notes", None):
        parts.append(f"Correction notes: {payload.correction_notes}")
    fb = (getattr(payload, "feedback", None) or getattr(payload, "comment", None) or "").strip()
    if fb:
        parts.append(fb)
    just = (getattr(payload, "justification", None) or "").strip()
    if just and just not in fb:
        parts.append(just)
    return "\n\n".join(p for p in parts if p).strip()


async def _apply_status_change(
    case: Case,
    new_status: str,
    current_user: User,
    db: AsyncSession,
    feedback: str = "",
) -> None:
    case.status = new_status
    if new_status == "in_review":
        if not case.assigned_to and current_user.role in ("REVIEWER", "ADMIN"):
            case.assigned_to = current_user.id if current_user.role == "REVIEWER" else case.assigned_to
            case.assigned_at = datetime.utcnow()
            case.assignment_status = "assigned"
    if new_status in ("approved", "rejected"):
        case.reviewed_by = current_user.id
        case.reviewed_at = datetime.utcnow()
        if new_status == "approved":
            case.review_confidence = case.review_confidence or 1.0
    if new_status == "rejected":
        if not feedback.strip():
            raise HTTPException(
                status_code=400,
                detail="Feedback is required when rejecting a case.",
            )
        case.reviewer_notes = feedback
    elif feedback.strip():
        case.reviewer_notes = feedback
    case.updated_at = datetime.utcnow()


def _case_to_dict(c: Case) -> dict:
    try:
        summary_data = json.loads(c.summary) if c.summary else {}
        if isinstance(summary_data, dict):
            summary_text = summary_data.get("summary", "")
            explanation_text = summary_data.get("explanation", "")
            removed_codes = summary_data.get("removed_codes", [])
        else:
            summary_text = c.summary
            explanation_text = ""
            removed_codes = []
    except Exception:
        summary_text = c.summary
        explanation_text = ""
        removed_codes = []

    return {
        "id":              c.id,
        "user_id":         c.user_id,
        "org_id":          c.org_id,
        "input_text":      c.input_text,
        "evidence":        json.loads(c.evidence or "[]"),
        "pipeline_log":    json.loads(c.pipeline_log or "[]"),
        "ai_codes":        json.loads(c.ai_codes or "[]"),
        "human_codes":     json.loads(c.human_codes or "[]"),
        "discrepancies":   json.loads(c.discrepancies or "[]"),
        "risk_score":      c.risk_score,
        "revenue_impact":  c.revenue_impact,
        "coding_accuracy": c.coding_accuracy,
        "avg_confidence":  c.avg_confidence,
        "processing_time": c.processing_time,
        "summary":         summary_text,
        "explanation":     explanation_text,
        "removed_codes":   removed_codes,
        "status":            status_for_response(c.status),
        "reviewer_notes":    c.reviewer_notes,
        "review_feedback":   c.reviewer_notes,
        "reviewed_by":       c.reviewed_by,
        "reviewed_at":       c.reviewed_at.isoformat() if c.reviewed_at else None,
        "review_confidence": c.review_confidence,
        "model_used":        c.model_used,
        "tokens_used":       c.tokens_used,
        "cost_estimate":     c.cost_estimate,
        "priority":          c.priority,
        "assignment_status": c.assignment_status,
        "reviewer_name":     c.reviewer.name if hasattr(c, "reviewer") and c.reviewer else "Unassigned",

        "created_at":        c.created_at.isoformat() if c.created_at else None,
    }


@router.get("")
async def list_cases(
    page:         int   = Query(1, ge=1),
    page_size:    int   = Query(20, ge=1, le=100),
    status:       str | None = Query(None),
    min_risk:     float | None = Query(None),
    max_risk:     float | None = Query(None),
    from_date:    str | None = Query(None),
    to_date:      str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    List cases with filters. Tenant-isolated: Coder sees only own cases,
    Admin/Reviewer sees all cases in their org.
    """
    filters = []

    # Step 1: Mandatory Environment Isolation
    mode = "DEMO" if current_user.is_demo else "PRODUCTION"
    filters.append(Case.is_demo == current_user.is_demo)
    logger.info("CASE_QUERY_MODE: %s (user=%s)", mode, current_user.email)

    # Step 2: Tenant isolation within the environment
    if current_user.role == "CODER":
        filters.append(Case.user_id == current_user.id)
    # ADMIN and REVIEWER see ALL cases in their environment.

    if status:
        try:
            norm_status = normalize_status(status)
        except ValueError as exc:
            allowed = ", ".join(CANONICAL_STATUSES)
            raise HTTPException(
                status_code=400,
                detail=str(exc) if "Allowed" in str(exc) else f"Allowed values: {allowed}",
            ) from exc
        if norm_status == "in_review":
            filters.append(Case.status.in_(["in_review", "under_review", "pending"]))
        else:
            filters.append(Case.status == norm_status)
    if min_risk is not None:
        filters.append(Case.risk_score >= min_risk)
    if max_risk is not None:
        filters.append(Case.risk_score <= max_risk)
    if from_date:
        try:
            dt = datetime.fromisoformat(from_date)
            filters.append(Case.created_at >= dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid from_date format. Use ISO 8601.")
    if to_date:
        try:
            dt = datetime.fromisoformat(to_date)
            filters.append(Case.created_at <= dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid to_date format. Use ISO 8601.")

    base_query = select(Case).where(and_(*filters)) if filters else select(Case)
    count_query = select(func.count(Case.id)).where(and_(*filters)) if filters else select(func.count(Case.id))

    try:
        total     = await db.scalar(count_query) or 0
        offset    = (page - 1) * page_size
        result    = await db.execute(
            base_query.options(selectinload(Case.reviewer))
            .order_by(Case.created_at.asc())
            .offset(offset).limit(page_size)
        )
        cases     = result.scalars().all()

    except Exception as e:
        logger.error(f"CASE FETCH ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    print("CURRENT USER:", current_user.id, current_user.role)
    logger.info("CASES_RETURNED: %d", len(cases))

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
        "cases":     [_case_to_dict(c) for c in cases],
    }


@router.get("/{case_id}")
async def get_case(
    case_id:      int,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Case).where(Case.id == case_id).options(selectinload(Case.reviewer))
    )
    case   = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    # Step 1: Mandatory Environment Isolation Check
    if case.is_demo != current_user.is_demo:
        logger.warning("ISOLATION_BREACH_ATTEMPT: user=%s (is_demo=%s) tried to access case=%d (is_demo=%s)",
                       current_user.email, current_user.is_demo, case_id, case.is_demo)
        raise HTTPException(status_code=403, detail="Forbidden: Environment mismatch.")

    # Step 2: Role-based access control
    if current_user.role == "CODER" and case.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied.")

    # Reviewer opening a submitted case moves it to in_review
    if current_user.role == "REVIEWER" and case.status in ("submitted", "under_review"):
        case.status = "in_review"
        if not case.assigned_to:
            case.assigned_to = current_user.id
            case.assigned_at = datetime.utcnow()
            case.assignment_status = "assigned"
        case.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(case)

    full = _case_to_dict(case)
    return full


@router.patch("/{case_id}/status")
async def update_case_status_route(
    case_id: int,
    payload: CaseStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Primary status transition endpoint for admin/reviewer UI."""
    logger.info(
        "STATUS_UPDATE_PAYLOAD: case_id=%s user_id=%s role=%s payload=%s",
        case_id,
        current_user.id,
        current_user.role,
        payload.model_dump(),
    )

    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders cannot update case status.")
    if payload.status is None:
        raise HTTPException(status_code=400, detail="Field 'status' is required.")

    new_status = payload.status.value

    try:
        assert_role_may_set(current_user.role, new_status)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    case = await _load_case(case_id, current_user, db)
    prev_status = status_for_response(case.status)
    feedback = _compose_feedback(payload) or (payload.feedback or payload.comment or "").strip()

    await _apply_status_change(case, new_status, current_user, db, feedback=feedback)
    await db.commit()
    await db.refresh(case)

    messages = {
        CaseStatus.approved.value: "Case approved successfully.",
        CaseStatus.rejected.value: "Review feedback submitted.",
        CaseStatus.in_review.value: "Case moved to In Review.",
        CaseStatus.submitted.value: "Case marked as Submitted.",
        CaseStatus.draft.value: "Case moved to Draft.",
    }
    logger.info(
        "STATUS_UPDATE_OK: case_id=%s %s -> %s by user_id=%s",
        case_id,
        prev_status,
        new_status,
        current_user.id,
    )
    return {
        "message": messages.get(new_status, "Case status updated."),
        "status": new_status,
        "case": _case_to_dict(case),
        "previous_status": prev_status,
    }


@router.patch("/{case_id}")
async def patch_case(
    case_id: int,
    payload: CaseStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """General patch — delegates status changes to the status route logic."""
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders cannot update case status.")

    case = await _load_case(case_id, current_user, db)
    prev = {"status": status_for_response(case.status), "priority": case.priority}

    if payload.status is not None:
        new_status = payload.status.value
        try:
            assert_role_may_set(current_user.role, new_status)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        feedback = _compose_feedback(payload) or (payload.feedback or payload.comment or "").strip()
        await _apply_status_change(case, new_status, current_user, db, feedback=feedback)

    if payload.priority:
        if payload.priority not in ("low", "normal", "high"):
            raise HTTPException(status_code=400, detail="Invalid priority value.")
        case.priority = payload.priority
        case.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(case)
    return {"message": "Case updated.", "case": _case_to_dict(case), "previous": prev}


@router.post("/{case_id}/submit")
async def submit_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _load_case(case_id, current_user, db)
    if case.status != "draft":
        raise HTTPException(status_code=400, detail=f"Case cannot be submitted from status '{case.status}'.")
    case.status = "submitted"
    case.updated_at = datetime.utcnow()
    from utils.assignment import auto_assign_reviewer
    await auto_assign_reviewer(db, case_id, current_user.id, is_demo=case.is_demo)
    await db.commit()
    return {"message": "Case submitted.", "status": case.status}


@router.post("/{case_id}/approve")
async def approve_case(
    case_id: int,
    review_confidence: float = Query(1.0, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders cannot approve cases.")
    case = await _load_case(case_id, current_user, db)
    await _apply_status_change(case, "approved", current_user, db)
    case.review_confidence = review_confidence
    await db.commit()
    return {"message": "Case approved successfully.", "status": "approved"}


@router.post("/{case_id}/reject")
async def reject_case(
    case_id: int,
    payload: RejectPayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders cannot reject cases.")
    case = await _load_case(case_id, current_user, db)
    feedback = _compose_feedback(payload)
    if not feedback:
        raise HTTPException(status_code=400, detail="Feedback is required when rejecting a case.")
    await _apply_status_change(case, "rejected", current_user, db, feedback=feedback)
    case.review_confidence = payload.review_confidence
    await db.commit()
    return {"message": "Review feedback submitted.", "status": "rejected"}


@router.post("/{case_id}/reopen")
async def reopen_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Only admins can re-open cases.")
    case = await _load_case(case_id, current_user, db)
    case.status = "in_review"
    case.updated_at = datetime.utcnow()
    await db.commit()
    return {"message": "Case re-opened for review.", "status": case.status}


@router.post("/{case_id}/assign")
async def assign_case(
    case_id: int,
    payload: AssignPayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role not in ("ADMIN", "REVIEWER"):
        raise HTTPException(status_code=403, detail="Forbidden.")
    case = await _load_case(case_id, current_user, db)
    case.assigned_to = payload.reviewer_id
    case.assigned_at = datetime.utcnow()
    case.assignment_status = "assigned"
    if case.status == "submitted":
        case.status = "in_review"
    case.updated_at = datetime.utcnow()
    await db.commit()
    return {"message": "Reviewer assigned.", "assigned_to": payload.reviewer_id}


@router.post("/{case_id}/update-codes")
async def update_case_codes(
    case_id: int,
    payload: UpdateCodesPayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders cannot update final codes.")
    case = await _load_case(case_id, current_user, db)
    case.final_code_set = json.dumps(payload.final_codes or [])
    if payload.justification:
        case.reviewer_notes = payload.justification
    case.updated_at = datetime.utcnow()
    await db.commit()
    return {"message": "Codes updated."}


@router.get("/{case_id}/audit")
async def get_case_audit_trail(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _load_case(case_id, current_user, db)
    from database.models import GovernanceLog
    result = await db.execute(
        select(GovernanceLog, User)
        .join(User, GovernanceLog.actor_id == User.id)
        .where(GovernanceLog.case_id == case_id)
        .order_by(GovernanceLog.timestamp.asc())
    )
    rows = result.all()
    trail = []
    for log, user in rows:
        prev = json.loads(log.previous_state) if log.previous_state else None
        new = json.loads(log.new_state) if log.new_state else None
        meta = json.loads(log.metadata_json) if log.metadata_json else {}
        trail.append({
            "action": log.action_type,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "user": user.name if user else f"User {log.actor_id}",
            "role": log.actor_role,
            "previous_state": prev,
            "new_state": new,
            "metadata": meta,
        })
    return trail


@router.delete("/{case_id}", dependencies=[Depends(require_admin)])
async def delete_case(case_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Case).where(Case.id == case_id))
    case   = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    await db.delete(case)
    await db.commit()
    return {"message": f"Case {case_id} deleted."}

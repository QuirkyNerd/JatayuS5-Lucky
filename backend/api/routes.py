import hashlib
import json
import re
import traceback
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

from config import settings
from database.db import get_db
from database.models import Document, AuditResult, AgentLog, FeedbackLog, Case, User
from schemas.audit import AuditRequest, FeedbackRequest
from services.audit_pipeline import AuditPipeline
from services.claim_values import ClaimValueEngine
from api.file_parser import FileParser
from security.auth import get_current_user
from utils.code_normalizer import deduplicate_codes
from utils.logging import get_logger, set_request_context, new_request_id
from utils.phi_encryptor import PHIEncryptor
from utils.governance import log_governance
from utils.public_labels import sanitize_audit_payload_for_api, sanitize_codes_list_for_api

logger = get_logger(__name__)
router = APIRouter()

_CACHE_ENABLED = settings.cache_max_size > 0 and settings.use_redis and REDIS_AVAILABLE
redis_client = None
if _CACHE_ENABLED:
    try:
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache client: {e}")
        _CACHE_ENABLED = False


def _compute_note_hash(note_text: str) -> str:
    normalised = re.sub(r"\s+", " ", note_text.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _make_cache_key(note_hash: str, human_codes: list[str]) -> str:
    codes_sorted = "|".join(sorted(c.upper() for c in human_codes))
    return f"{note_hash}::{codes_sorted}"


def _compute_risk_score(discrepancies: list[dict]) -> float:
    """Simple heuristic: high-severity disc → +30 pts, medium → +10, overcoding → +5."""
    score = 0.0
    for d in discrepancies:
        sev = d.get("severity", "low")
        if sev == "high":
            score += 30
        elif sev == "medium":
            score += 10
        else:
            score += 2
    return min(score, 100.0)


def _compute_revenue_impact(discrepancies: list[dict]) -> float:
    """Compute real revenue impact using CMS 2024 claim values from ClaimValueEngine."""
    missed_codes = [d.get("code", "") for d in discrepancies if d.get("type") == "missed_code"]
    overcoded_codes = [d.get("code", "") for d in discrepancies if d.get("type") == "unsupported_code"]
    impact = ClaimValueEngine.estimate_revenue_impact(missed_codes, overcoded_codes, currency="usd")
    return impact.get("net_impact", 0.0)


def _compute_accuracy(ai_codes: list[dict], discrepancies: list[dict]) -> float:
    """
    Compute coding accuracy using rule-engine validated codes as source of truth.
    Accuracy = (codes confirmed correct) / (total AI codes validated).
    Does NOT filter by confidence — rule engine is the source of truth.
    """
    correct = sum(1 for d in discrepancies if d.get("type") == "correct_code")
    total   = len(ai_codes)   # post-rule-engine validated set
    return round((correct / total) * 100, 1) if total else 0.0


_RATE_LIMIT = f"{settings.rate_limit_requests}/{settings.rate_limit_window_seconds}second"


@router.post("/audit", tags=["audit"])
async def run_audit(
    request: Request,
    payload: AuditRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    req_id = new_request_id()
    set_request_context(request_id=req_id)

    # ── RBAC Guard ──────────────────────────────────────────────────────────
    if current_user and current_user.role == "REVIEWER":
        raise HTTPException(
            status_code=403,
            detail="Reviewers cannot initiate automated audits. This feature is restricted to Coders and Admins."
        )

    # ── CASE LOCK CHECK ──────────────────────────────────────────────────
    if payload.case_id:
        result = await db.execute(select(Case).where(Case.id == payload.case_id))
        existing_case = result.scalar_one_or_none()
        if existing_case:
            # Step 1: Mandatory Environment Isolation Check
            if existing_case.is_demo != current_user.is_demo:
                 logger.warning("ISOLATION_BREACH_ATTEMPT (Audit): user=%s tried to access case=%d",
                                current_user.email, payload.case_id)
                 raise HTTPException(status_code=403, detail="Forbidden: Environment mismatch.")
            if existing_case.status != "draft":
                raise HTTPException(
                    status_code=403,
                    detail=f"This case has been {existing_case.status} and is locked for editing."
                )
            if existing_case.user_id != current_user.id and current_user.role == "CODER":
                 raise HTTPException(status_code=403, detail="Access denied.")

    logger.info(
        "POST /audit | human_codes=%d | user=%s | ip=%s",
        len(payload.human_codes),
        current_user.email if current_user else "anonymous",
        request.client.host if request.client else "unknown",
    )

    norm_human_codes = deduplicate_codes(payload.human_codes)
    note_hash        = _compute_note_hash(payload.note_text)
    cache_key        = _make_cache_key(note_hash, norm_human_codes)

    # ── Cache check ─────────────────────────────────────────────────────────
    if _CACHE_ENABLED and redis_client:
        try:
            cached_data_str = await redis_client.get(cache_key)
            if cached_data_str:
                cached = sanitize_audit_payload_for_api(json.loads(cached_data_str))
                logger.info("Cache HIT for note_hash=%s.", note_hash[:12])

                async def cached_stream():
                    yield f"data: {json.dumps({'event': 'info', 'data': 'Cache HIT – returning stored result'})}\\n\\n"
                    yield f"data: {json.dumps({'event': 'complete', 'data': {**cached, 'cache_hit': True}})}\\n\\n"

                return StreamingResponse(cached_stream(), media_type="text/event-stream")
        except Exception as e:
            logger.error("Redis cache error on GET: %s", e)

    logger.info("Cache MISS – running pipeline.")

    t_start = datetime.utcnow()

    # ── Pipeline SSE Generator ───────────────────────────────────────────────
    async def sse_generator():
        nonlocal t_start
        pipeline      = AuditPipeline()
        final_payload = None

        try:
            async for chunk in pipeline.run_stream(payload.note_text, norm_human_codes):
                if chunk.get("event") == "complete":
                    final_payload = chunk.get("data")
                else:
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
            yield f"data: {json.dumps({'event': 'error', 'data': str(exc)})}\n\n"
            return

        if final_payload:
            processing_time = (datetime.utcnow() - t_start).total_seconds()

            final_payload = sanitize_audit_payload_for_api(final_payload)
            ai_codes     = final_payload.get("ai_codes", [])
            discrepancies = final_payload.get("discrepancies", [])
            tokens_used  = final_payload.get("tokens_used", 0)
            cost_est     = tokens_used * (0.075 / 1_000_000)

            risk_score      = _compute_risk_score(discrepancies)
            revenue_impact  = _compute_revenue_impact(discrepancies)
            coding_accuracy = _compute_accuracy(ai_codes, discrepancies)
            avg_confidence  = (
                sum(c.get("confidence", 0) for c in ai_codes) / len(ai_codes)
                if ai_codes else 0.0
            )

            try:
                # ── Persist Case row ───────────────────────────────────────────
                case = Case(
                    user_id         = current_user.id,
                    org_id          = current_user.org_id,
                    input_text      = payload.note_text,
                    note_hash       = note_hash,
                    ai_codes        = json.dumps(ai_codes),
                    human_codes     = json.dumps(norm_human_codes),
                    discrepancies   = json.dumps(discrepancies),
                    evidence        = json.dumps(final_payload.get("evidence", [])),
                    pipeline_log    = json.dumps(final_payload.get("pipeline_log", [])),
                    risk_score      = risk_score,
                    revenue_impact  = revenue_impact,
                    coding_accuracy = coding_accuracy,
                    avg_confidence  = round(avg_confidence, 3),
                    processing_time = processing_time,
                    summary         = json.dumps({
                        "summary": final_payload.get("summary", ""),
                        "explanation": final_payload.get("explanation", ""),
                        "removed_codes": final_payload.get("removed_codes", []),
                    }),
                    model_used      = settings.groq_model_primary,
                    tokens_used     = tokens_used,
                    cost_estimate   = f"${cost_est:.5f}",
                    status          = "draft",
                    # ✅ STEP 1: FORCE CORRECT CASE CREATION
                    is_demo         = current_user.is_demo,
                )
                
                # ✅ STEP 1: VERIFY DATABASE SOURCE
                from config import settings as cfg_settings
                print("DB URL (Creation):", cfg_settings.database_url)

                db.add(case)
                
                # ✅ STEP 3: FORCE COMMIT
                await db.commit()
                await db.refresh(case)
                
                # ✅ SECTION 2 (DEMO): Auto-submit + auto-assign immediately
                if current_user.is_demo:
                    case.status = "submitted"
                    await db.commit()
                    from utils.assignment import auto_assign_reviewer
                    assigned_id = await auto_assign_reviewer(
                        db, case.id, current_user.id, is_demo=True
                    )
                    await db.commit()
                    await db.refresh(case)
                    print(f"CASE CREATED (demo): id={case.id}, assigned_to={case.assigned_to}, status={case.status}")
                else:
                    print(f"CASE CREATED: id={case.id}, status={case.status}, assigned_to={case.assigned_to}")
                
                # ✅ STEP 2: VERIFY CASE INSERT
                stmt_verify = select(Case).where(Case.id == case.id)
                res_verify = await db.execute(stmt_verify)
                saved = res_verify.scalar_one_or_none()
                if saved:
                    print(f"AFTER INSERT: ID={saved.id} is_demo={saved.is_demo} assigned_to={saved.assigned_to}")
                else:
                    print(f"CRITICAL ERROR: CASE {case.id} NOT FOUND IN DB AFTER COMMIT!")

                await log_governance(
                    db, "create", current_user.id, current_user.role, 
                    case_id=case.id,
                    new_state={"status": "draft"},
                    metadata="Case created via automated audit."
                )

                # ── Legacy persist (backward compat) ───────────────────────────
                encrypted_note = PHIEncryptor.encrypt(payload.note_text)
                doc = Document(
                    note_text=encrypted_note, note_hash=note_hash,
                    human_codes=json.dumps(norm_human_codes),
                )
                db.add(doc)
                await db.flush()

                audit_rec = AuditResult(
                    document_id=doc.id,
                    ai_codes=json.dumps(ai_codes),
                    discrepancies=json.dumps(discrepancies),
                    evidence=json.dumps(final_payload.get("evidence", [])),
                    summary=json.dumps({
                        "summary": final_payload.get("summary", ""),
                        "explanation": final_payload.get("explanation", ""),
                        "removed_codes": final_payload.get("removed_codes", []),
                    }),
                    tokens_used=tokens_used,
                    cost_estimate=f"${cost_est:.5f}",
                )
                db.add(audit_rec)
                await db.flush()  # Step 2b: Flush to generate audit_rec ID
                
                log_entry = AgentLog(
                    audit_id=audit_rec.id,  # Points to audit_results per model constraint
                    pipeline_log=json.dumps(final_payload.get("pipeline_log", [])),
                )
                db.add(log_entry)
                await db.flush()
                
                print("LOG LINKED TO:", log_entry.audit_id)
                
                # Step 4: Final commit
                await db.commit()
                await db.refresh(case)
                
                print(f"CASE SAVED: {case.id} {case.user_id}")
                logger.info("Case #%d persisted. Risk=%.1f Revenue=$%.2f", case.id, risk_score, revenue_impact)
                
                # Now yield the final completion event to the stream after successful persistence
                yield f"data: {json.dumps({'event': 'complete', 'data': final_payload})}\n\n"
                
            except Exception as e:
                print("CASE SAVE FAILED:", str(e))
                logger.error(f"CASE SAVE FAILED: {str(e)}")
                yield f"data: {json.dumps({'event': 'error', 'data': 'Case persistence failed: ' + str(e)})}\n\n"
                return

            # ── Cache persist ──────────────────────────────────────────────
            if _CACHE_ENABLED and redis_client:
                try:
                    cache_val = sanitize_audit_payload_for_api({
                        "ai_codes":           ai_codes,
                        "low_confidence_codes": final_payload.get("low_confidence_codes", []),
                        "discrepancies":      discrepancies,
                        "evidence":           final_payload.get("evidence", []),
                        "summary":            final_payload.get("summary", ""),
                        "removed_codes":      final_payload.get("removed_codes", []),
                    })
                    await redis_client.setex(cache_key, 3600 * 24, json.dumps(cache_val))
                except Exception as e:
                    logger.error("Redis cache error on SET: %s", e)

            logger.info("Case #%d persisted. Risk=%.1f Revenue=$%.2f", case.id, risk_score, revenue_impact)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.post("/audit/file", tags=["audit"])
async def run_audit_file(
    request: Request,
    file: UploadFile = File(...),
    human_codes: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        note_text = await FileParser.parse_file(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    try:
        parsed_codes = json.loads(human_codes)
        if not isinstance(parsed_codes, list):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="human_codes must be a JSON list of strings")

    payload = AuditRequest(note_text=note_text, human_codes=parsed_codes)
    return await run_audit(request, payload, db, current_user)


@router.post("/feedback", tags=["feedback"])
async def submit_feedback(
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        fb = FeedbackLog(note_hash=payload.note_hash, ai_code=payload.ai_code, decision=payload.decision)
        db.add(fb)
        await db.commit()
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        logger.error("Failed to record feedback: %s", e)
        raise HTTPException(status_code=500, detail="Failed to record feedback")



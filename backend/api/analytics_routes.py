"""
api/analytics_routes.py – Analytics engine for CodePerfectAuditor.

Computes metrics dynamically from the Case table (not a snapshot table).
Optionally caches results in Redis (1-min TTL for fast dashboard loads).

Endpoints:
  GET /analytics/overview  – KPI summary metrics
  GET /analytics/trends    – time-series data for charts
"""

import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

try:
    # When running from project root (development)
    from database.db import get_db
    from database.models import Case, User
    from security.auth import get_current_user
    from services.claim_values import USD_TO_INR
    from utils.logging import get_logger
except ImportError:
    # When running from backend directory (Docker/production)
    from database.db import get_db
    from database.models import Case, User
    from security.auth import get_current_user
    from services.claim_values import USD_TO_INR
    from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


def _tenant_filter(current_user: User):
    """Returns SQLAlchemy filter conditions for tenant isolation and demo sandbox."""
    if current_user.is_demo:
        return [Case.is_demo == True]
    
    # Real users see ONLY real data
    filters = [Case.is_demo == False]
    if current_user.role == "CODER":
        filters.append(Case.user_id == current_user.id)
    return filters


@router.get("/overview")
async def analytics_overview(
    days:         int  = Query(30, ge=1, le=365, description="Lookback window in days"),
    currency:     str  = Query("usd", description="Currency for revenue values: 'usd' or 'inr'"),
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Returns KPI summary metrics derived dynamically from the Case table.
    Supports ?currency=usd (default) or ?currency=inr for INR conversion.
    """
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders do not have access to analytics.")

    since   = datetime.utcnow() - timedelta(days=days)
    filters = _tenant_filter(current_user) + [Case.created_at >= since]

    # ── Core counts ──────────────────────────────────────────────────────────
    total_cases = await db.scalar(
        select(func.count(Case.id)).where(and_(*filters))
    ) or 0

    # Avg confidence across all ai_codes JSON fields
    # We compute Python-side to avoid DB JSON function portability issues
    result = await db.execute(
        select(Case.ai_codes, Case.discrepancies, Case.risk_score, Case.revenue_impact, Case.coding_accuracy, Case.status, Case.processing_time)
        .where(and_(*filters))
    )
    rows = result.all()

    total_revenue_impact = 0.0
    total_confidence = 0.0
    confidence_count = 0
    high_risk_cases  = 0
    undercoding      = 0  # missed_code discrepancies
    overcoding       = 0  # unsupported_code discrepancies
    correct_codes    = 0
    accuracy_sum     = 0.0
    status_counts    = {"draft": 0, "submitted": 0, "under_review": 0, "approved": 0, "rejected": 0}
    processing_times = []

    for ai_json, disc_json, risk, revenue, acc, status, processing_time in rows:
        codes = json.loads(ai_json or "[]")
        discs = json.loads(disc_json or "[]")

        for c in codes:
            conf = c.get("confidence", 0)
            total_confidence += conf
            confidence_count += 1

        def _is_valid_billable(code: str) -> bool:
            """Exclude R-codes (symptoms), Z-codes (screening), and diagnostic CPTs from KPI."""
            c = code.upper().strip()
            if c.startswith("R") or c.startswith("Z"):
                return False
            diagnostic_cpts = {
                "71045","71046","71047","71048","93306","93307","93308",
                "85025","85027","80053","80048","93000","93005","93010",
                "74176","74177","74178","70450","70460","70470",
            }
            if c in diagnostic_cpts:
                return False
            return True

        symptom_suppressed = 0
        diagnostic_suppressed = 0

        for d in discs:
            dtype = d.get("type", "")
            code_str = d.get("code", "")
            if dtype == "missed_code":
                if _is_valid_billable(code_str):
                    undercoding += 1
                elif code_str.upper().startswith("R"):
                    symptom_suppressed += 1
                else:
                    diagnostic_suppressed += 1
            elif dtype == "unsupported_code":
                overcoding += 1
            elif dtype == "correct_code":
                correct_codes += 1

        if (risk or 0) >= 70:
            high_risk_cases += 1
        total_revenue_impact += revenue or 0.0
        accuracy_sum += acc or 0.0
        
        # Governance
        status_counts[status] = status_counts.get(status, 0) + 1
        if processing_time:
            processing_times.append(processing_time)

    avg_confidence   = round(total_confidence / confidence_count, 3) if confidence_count else 0.0
    avg_accuracy     = round(accuracy_sum / total_cases, 1) if total_cases else 0.0
    
    # Approval/Rejection rates
    resolved_cases   = status_counts["approved"] + status_counts["rejected"]
    approval_rate    = round((status_counts["approved"] / resolved_cases) * 100, 1) if resolved_cases else 0.0
    rejection_rate   = round((status_counts["rejected"] / resolved_cases) * 100, 1) if resolved_cases else 0.0
    avg_proc_time    = round(sum(processing_times) / len(processing_times), 2) if processing_times else 0.0

    # ── Review SLA Metrics ───────────────────────────────────────────
    review_durations_res = await db.execute(
        select(Case.review_duration).where(and_(Case.review_duration != None, *filters))
    )
    rev_durations = [r[0] for r in review_durations_res.all()]
    avg_review_dur = round(sum(rev_durations) / len(rev_durations), 1) if rev_durations else 0.0
    
    # Priority Metrics
    high_priority_count = await db.scalar(
        select(func.count(Case.id)).where(and_(Case.priority == 'high', *filters))
    ) or 0
    
    priority_sla_res = await db.execute(
        select(Case.priority, func.avg(Case.review_duration))
        .where(and_(Case.review_duration != None, *filters))
        .group_by(Case.priority)
    )
    priority_sla = {p: round(avg or 0, 1) for p, avg in priority_sla_res.all()}

    # Workload Distribution (Active cases per reviewer)
    # Mandatory Environment Isolation
    workload_res = await db.execute(
        select(User.name, func.count(Case.id))
        .join(Case, Case.assigned_to == User.id)
        .where(and_(
            Case.status.in_(["submitted", "under_review"]), 
            User.is_demo == current_user.is_demo,
            *filters
        ))
        .group_by(User.name)
    )
    workload_dist = {name: count for name, count in workload_res.all()}

    fx = USD_TO_INR if currency.lower() == "inr" else 1.0
    symbol = "\u20b9" if currency.lower() == "inr" else "$"

    return {
        "status": "success",
        "period_days": days,
        "currency": currency.upper(),
        "currency_symbol": symbol,
        "data": {
            "total_cases":         total_cases,
            "coding_accuracy_pct": avg_accuracy,
            "avg_confidence":      avg_confidence,
            "total_revenue_impact": round(total_revenue_impact * fx, 2),
            "high_risk_cases":     high_risk_cases,
            "undercoding_count":   undercoding,
            "overcoding_count":    overcoding,
            "correct_code_count":  correct_codes,
            "approval_rate_pct":   approval_rate,
            "rejection_rate_pct":  rejection_rate,
            "avg_processing_time": avg_proc_time,
            "avg_review_duration": avg_review_dur,
            "high_priority_count": high_priority_count,
            "priority_sla":        priority_sla,
            "workload_distribution": workload_dist,
            "status_distribution": status_counts
        },
    }


@router.get("/trends")
async def analytics_trends(
    days:         int  = Query(30, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Returns daily time-series data for charting (cases per day, revenue, risk).
    """
    if current_user.role == "CODER":
        raise HTTPException(status_code=403, detail="Coders do not have access to analytics.")

    since   = datetime.utcnow() - timedelta(days=days)
    filters = _tenant_filter(current_user) + [Case.created_at >= since]

    result = await db.execute(
        select(Case.created_at, Case.risk_score, Case.revenue_impact, Case.coding_accuracy)
        .where(and_(*filters))
        .order_by(Case.created_at)
    )
    rows = result.all()

    # Bucket by date
    daily: dict[str, dict] = {}
    for created_at, risk, revenue, accuracy in rows:
        date_key = created_at.strftime("%Y-%m-%d")
        if date_key not in daily:
            daily[date_key] = {"date": date_key, "cases": 0, "revenue": 0.0, "avg_risk": 0.0, "risks": [], "accuracy": []}
        daily[date_key]["cases"] += 1
        daily[date_key]["revenue"] += revenue or 0.0
        daily[date_key]["risks"].append(risk or 0.0)
        daily[date_key]["accuracy"].append(accuracy or 0.0)

    trend_data = []
    for day in sorted(daily.keys()):
        d = daily[day]
        risks = d.pop("risks")
        accs  = d.pop("accuracy")
        d["avg_risk"]     = round(sum(risks) / len(risks), 1) if risks else 0.0
        d["avg_accuracy"] = round(sum(accs)  / len(accs),  1) if accs  else 0.0
        d["revenue"]      = round(d["revenue"], 2)
        trend_data.append(d)

    return {"status": "success", "trends": trend_data}

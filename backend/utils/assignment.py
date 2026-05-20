import logging
from datetime import datetime
from sqlalchemy import select, and_, not_, exists
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import User, Case
from utils.governance import log_governance

logger = logging.getLogger(__name__)


# ─── CODER AVAILABILITY ──────────────────────────────────────────────────────
async def find_available_coder(db: AsyncSession, is_demo: bool = False) -> User | None:
    """
    Returns the first CODER who has no active (non-approved) case.
    A coder is BUSY if they have any case where status != 'approved'.
    """
    busy_subq = (
        select(Case.user_id)
        .where(
            and_(
                Case.user_id == User.id,
                Case.status != "approved",
                Case.is_demo == is_demo,
            )
        )
        .correlate(User)
        .exists()
    )
    stmt = (
        select(User)
        .where(
            and_(
                User.role == "CODER",
                User.is_active == True,
                User.is_demo == is_demo,
                not_(busy_subq),
            )
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    coder = result.scalar_one_or_none()
    print(f"AVAILABLE CODER: {coder.id if coder else None} (is_demo={is_demo})")
    return coder


# ─── REVIEWER AVAILABILITY ───────────────────────────────────────────────────
async def find_available_reviewer(db: AsyncSession, is_demo: bool = False) -> User | None:
    """
    Returns the first REVIEWER who has no active case.
    A reviewer is BUSY if they have any case where status NOT IN ('approved', 'rejected').
    """
    busy_subq = (
        select(Case.assigned_to)
        .where(
            and_(
                Case.assigned_to == User.id,
                Case.status.not_in(["approved", "rejected"]),
                Case.is_demo == is_demo,
            )
        )
        .correlate(User)
        .exists()
    )
    stmt = (
        select(User)
        .where(
            and_(
                User.role == "REVIEWER",
                User.is_active == True,
                User.is_demo == is_demo,
                not_(busy_subq),
            )
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    reviewer = result.scalar_one_or_none()
    print(f"AVAILABLE REVIEWER: {reviewer.id if reviewer else None} (is_demo={is_demo})")
    return reviewer


# ─── AUTO ASSIGN (called after case is created) ──────────────────────────────
async def auto_assign_reviewer(
    db: AsyncSession,
    case_id: int,
    trigger_actor_id: int,
    is_demo: bool = False,
) -> int | None:
    """
    Assign the first available reviewer to the given case.
    Reviewer must have no active (non-approved/rejected) cases.
    """
    try:
        reviewer = await find_available_reviewer(db, is_demo=is_demo)

        if not reviewer:
            logger.warning(f"AUTO-ASSIGN: No available reviewer (is_demo={is_demo}) for Case #{case_id}")
            print(f"AUTO-ASSIGN WARNING: No free reviewer for Case #{case_id}")
            return None

        # Fetch case
        case_res = await db.execute(select(Case).where(Case.id == case_id))
        case = case_res.scalar_one_or_none()
        if not case:
            return None

        case.assigned_to      = reviewer.id
        case.assigned_at      = datetime.utcnow()
        case.assignment_status = "assigned"

        print(f"AUTO-ASSIGN: Case #{case_id} → Reviewer #{reviewer.id}")

        await log_governance(
            db,
            action_type="auto_assign",
            actor_id=trigger_actor_id,
            actor_role="system",
            case_id=case_id,
            metadata={
                "assigned_to": reviewer.id,
                "strategy":    "first_available",
                "is_demo":     is_demo,
            }
        )

        logger.info(f"AUTO-ASSIGN: Case #{case_id} → Reviewer #{reviewer.id}")
        return reviewer.id

    except Exception as e:
        logger.error(f"AUTO-ASSIGN ERROR for Case #{case_id}: {str(e)}")
        print(f"AUTO-ASSIGN EXCEPTION: {str(e)}")
        return None

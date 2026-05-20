import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import GovernanceLog

async def log_governance(
    db: AsyncSession,
    action_type: str,
    actor_id: int,
    actor_role: str,
    case_id: int | None = None,
    previous_state: dict | str | None = None,
    new_state: dict | str | None = None,
    metadata: dict | str | None = None
):
    """
    Utility to record enterprise governance/audit events.
    """
    if isinstance(previous_state, (dict, list)):
        previous_state = json.dumps(previous_state)
    if isinstance(new_state, (dict, list)):
        new_state = json.dumps(new_state)
    if isinstance(metadata, (dict, list)):
        metadata = json.dumps(metadata)
    
    log = GovernanceLog(
        case_id=case_id,
        actor_id=actor_id,
        actor_role=actor_role.lower(),
        action_type=action_type,
        timestamp=datetime.utcnow(),
        previous_state=previous_state,
        new_state=new_state,
        metadata_json=metadata
    )
    db.add(log)
    await db.flush()

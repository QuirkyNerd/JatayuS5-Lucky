import logging
import os
import time
from fastapi import APIRouter, Depends, HTTPException, Query
from security.auth import get_current_user, require_admin
from database.models import User
from services.evaluation_engine import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["admin", "evaluation"])

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_standardized.json")

@router.get("", dependencies=[Depends(require_admin)])
async def get_system_evaluation(
    force_refresh: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """
    Returns system performance metrics against benchmark dataset.
    Only accessible by Admin. Supports persistent disk caching.
    """
    logger.info(f"Evaluation requested by Admin: {current_user.email} (force_refresh={force_refresh})")
    
    try:
        # Task 4 & 6: Trigger evaluation (engine handles persistence/cache internally)
        result = await run_evaluation(BENCHMARK_PATH, force_refresh=force_refresh)
        
        if result.get("status") == "error":
            logger.error(f"Evaluation returned error: {result.get('message')}")
            raise HTTPException(status_code=500, detail=result.get("message"))
            
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected evaluation route error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during evaluation: {str(e)}"
        )

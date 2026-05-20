from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database.db import get_db
from config import settings
from utils.logging import get_logger
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

# Initialize Redis client for health checks if enabled and library exists
redis_client = None
if settings.use_redis and REDIS_AVAILABLE:
    try:
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to initialize Redis client: {e}")
        redis_client = None

@router.get("")
async def health_summary(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check for frontend and monitoring."""
    db_status = "disconnected"
    chroma_status = "disconnected"
    redis_status = "disconnected"

    # 1. Database Check
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    # 2. Vector DB (Chroma) Check
    try:
        from services.rag_engine import RAGEngine
        RAGEngine().client.heartbeat()
        chroma_status = "connected"
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")

    # 3. Redis Check
    try:
        if redis_client and await redis_client.ping():
            redis_status = "connected"
        elif not settings.use_redis:
            redis_status = "disabled"
        elif not REDIS_AVAILABLE:
            redis_status = "not_installed"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    overall_status = "ok" if db_status == "connected" and redis_status in ("connected", "disabled") else "degraded"
    
    return {
        "status": overall_status,
        "database": db_status,
        "vector_db": chroma_status,
        "redis": redis_status,
        "service": "CodePerfectAuditor",
        "version": "2.5.0"
    }

@router.get("/live")
async def liveness():
    """Simple liveness probe for Docker/K8s."""
    return {"status": "healthy"}

@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Readiness probe checking critical dependencies."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {"status": "unready", "database": "disconnected"}

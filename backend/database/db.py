"""
database/db.py – Async SQLAlchemy engine and session management.

Optimized for PostgreSQL with connection pooling and stability checks.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from config import settings
from database.models import Base
from utils.logging import get_logger

logger = get_logger(__name__)

# -------------------------------
# 🗄️ Engine (Postgres optimized)
# -------------------------------
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,      # Verify connection health before use
    pool_size=10,            # Balanced pool size
    max_overflow=20,         # Allow temporary spikes
)

# -------------------------------
# 📦 Session factory
# -------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# -------------------------------
# 🚀 Init DB
# -------------------------------
async def init_db() -> None:
    """Create all tables defined in models.py (idempotent)."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 🚀 Seed initial data (Local import to avoid circular dependency)
        try:
            from database.seed_users import seed
            await seed()
        except Exception as seed_err:
            logger.error(f"⚠️ Seeding failed: {seed_err}")
            # We continue startup even if seeding fails

        logger.info("✅ Database schema verified / created")
        logger.info(f"🔗 DB Connected: {settings.database_url.split('@')[-1]}")

    except Exception as e:
        logger.error("❌ Database initialization failed")
        logger.error(str(e))
        raise


# -------------------------------
# 🔄 Dependency (FastAPI)
# -------------------------------
async def get_db():
    """
    FastAPI dependency that yields an AsyncSession per request.
    """
    from sqlalchemy.exc import InterfaceError, OperationalError
    async with AsyncSessionLocal() as session:
        try:
            yield session
            if session.is_active:
                try:
                    await session.commit()
                except (InterfaceError, OperationalError) as conn_err:
                    logger.warning(f"⚠️ Connection closed on commit: {conn_err}")
        except (InterfaceError, OperationalError) as e:
            logger.warning(f"⚠️ DB connection closed or timed out during request: {str(e)}")
        except Exception as e:
            if session.is_active:
                try:
                    await session.rollback()
                except (InterfaceError, OperationalError) as rollback_conn_err:
                    logger.warning(f"⚠️ Connection closed on rollback: {rollback_conn_err}")
                except Exception as rollback_err:
                    logger.error(f"❌ DB rollback failed: {str(rollback_err)}")
            logger.error(f"❌ DB transaction failed: {str(e)}")
            raise
        finally:
            try:
                await session.close()
            except (InterfaceError, OperationalError) as close_conn_err:
                logger.warning(f"⚠️ Connection closed on session close: {close_conn_err}")
            except Exception as close_err:
                logger.warning(f"⚠️ Failed to close DB session cleanly: {str(close_err)}")
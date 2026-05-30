"""
utils/qdrant_keepalive.py
─────────────────────────
Lightweight Qdrant keep-alive mechanism for free-tier instances.

Sends a minimal `get_collections()` ping every 12 hours to prevent
Qdrant Cloud free-tier inactivity suspension.

• Zero extra dependencies  – uses asyncio + qdrant-client (already in requirements.txt)
• Non-blocking            – runs as a background asyncio task
• Auto-starts via lifespan – integrated into FastAPI's @asynccontextmanager lifespan
"""

import asyncio
import traceback
from datetime import datetime, timezone


_KEEPALIVE_INTERVAL_SECONDS = 12 * 60 * 60  # 12 hours
_TAG = "[QDRANT_KEEPALIVE]"

_keepalive_task: asyncio.Task | None = None


def _now_utc() -> str:
    """Return current UTC time as a human-readable string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


async def _ping_loop(qdrant_url: str, qdrant_api_key: str) -> None:
    """
    Background coroutine that pings Qdrant every 12 hours.
    Runs until cancelled (via task.cancel() on shutdown).
    """
    # Import here to avoid circular imports at module load time
    from qdrant_client import QdrantClient

    print(f"{_TAG} Keep-alive scheduler started — interval=12h | target={qdrant_url}")

    while True:
        # ── Wait 12 hours before the first ping so startup connectivity
        #    check in main.py already serves as the initial verification.
        await asyncio.sleep(_KEEPALIVE_INTERVAL_SECONDS)

        print(f"{_TAG} Pinging Qdrant...")
        try:
            # Use a short timeout — we only need to know the cluster is alive
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=15.0,
            )
            collections = client.get_collections()
            col_names = [c.name for c in collections.collections]
            print(f"{_TAG} Success - {_now_utc()} | collections={col_names}")
        except asyncio.CancelledError:
            # Propagate cancellation so the task can shut down cleanly
            raise
        except Exception as exc:
            print(f"{_TAG} ERROR - {_now_utc()}")
            print(f"{_TAG} Exception: {exc}")
            traceback.print_exc()
            # Do NOT re-raise — keep pinging on the next interval


def start_keepalive(qdrant_url: str, qdrant_api_key: str) -> None:
    """
    Schedule the keep-alive background task on the running asyncio event loop.

    Call this inside FastAPI's lifespan (after `await init_db()`).
    The task is stored in a module-level variable so it can be cancelled on shutdown.

    Args:
        qdrant_url:     The Qdrant Cloud cluster URL  (from settings.qdrant_url)
        qdrant_api_key: The Qdrant API key            (from settings.qdrant_api_key)
    """
    global _keepalive_task

    if not qdrant_url:
        print(f"{_TAG} QDRANT_URL not set — keep-alive disabled.")
        return

    _keepalive_task = asyncio.create_task(
        _ping_loop(qdrant_url, qdrant_api_key),
        name="qdrant_keepalive",
    )
    print(f"{_TAG} Background task created successfully.")


def stop_keepalive() -> None:
    """
    Cancel the keep-alive background task.

    Call this inside the shutdown section of FastAPI's lifespan (after `yield`).
    """
    global _keepalive_task

    if _keepalive_task and not _keepalive_task.done():
        _keepalive_task.cancel()
        print(f"{_TAG} Keep-alive task cancelled (shutdown).")
    _keepalive_task = None

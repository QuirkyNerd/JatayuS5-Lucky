import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings
from database.db import init_db
from api.routes import router as audit_router
from api.auth_routes import router as auth_router
from api.case_routes import router as case_router
from api.analytics_routes import router as analytics_router
from api.health import router as health_router
from api.admin_routes import router as admin_router
from utils.logging import get_logger
from utils.qdrant_keepalive import start_keepalive, stop_keepalive

logger = get_logger(__name__)

_window = f"{settings.rate_limit_window_seconds}second"
_limit_str = f"{settings.rate_limit_requests}/{_window}"
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Auditor Platform")
    await init_db()
    logger.info("Database initialized successfully")

    # ── Qdrant Cloud connectivity verification ─────────────────────────────
    is_qdrant = settings.vector_db_provider.lower() == "qdrant"
    if is_qdrant or settings.qdrant_url:
        try:
            from qdrant_client import QdrantClient
            _q = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=15.0,
            )
            collections = _q.get_collections()
            col_names = [c.name for c in collections.collections]
            logger.info(f"QDRANT_CONNECTED_OK | url={settings.qdrant_url} | collections={col_names}")
        except Exception as qe:
            import traceback
            logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | component=QDRANT | error={qe}")
            traceback.print_exc()
            logger.exception("FULL_TRACEBACK")
            if is_qdrant:
                raise RuntimeError(f"Qdrant connection failed: {qe}")
    else:
        logger.warning("QDRANT_URL not set — using local ChromaDB (not production-ready)")

    # ── Qdrant keep-alive scheduler (prevents free-tier inactivity suspension) ──
    start_keepalive(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )

    # ── RAG engine init (loads all models: embedding, reranker, SapBERT) ──
    import time as _time
    t_rag_start = _time.perf_counter()
    from services.rag_engine import get_rag_engine
    rag = get_rag_engine()
    rag_load_ms = round((_time.perf_counter() - t_rag_start) * 1000, 1)
    
    counts = rag.collection_counts()
    logger.info(f"COLLECTION_COUNTS: {counts}")

    if is_qdrant:
        logger.info("══════════════════════════════")
        logger.info("VECTOR DATABASE STATUS")
        logger.info("══════════════════════════════")
        logger.info(f"ICD10_COUNT       = {counts.get('icd10', 0)}")
        logger.info(f"CPT_COUNT         = {counts.get('cpt', 0)}")
        logger.info(f"GUIDELINES_COUNT  = {counts.get('guidelines', 0)}")
        logger.info(f"SYMPTOMS_COUNT    = {counts.get('symptoms', 0)}")
        logger.info("══════════════════════════════")
        
        total_counts = sum(counts.values())
        if total_counts == 0:
            logger.error("BOOTSTRAP_FAILURE: All Qdrant collections are empty!")
            raise RuntimeError("BOOTSTRAP_FAILURE: All Qdrant collections are empty!")

    # ── HuggingFace model readiness check ─────────────────────────────────
    if rag.embedding_service and rag.embedding_service.local_model:
        logger.info(f"HF_MODEL_READY | model={getattr(settings, 'embedding_model', 'unknown')}")
    else:
        logger.info("HF_MODEL_READY | mode=REMOTE_API")

    # ── Production readiness summary ──────────────────────────────────────
    from services.validation_utils import PIPELINE_DEBUG_MODE
    active_backend = "QDRANT" if rag.q_client else "CHROMADB"
    logger.info("═══════════════════════════════════════════════════════")
    logger.info("PRODUCTION READINESS SUMMARY")
    logger.info("═══════════════════════════════════════════════════════")
    logger.info(f"  ACTIVE_VECTOR_BACKEND  = {active_backend}")
    logger.info(f"  RERANKER               = LOADED")
    logger.info(f"  SAPBERT                = LOADED ({settings.sapbert_model})")
    logger.info(f"  EMBEDDING              = {'LOCAL' if rag.embedding_service.use_local else 'REMOTE'}")
    logger.info(f"  RAG_TOP_K              = {settings.rag_top_k}")
    logger.info(f"  PIPELINE_DEBUG_MODE    = {PIPELINE_DEBUG_MODE}")
    logger.info(f"  CORS_ORIGINS           = {settings.cors_origins}")
    logger.info(f"  PORT                   = {os.environ.get('PORT', '8000')}")
    logger.info(f"  RAG_LOAD_TIME          = {rag_load_ms}ms")
    logger.info("═══════════════════════════════════════════════════════")
    yield

    stop_keepalive()
    logger.info("Shutting down Auditor Platform")

app = FastAPI(
    title="Auditor Platform",
    version="2.5.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Limit: {_limit_str} per IP."
        },
    )
    
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    import time
    from utils.logging import set_request_context, new_request_id
    
    request_id = request.headers.get("X-Request-ID", new_request_id())
    set_request_context(request_id=request_id)
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            "Request processed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": int(process_time * 1000),
            }
        )
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

app.include_router(audit_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(case_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "Auditor Platform",
        "version": "2.5.0",
        "status": "healthy"
    }
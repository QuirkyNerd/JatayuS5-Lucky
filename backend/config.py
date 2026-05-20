"""
config.py – Central application configuration for CodePerfectAuditor.

Loads settings from environment variables / .env file using pydantic-settings.
All agents and services import from this module to avoid hardcoded values.
"""

import json
from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# Resolve the project root (parent of the backend directory)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables or .env file."""


    # Groq Configuration (ENTERPRISE THREE-TIER ARCHITECTURE)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model_primary: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL_PRIMARY")
    groq_model_fast: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL_FAST")
    
    # LLM Runtime Toggles
    benchmark_mode: bool = Field(default=False, env="BENCHMARK_MODE")
    enable_llm_reasoning: bool = Field(default=False, env="ENABLE_LLM_REASONING")
    enable_groq_fallbacks: bool = Field(default=True, env="ENABLE_GROQ_FALLBACKS")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    sapbert_model: str = Field(default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", env="SAPBERT_MODEL")


    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@postgres:5432/codeperfect",
        env="DATABASE_URL",
    )

    redis_url: str = Field(
        default="redis://redis:6379/0",
        env="REDIS_URL",
    )
    use_redis: bool = Field(default=True, env="USE_REDIS")


    secret_key: str = Field(default="dev-secret-keep-it-32-chars-long", env="SECRET_KEY")
    phi_encryption_key: str = Field(default="dev-encryption-key-32-chars-long", env="PHI_ENCRYPTION_KEY")

    qdrant_url: str = Field(default="", env="QDRANT_URL")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    vector_db_provider: str = Field(default="chromadb", env="VECTOR_DB_PROVIDER")

    chroma_persist_dir: str = Field(
        default="backend/backend/chroma_db",
        env="CHROMA_PERSIST_DIR",
    )
    chroma_collection_icd: str = Field(
        default="icd10_codes", env="CHROMA_COLLECTION_ICD"
    )
    chroma_collection_cpt: str = Field(
        default="cpt_codes", env="CHROMA_COLLECTION_CPT"
    )
    chroma_collection_guidelines: str = Field(
        default="coding_guidelines", env="CHROMA_COLLECTION_GUIDELINES"
    )
    chroma_collection_symptoms: str = Field(
        default="symptoms", env="CHROMA_COLLECTION_SYMPTOMS"
    )

    rag_hybrid_alpha: float = Field(default=0.8, env="RAG_HYBRID_ALPHA")
    rag_hybrid_beta: float = Field(default=0.2, env="RAG_HYBRID_BETA")
    rag_top_k: int = Field(default=10, env="RAG_TOP_K")
    min_code_confidence: float = Field(default=0.65, env="MIN_CODE_CONFIDENCE")
    agent_max_retries: int = Field(default=2, env="AGENT_MAX_RETRIES")
    
    # Embedding Ingestion Hardening
    use_local_embeddings: bool = Field(default=True, env="USE_LOCAL_EMBEDDINGS")
    embedding_batch_size: int = Field(default=50, env="EMBEDDING_BATCH_SIZE")
    embedding_retry_limit: int = Field(default=5, env="EMBEDDING_RETRY_LIMIT")
    embedding_timeout: int = Field(default=120, env="EMBEDDING_TIMEOUT")
    embedding_concurrency: int = Field(default=2, env="EMBEDDING_CONCURRENCY")

    rate_limit_requests: int = Field(default=10, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, env="RATE_LIMIT_WINDOW_SECONDS")
    cache_max_size: int = Field(default=128, env="CACHE_MAX_SIZE")
    cors_origins: List[str] = Field(
        default=[
            "https://codeperfect-audit.vercel.app",
            "https://code-perfect-auditor-v2.vercel.app",
            "https://code-perfect-auditor-v2-e8fsrac9f-quirkynerds-projects.vercel.app",
            "http://161.118.217.29:3000",
            "http://localhost:3000"
        ],
        env="CORS_ORIGINS",
    )

    data_dir: str = Field(default="/app/backend/data", env="DATA_DIR")
    prompts_dir: str = Field(default="/app/backend/prompts", env="PROMPTS_DIR")
    
    # Persistent Ingestion Checkpoints
    checkpoint_dir: str = Field(default="/app/backend/data/checkpoints", env="CHECKPOINT_DIR")
    
    # Internal project root (for path resolution)
    root_dir: Path = Field(default=ROOT_DIR)
    
    

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        """
        Allows CORS_ORIGINS to be passed as JSON string or comma-separated list.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("data_dir", "prompts_dir", "chroma_persist_dir", "checkpoint_dir", mode="after")
    @classmethod
    def resolve_paths(cls, v: str) -> str:
        """
        Ensures paths are absolute. If a relative path is provided (e.g. from .env),
        it is resolved relative to the project ROOT_DIR to avoid CWD-dependent bugs.
        """
        path = Path(v)
        if not path.is_absolute():
            # Resolve relative to project root
            return str((ROOT_DIR / path).resolve())
        return str(path.resolve())

    model_config = {
        "env_file": [
            str(ROOT_DIR / ".env"),
            str(ROOT_DIR / ".env.prod"),
            str(Path(__file__).resolve().parent / ".env.prod"),  # Docker: /app/backend/.env.prod
        ],
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
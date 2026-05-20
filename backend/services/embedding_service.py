import hashlib
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from typing import Optional, List, Dict, Any, AsyncGenerator

from config import settings
try:
    from utils.logging import get_logger
except ImportError:
    from utils.logging import get_logger

logger = get_logger(__name__)

_MAX_CACHE = 100000
_embed_cache: dict[str, list[float]] = {}

HF_EMBED_URL = "https://adithya3003-codeperfect-embeddings.hf.space/embed"


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_get(text: str) -> list[float] | None:
    return _embed_cache.get(_cache_key(text))


def _cache_set(text: str, vector: list[float]) -> None:
    if len(_embed_cache) < _MAX_CACHE:
        _embed_cache[_cache_key(text)] = vector


class EmbeddingService:
    def __init__(self):
        self.use_local = settings.use_local_embeddings
        self.url = HF_EMBED_URL
        self.timeout = settings.embedding_timeout
        self.retry_limit = settings.embedding_retry_limit
        
        self.local_model = None
        if self.use_local:
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(settings.embedding_model)
                logger.info("EMBEDDING_MODEL_LOADED | model=%s | mode=LOCAL", settings.embedding_model)
            except Exception as e:
                logger.error("CRITICAL_COMPONENT_LOAD_FAILURE | component=EMBEDDING | error=%s — falling back to remote", e)
                self.use_local = False

        if not self.use_local:
            logger.info(
                "EMBEDDING_MODEL_LOADED | mode=REMOTE | url=%s | timeout=%ds | retries=%d",
                HF_EMBED_URL, self.timeout, self.retry_limit,
            )

    @retry(
        stop=stop_after_attempt(settings.embedding_retry_limit if hasattr(settings, "embedding_retry_limit") else 5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(httpx.RemoteProtocolError)
            | retry_if_exception_type(httpx.ReadTimeout)
            | retry_if_exception_type(httpx.ConnectTimeout)
            | retry_if_exception_type(httpx.ConnectError)
            | retry_if_exception_type(RuntimeError)
        ),
        before_sleep=lambda retry_state: logger.warning(
            "EmbeddingService: API call failed (Attempt %d). Retrying in %0.1fs... Error: %s",
            retry_state.attempt_number,
            retry_state.next_action.sleep,
            retry_state.outcome.exception(),
        ),
    )
    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Internal method to call the embedding API with retry logic."""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:
            response = await client.post(
                self.url,
                json={"texts": texts},
            )

            if response.status_code != 200:
                # If we get a rate limit or server error, raise for tenacity to catch
                raise RuntimeError(
                    f"Embedding API failed: {response.status_code} {response.text}"
                )

            data = response.json()
            if "embeddings" not in data:
                raise RuntimeError(f"Unexpected API response format: {data}")

            return data["embeddings"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = _cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            try:
                if self.use_local and self.local_model:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    vectors = await loop.run_in_executor(
                        None, lambda: self.local_model.encode(miss_texts).tolist()
                    )
                else:
                    vectors = await self._call_api(miss_texts)
                
                if len(vectors) != len(miss_texts):
                    raise RuntimeError(
                        f"API returned {len(vectors)} embeddings for {len(miss_texts)} texts"
                    )

                for idx, text, vec in zip(miss_indices, miss_texts, vectors):
                    _cache_set(text, vec)
                    results[idx] = vec

                logger.info(
                    "EmbeddingService: fetched %d embeddings (%d cache hits)",
                    len(miss_texts),
                    len(texts) - len(miss_texts),
                )
            except Exception as e:
                logger.error("EmbeddingService: Critical failure after all retries: %s", e)
                raise

        return results  # type: ignore

    async def embed_single(self, text: str) -> list[float]:
        cached = _cache_get(text)
        if cached is not None:
            return cached

        result = await self.embed_texts([text])
        return result[0]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Reranks documents relative to query using a Cross-Encoder.
        Evaluates query + document pairs together for superior semantic precision.
        """
        if not documents:
            return []
            
        try:
            from sentence_transformers import CrossEncoder
            
            # Lazy loading to save memory (especially in 512MB RAM environment)
            if not hasattr(self, "_reranker") or self._reranker is None:
                logger.info("EmbeddingService: Loading lightweight Cross-Encoder (MiniLM-L-6)...")
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            
            pairs = [[query, doc] for doc in documents]
            import asyncio
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, lambda: self._reranker.predict(pairs).tolist()
            )
            
            # Sigmoid normalization for raw logits
            import math
            return [1.0 / (1.0 + math.exp(-s)) for s in scores]
        except Exception as e:
            logger.error("EmbeddingService: Reranking failed: %s", e)
            return [0.5] * len(documents)

# ── Singleton Management ───────────────────────────────────────────────────
import threading
_service_lock = threading.Lock()
_embedding_service_instance: Optional['EmbeddingService'] = None

def get_embedding_service() -> 'EmbeddingService':
    global _embedding_service_instance
    if _embedding_service_instance is None:
        with _service_lock:
            if _embedding_service_instance is None:
                _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
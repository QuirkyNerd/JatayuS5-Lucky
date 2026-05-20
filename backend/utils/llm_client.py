"""
utils/llm_client.py - Resilient Multi-Model Groq Routing Architecture.

Provides a production-grade interface for Groq with three-tier failover:
1. Tier 1 (groq-best): llama-3.3-70b-versatile (Clinical Reasoning)
2. Tier 2 (groq-fast): llama-3.1-8b-instant (Extraction/Fast Fallback)
3. Tier 3 (deterministic-safe): No-LLM Path (Authoritative Recovery)
"""

import json
import time
import asyncio
import requests
from functools import partial
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Model Aliases
MODELS = {
    "best": settings.groq_model_primary,
    "fast": settings.groq_model_fast
}

def _call_groq_sync(prompt: str, model_tier: str = "best") -> str:
    """
    Synchronous implementation of Groq REST call with quota-aware failover.
    """
    if not settings.groq_api_key:
        logger.error("DETERMINISTIC_FALLBACK_ACTIVE: Groq API key missing.")
        raise RuntimeError("GROQ_API_KEY_MISSING")

    # Initial Model Selection
    current_model = MODELS.get(model_tier, settings.groq_model_fast)
    
    # Phase 5: Benchmark Safety Mode
    if settings.benchmark_mode and model_tier == "best":
        logger.info("MODEL_DOWNGRADED: Tier 1 (70B) disabled in benchmark mode. Routing to Tier 2 (8B).")
        current_model = settings.groq_model_fast
        model_tier = "fast"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json"
    }

    # Phase 3 & 4: failover loop
    tries = 0
    max_tries = 2 if settings.enable_groq_fallbacks else 1
    
    while tries < max_tries:
        logger.info("MODEL_SELECTED: Tier=%s, Model=%s (Attempt %d)", model_tier, current_model, tries + 1)
        
        payload = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": "You are a professional medical coding assistant. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096
        }

        t0 = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            duration = (time.time() - t0) * 1000
            
            if response.status_code == 200:
                logger.info("GROQ_REQUEST_SUCCESS: Model=%s, Duration=%.2fms", current_model, duration)
                return response.json()["choices"][0]["message"]["content"]
            
            # Quota/Rate Limit Handling (429)
            if response.status_code == 429:
                logger.warning("QUOTA_FAILOVER: Model=%s returned 429 (Rate Limit).", current_model)
                if model_tier == "best" and settings.enable_groq_fallbacks:
                    logger.info("MODEL_DOWNGRADED: Automatic failover from Tier 1 to Tier 2.")
                    current_model = settings.groq_model_fast
                    model_tier = "fast"
                    tries += 1
                    continue
                else:
                    raise RuntimeError(f"GROQ_QUOTA_EXCEEDED: {response.text}")
            
            # Other errors
            logger.error("GROQ_REQUEST_FAILED: Status=%d, Error=%s", response.status_code, response.text)
            if model_tier == "best" and settings.enable_groq_fallbacks:
                current_model = settings.groq_model_fast
                model_tier = "fast"
                tries += 1
                continue
            else:
                raise RuntimeError(f"GROQ_API_ERROR_{response.status_code}")

        except Exception as e:
            logger.error("GROQ_CONNECTION_ERROR: Model=%s, Error=%s", current_model, e)
            if model_tier == "best" and settings.enable_groq_fallbacks:
                logger.info("MODEL_DOWNGRADED: CONNECTION_FAILOVER to Tier 2.")
                current_model = settings.groq_model_fast
                model_tier = "fast"
                tries += 1
                continue
            else:
                raise e

    raise RuntimeError("GROQ_ALL_TIERS_FAILED")


async def generate_json_async(prompt: str, tier: str = "best") -> str:
    """
    Async wrapper for Multi-Tier Groq Routing.
    In BENCHMARK_MODE, this enforces the deterministic/fast path.
    """
    # Phase 2: Intelligent Task Routing (Optional override for benchmark mode)
    # If benchmark mode is extremely strict, we can force Tier 3 here.
    # But current request allows groq-fast or deterministic-safe.
    
    # Check if we should skip LLM entirely (Tier 3)
    # In some benchmarks, we might want ZERO network calls.
    # For now, we allow the request to proceed to _call_groq_sync 
    # where it will be downgraded to Tier 2.
    
    loop = asyncio.get_event_loop()
    fn = partial(_call_groq_sync, prompt, tier)
    
    try:
        return await loop.run_in_executor(None, fn)
    except Exception as e:
        logger.error("DETERMINISTIC_FALLBACK_ACTIVE: %s", e)
        raise e

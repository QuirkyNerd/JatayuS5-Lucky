"""
utils/logging.py – Structured JSON logger factory for CodePerfectAuditor.
"""

import logging
import json
import sys
import uuid
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── Context variables ─────────────────────────────────────────────
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_document_id_var: ContextVar[int] = ContextVar("document_id", default=-1)
_agent_name_var: ContextVar[str] = ContextVar("agent_name", default="")

# ── Request Context ───────────────────────────────────────────────
def set_request_context(
    *,
    request_id: str = "",
    document_id: Optional[int] = None,
    agent_name: str = "",
) -> None:
    if request_id:
        _request_id_var.set(request_id)
    if document_id is not None:
        _document_id_var.set(document_id)
    if agent_name:
        _agent_name_var.set(agent_name)


def new_request_id() -> str:
    return str(uuid.uuid4())


# ── JSON Formatter ────────────────────────────────────────────────
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Context injection
        rid = _request_id_var.get()
        did = _document_id_var.get()
        aname = _agent_name_var.get()

        if rid:
            payload["request_id"] = rid
        if did != -1:
            payload["document_id"] = did
        if aname:
            payload["agent_name"] = aname

        if record.exc_info and record.exc_info[0]:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload)


# ── Logger Factory ────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())

    logger.addHandler(handler)

    # Read level from ENV (default INFO)
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    logger.propagate = False
    return logger
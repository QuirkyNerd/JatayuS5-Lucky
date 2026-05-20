"""
security/auth.py – JWT Authentication + RBAC for CodePerfectAuditor.

Implements:
  - Password hashing via bcrypt (passlib)
  - JWT access tokens (15-min lifetime, HS256)
  - JWT refresh tokens (7-day, stored in httpOnly cookie)
  - FastAPI dependency injection: get_current_user, require_admin, etc.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

try:
    from database.db import get_db
    from database.models import User
    from utils.logging import get_logger
except ImportError:
    from database.db import get_db
    from database.models import User
    from utils.logging import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_SECRET_KEY     = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-use-32-char-secret")
_ALGORITHM      = "HS256"
_ACCESS_EXPIRE  = int(os.environ.get("JWT_ACCESS_EXPIRE_MINUTES", "15"))
_REFRESH_EXPIRE = int(os.environ.get("JWT_REFRESH_EXPIRE_DAYS",   "7"))

# ── Password hashing ──────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(user_id: int, role: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_ACCESS_EXPIRE)
    payload = {
        "sub":   str(user_id),
        "role":  role,
        "email": email,
        "exp":   expire,
        "type":  "access",
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_REFRESH_EXPIRE)
    payload = {
        "sub":  str(user_id),
        "exp":  expire,
        "type": "refresh",
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Decode access token and return the authenticated User ORM object."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type.")

    user_id = int(payload["sub"])
    result  = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user    = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated.")
    return user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Like get_current_user but returns None instead of raising for anonymous endpoints."""
    if not token:
        return None
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


def _require_role(*roles: str):
    async def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {' or '.join(roles)}.",
            )
        return current_user
    return dependency


require_admin    = _require_role("ADMIN")
require_coder    = _require_role("ADMIN", "CODER")
require_reviewer = _require_role("ADMIN", "REVIEWER")

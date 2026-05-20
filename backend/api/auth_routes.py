import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, Cookie, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database.db import get_db
from database.models import User, Organization, Branch
from security.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user, require_admin,
)
from utils.logging import get_logger
from utils.governance import log_governance
from config import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name:      str
    email:     EmailStr
    password:  str
    org_id:    int | None = None
    branch_id: int | None = None


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class UserOut(BaseModel):
    id:         int
    name:       str
    email:      str
    role:       str
    org_id:     int | None
    branch_id:  int | None
    created_at: datetime

    class Config:
        from_attributes = True


class AdminCreateUserRequest(BaseModel):
    name:      str
    email:     EmailStr
    password:  str
    role:      str = "CODER"   # ADMIN | CODER | REVIEWER
    org_id:    int | None = None
    branch_id: int | None = None


class OrgRequest(BaseModel):
    name: str


class BranchRequest(BaseModel):
    name:   str
    org_id: int


class RoleUpdateRequest(BaseModel):
    role: str


class ResetPasswordRequest(BaseModel):
    new_password: str


class DemoLoginRequest(BaseModel):
    role: str   # coder | reviewer | admin


# ── Signup ────────────────────────────────────────────────────────────────────

@router.post("/signup", status_code=201)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    """Public signup — always creates CODER role."""
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="CODER",
        org_id=payload.org_id,
        branch_id=payload.branch_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("New user registered: %s (role=CODER)", user.email)
    return {"message": "Account created. Role: CODER", "user_id": user.id, "role": "CODER"}


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(payload: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Authenticate user and issue access_token + refresh_token cookie."""
    print("LOGIN ATTEMPT:", payload.email)
    result = await db.execute(select(User).where(User.email == payload.email, User.is_active == True))
    user   = result.scalar_one_or_none()

    if user:
        print("LOGIN INPUT:", payload.password)
        print("DB HASH:", user.password_hash)

    if not user or not verify_password(payload.password, user.password_hash):
        if user:
            print("LOGIN VERIFICATION FAILED. Passwords do not match hash.")
        else:
            print("LOGIN FAILED: User not found.")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    access_token  = create_access_token(user.id, user.role, user.email)
    refresh_token = create_refresh_token(user.id)

    # Set refresh token as httpOnly secure cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=7 * 24 * 3600,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
    )

    return {
        "access_token": access_token,
        "token_type":   "bearer",
        "user": {
            "id":    user.id,
            "name":  user.name,
            "email": user.email,
            "role":  user.role,
        },
    }


@router.post("/demo-login")
async def demo_login(payload: DemoLoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """
    SECTION 2, 3, 5, 6: GUARANTEED DEMO LOGIN
    Must never return 403. Must always return a valid token + role.
    """
    print(f"DEBUG: Demo login requested for role: {payload.role}")
    logger.info("Demo login requested for role: %s", payload.role)
    
    try:
        role = payload.role.upper()
        if role not in ("CODER", "REVIEWER", "ADMIN"):
            role = "CODER"

        email = f"{role.lower()}_demo@codeperfect.demo"
        password = "demo-password-2026"

        # Find or Create Demo User
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            # Get or create demo org
            res_org = await db.execute(select(Organization).where(Organization.name == "CodePerfect Hospital"))
            org = res_org.scalar_one_or_none()
            if not org:
                org = Organization(name="CodePerfect Hospital")
                db.add(org)
                await db.commit()
                await db.refresh(org)

            user = User(
                name=f"Demo {role.capitalize()}",
                email=email,
                password_hash=hash_password(password),
                role=role,
                org_id=org.id,
                is_active=True,
                is_demo=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            user.is_demo = True
            user.is_active = True
            await db.commit()

        access_token  = create_access_token(user.id, user.role, user.email)
        refresh_token = create_refresh_token(user.id)

        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            max_age=7 * 24 * 3600,
            samesite="lax",
            secure=False,
        )

        return {
            "access_token": access_token,
            "token_type":   "bearer",
            "role":         user.role,
            "user": {
                "id":    user.id,
                "name":  user.name,
                "email": user.email,
                "role":  user.role,
            },
            "demo_session": True
        }
    except Exception as e:
        logger.error(f"CRITICAL DEMO FAILURE: {str(e)}")
        # If DB fails completely, we try a desperate fallback with a hardcoded mock user ID if possible
        # but in a real system, we might need a working DB. 
        # We'll return a 500 with the error to help the user debug.
        raise HTTPException(
            status_code=500, 
            detail=f"Demo Login Error: {str(e)}. Ensure database is running and seeded."
        )


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post("/refresh")
async def refresh_token(
    refresh_token: str | None = Cookie(default=None),
    db: AsyncSession = Depends(get_db),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token.")

    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    user_id = int(payload["sub"])
    result  = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user    = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    new_access = create_access_token(user.id, user.role, user.email)
    return {"access_token": new_access, "token_type": "bearer"}


# ── Profile ───────────────────────────────────────────────────────────────────

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id":        current_user.id,
        "name":      current_user.name,
        "email":     current_user.email,
        "role":      current_user.role,
        "is_demo":   current_user.is_demo,
        "org_id":    current_user.org_id,
        "branch_id": current_user.branch_id,
    }


# ── Admin: User Management ────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Admin panel: List users with strict demo isolation."""
    if current_user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Access denied.")

    # ✅ SECTION 4: USER MANAGEMENT ISOLATION
    if current_user.is_demo:
        print(f"USER FETCH: is_demo={current_user.is_demo} → returning demo accounts only")
        stmt = select(User).where(User.is_demo == True).order_by(User.created_at.desc())
    else:
        print(f"USER FETCH: is_demo={current_user.is_demo} → returning production accounts only")
        stmt = select(User).where(User.is_demo == False).order_by(User.created_at.desc())

    result = await db.execute(stmt)
    users  = result.scalars().all()
    print(f"USER FETCH: returning {len(users)} users")
    return [
        {
            "id": u.id, "name": u.name, "email": u.email,
            "role": u.role, "org_id": u.org_id, "branch_id": u.branch_id,
            "is_active": u.is_active, "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.post("/users", status_code=201)
async def create_user(
    payload: AdminCreateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin-only: create a new user account."""
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")
    user = User(
        name=payload.name, email=payload.email,
        password_hash=hash_password(payload.password),
        role=payload.role, org_id=payload.org_id, branch_id=payload.branch_id,
        is_demo=current_user.is_demo,  # Inherit demo status from admin
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await log_governance(
        db, "user_create", current_user.id, current_user.role,
        metadata=f"Admin created user {user.email} with role {user.role}"
    )
    return {"message": "User created.", "user_id": user.id, "role": user.role}


@router.patch("/users/{user_id}/role")
async def update_role(
    user_id: int,
    payload: RoleUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user   = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Environment Isolation Check
    if user.is_demo != current_user.is_demo:
        raise HTTPException(status_code=403, detail="Forbidden: User environment mismatch.")
    if payload.role not in ("ADMIN", "CODER", "REVIEWER"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be ADMIN, CODER, or REVIEWER.")
    old_role = user.role
    user.role = payload.role
    await log_governance(
        db, "user_update", current_user.id, current_user.role,
        metadata=f"Changed role of {user.email} from {old_role} to {payload.role}"
    )
    await db.commit()
    logger.info(
        "AUDIT: Admin %s (id=%d) changed role of user %s (id=%d) from %s to %s.",
        current_user.email, current_user.id, user.email, user.id, old_role, payload.role,
    )
    return {"message": f"Role updated from {old_role} to {payload.role}."}


@router.patch("/users/{user_id}/reset-password")
async def reset_password(
    user_id: int,
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if not payload.new_password or len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    # Environment Isolation Check
    if target.is_demo != current_user.is_demo:
        raise HTTPException(status_code=403, detail="Forbidden: User environment mismatch.")

    target.password_hash = hash_password(payload.new_password)
    await log_governance(
        db, "user_update", current_user.id, current_user.role,
        metadata=f"Reset password for {target.email}"
    )
    await db.commit()
    await db.refresh(target)

    print("UPDATED HASH:", target.password_hash)

    logger.info(
        "AUDIT: Admin %s (id=%d) reset password for user %s (id=%d).",
        current_user.email, current_user.id, target.email, target.id,
    )
    return {"message": f"Password for '{target.name}' has been reset successfully."}


@router.patch("/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Toggle a user's is_active status. ADMIN-only.
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot deactivate your own account.",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Environment Isolation Check
    if user.is_demo != current_user.is_demo:
        raise HTTPException(status_code=403, detail="Forbidden: User environment mismatch.")

    user.is_active = not user.is_active
    action = "activated" if user.is_active else "deactivated"
    await log_governance(
        db, "user_update", current_user.id, current_user.role,
        metadata=f"{action.capitalize()} user {user.email}"
    )
    await db.commit()
    
    logger.info("AUDIT: Admin %s %s user %s.", current_user.email, action, user.email)
    return {"message": f"User {action}.", "is_active": user.is_active}


@router.delete("/users/{user_id}", status_code=200)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin-only: permanently delete a user account."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot delete your own account.",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Environment Isolation Check
    if user.is_demo != current_user.is_demo:
        raise HTTPException(status_code=403, detail="Forbidden: User environment mismatch.")

    await log_governance(
        db, "user_delete", current_user.id, current_user.role,
        metadata=f"Admin permanently deleted user {user.email} (role={user.role})"
    )
    await db.delete(user)
    await db.commit()

    logger.info("AUDIT: Admin %s deleted user %s (id=%d).", current_user.email, user.email, user_id)
    return {"message": "User deleted successfully."}


# ── Admin: Organisation & Branch Management ───────────────────────────────────

@router.post("/org", status_code=201, dependencies=[Depends(require_admin)])
async def create_org(payload: OrgRequest, db: AsyncSession = Depends(get_db)):
    org = Organization(name=payload.name)
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return {"org_id": org.id, "name": org.name}


@router.get("/org", dependencies=[Depends(require_admin)])
async def list_orgs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Organization).order_by(Organization.created_at.desc()))
    return [{"id": o.id, "name": o.name, "created_at": o.created_at.isoformat()}
            for o in result.scalars().all()]


@router.post("/branches", status_code=201, dependencies=[Depends(require_admin)])
async def create_branch(payload: BranchRequest, db: AsyncSession = Depends(get_db)):
    branch = Branch(name=payload.name, org_id=payload.org_id)
    db.add(branch)
    await db.commit()
    await db.refresh(branch)
    return {"branch_id": branch.id, "name": branch.name}


@router.get("/branches")
async def list_branches(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    query = select(Branch)
    if current_user.role != "ADMIN" and current_user.org_id:
        query = query.where(Branch.org_id == current_user.org_id)
    result = await db.execute(query.order_by(Branch.created_at.desc()))
    return [{"id": b.id, "name": b.name, "org_id": b.org_id} for b in result.scalars().all()]

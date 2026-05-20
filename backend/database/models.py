"""
database/models.py – Full ORM models for CodePerfectAuditor Platform.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Index, Integer, Float, Text, String,
    DateTime, ForeignKey, Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Multi-Tenant ──────────────────────────────────────────────────────────────

class Organization(Base):
    __tablename__ = "organizations"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(200), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    branches = relationship("Branch", back_populates="organization")
    users    = relationship("User",   back_populates="organization")
    cases    = relationship("Case",   back_populates="organization")


class Branch(Base):
    __tablename__ = "branches"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    name    = Column(String(200), nullable=False)
    org_id  = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    organization = relationship("Organization", back_populates="branches")
    users        = relationship("User", back_populates="branch")


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(200), nullable=False)
    email         = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)

    # ✅ ROLE SYSTEM (ADMIN / CODER / REVIEWER)
    role          = Column(String(20), nullable=False, default="CODER")

    org_id        = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    branch_id     = Column(Integer, ForeignKey("branches.id"), nullable=True)
    is_active     = Column(Boolean, default=True, nullable=False)
    is_demo       = Column(Boolean, default=False, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    organization = relationship("Organization", back_populates="users")
    branch       = relationship("Branch", back_populates="users")

    # ✅ CREATED CASES (CODER)
    created_cases = relationship(
        "Case",
        back_populates="creator",
        foreign_keys="Case.user_id"
    )


# ── Case Management ───────────────────────────────────────────────────────────

class Case(Base):
    __tablename__ = "cases"

    id               = Column(Integer, primary_key=True, autoincrement=True)

    # ✅ CODER
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=True)

    org_id           = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    # Input
    input_text       = Column(Text, nullable=False)
    note_hash        = Column(String(64), nullable=True, index=True)
    uploaded_file    = Column(String(500), nullable=True)

    # AI output
    ai_codes         = Column(Text, default="[]")
    human_codes      = Column(Text, default="[]")
    discrepancies    = Column(Text, default="[]")
    evidence         = Column(Text, default="[]")
    pipeline_log     = Column(Text, default="[]")
    retrieved_docs   = Column(Text, default="[]")

    # Metrics
    risk_score       = Column(Float, default=0.0)
    revenue_impact   = Column(Float, default=0.0)
    coding_accuracy  = Column(Float, default=0.0)
    avg_confidence   = Column(Float, default=0.0)
    processing_time  = Column(Float, default=0.0)

    # Metadata
    model_used        = Column(String(100), default="llama-3.3-70b-versatile")
    embedding_version = Column(String(50), default="all-MiniLM-L6-v2")
    summary           = Column(Text, default="")
    status            = Column(String(20), default="draft") # draft | submitted | under_review | approved | rejected
    tokens_used       = Column(Integer, default=0)
    cost_estimate     = Column(String(50), default="$0.000")

    # ✅ REVIEWER FIELDS
    final_code_set    = Column(Text, default="[]")
    reviewer_notes    = Column(Text, nullable=True)
    reviewed_by       = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at       = Column(DateTime, nullable=True)
    
    # ✅ ASSIGNMENT & ACCOUNTABILITY
    assigned_to       = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_at       = Column(DateTime, nullable=True)
    assignment_status = Column(String(50), default="unassigned") # unassigned, assigned, reassigned
    review_confidence = Column(Float, nullable=True) # 0.0 - 1.0
    review_completeness = Column(String(50), default="complete")
    review_duration   = Column(Float, nullable=True) # seconds
    
    # ✅ LOCKING MECHANISM
    locked_by         = Column(Integer, ForeignKey("users.id"), nullable=True)
    locked_at         = Column(DateTime, nullable=True)

    # ✅ PRIORITY QUEUE
    priority          = Column(String(20), default="normal") # low, normal, high

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_demo           = Column(Boolean, default=False, nullable=False)

    # ✅ FIXED RELATIONSHIPS
    creator = relationship(
        "User",
        back_populates="created_cases",
        foreign_keys=[user_id]
    )

    organization = relationship("Organization", back_populates="cases")
    
    reviewer = relationship("User", foreign_keys=[assigned_to])
    locker   = relationship("User", foreign_keys=[locked_by])

    __table_args__ = (
        Index("ix_cases_org_created", "org_id", "created_at"),
        Index("ix_cases_user_created", "user_id", "created_at"),
        Index("ix_cases_risk", "risk_score"),
    )


# ── Governance & Audit ────────────────────────────────────────────────────────

class GovernanceLog(Base):
    __tablename__ = "governance_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    case_id        = Column(Integer, ForeignKey("cases.id"), nullable=True)
    actor_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    actor_role     = Column(String(20), nullable=False) # coder, reviewer, admin
    action_type    = Column(String(50), nullable=False)
    timestamp      = Column(DateTime, default=datetime.utcnow, nullable=False)
    previous_state = Column(Text, nullable=True) # JSON
    new_state      = Column(Text, nullable=True) # JSON
    metadata_json  = Column(Text, nullable=True) # JSON (renamed from metadata to avoid SQLAlchemy conflicts if any)

    # Relationships
    case = relationship("Case")
    user = relationship("User")


# ── Legacy tables ───────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    note_text   = Column(Text, nullable=False)
    note_hash   = Column(String(64), nullable=True, index=True)
    human_codes = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    audit_results = relationship("AuditResult", back_populates="document")


class AuditResult(Base):
    __tablename__ = "audit_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    document_id   = Column(Integer, ForeignKey("documents.id"), nullable=False)
    ai_codes      = Column(Text, nullable=False)
    discrepancies = Column(Text, nullable=False)
    evidence      = Column(Text, nullable=False)
    summary       = Column(Text, default="")
    tokens_used   = Column(Integer, default=0)
    cost_estimate = Column(String(50), default="$0.000")
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    document   = relationship("Document", back_populates="audit_results")
    agent_logs = relationship("AgentLog", back_populates="audit_result")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    audit_id     = Column(Integer, ForeignKey("audit_results.id"), nullable=False)
    pipeline_log = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)

    audit_result = relationship("AuditResult", back_populates="agent_logs")


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    note_hash  = Column(String(64), nullable=False, index=True)
    ai_code    = Column(String(50), nullable=False)
    decision   = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
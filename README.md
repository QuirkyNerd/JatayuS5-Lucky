# CodePerfectAuditor

**AI-powered Clinical Coding Audit & Revenue Integrity Platform for ICD-10 and CPT validation.**


## Deployment

**Frontend**  
http://161.118.217.29:3000/login

**API Documentation**  
http://161.118.217.29:8000/docs

---

## Overview

**CodePerfectAuditor** is an agentic AI platform that audits medical coding in real time. It reads clinical documentation—operative notes, admission summaries, and discharge records—and validates ICD-10 and CPT assignments **before** claims are submitted.

The system is built for **Revenue Integrity**: helping hospitals capture appropriate reimbursement while maintaining coding compliance. Rather than a single black-box model, the platform combines deterministic clinical extraction, retrieval-augmented coding knowledge, multi-stage validation, and human-in-the-loop review workflows.

---

## Key Features

| Capability | Description |
|------------|-------------|
| **Real-time coding audit** | Streamed pipeline with step-by-step progress for coders |
| **ICD-10 & CPT validation** | Diagnosis and procedure code sets with evidence grounding |
| **Human vs AI comparison** | Auditor Agent surfaces missed, unsupported, and mismatched codes |
| **Evidence highlighting** | Supporting spans linked to each recommended code |
| **Case workflow** | Draft → submit → review → approve/reject with audit trail |
| **Role-based access** | Separate experiences for coders, reviewers, and administrators |
| **Hybrid RAG retrieval** | Dense + sparse search over a large medical knowledge index |
| **Governance validators** | Terminal validation layer before codes reach the client |
| **Docker-ready deployment** | Containerized frontend and backend for consistent environments |

---

## 2. Folder Architecture

JatayuS5-Lucky/
├── backend/                 # FastAPI application root (WORKDIR in Docker)
│   ├── agents/              # Coding, auditor, evidence agents
│   ├── api/                 # HTTP route modules
│   ├── constants/           # Case status enum, normalization
│   ├── data/                # ICD/CPT CSVs, benchmarks, checkpoints
│   ├── database/            # SQLAlchemy models + async session
│   ├── prompts/             # LLM prompt templates
│   ├── scratch/             # Development / forensic scratch
│   ├── scripts/             # Backend-local eval & smoke scripts
│   ├── security/            # JWT auth dependencies
│   ├── services/            # Core pipeline services (RAG, validators, eval)
│   └── utils/               # PHI, logging, LLM client, normalizers
├── frontend/                # React + Vite SPA
│   └── src/
│       ├── components/      # Upload, audit results, sidebar
│       ├── pages/           # Dashboard, cases, analytics, evaluation
│       ├── services/        # axios API client
│       ├── data/            # sampleNotes.js (Load Sample)
│       └── styles/          # CSS (no Tailwind in package.json)
├── scripts/                 # Root ingestion (ingest_guidelines.py)
├── tests/                   # pytest suite
├── docker-compose.yml
├── Dockerfile               # Backend image
├── requirements.txt         # Python deps (repo root)
├── README.md

---

## Role-Based Access Control

| Role | Responsibilities |
|------|----------------|
| **Coder** | Upload or paste clinical notes, enter human code sets, run audits, review AI output, submit cases for review |
| **Reviewer** | Access case queue, inspect discrepancies and evidence, approve or reject submissions, update final code sets |
| **Admin** | Manage users and organizations, assign reviewers, monitor analytics, run system evaluation jobs |

---

## Target Users

| User Group | How They Benefit |
|------------|------------------|
| **Medical Coders** | Pre-submission validation and discrepancy detection |
| **CDI Teams** | Documentation alignment with billed codes |
| **Revenue Integrity Teams** | Reduced leakage and overcoding risk |
| **Hospital Auditors** | Traceable defense trail per chart |
| **Compliance Teams** | Evidence-backed coding decisions |
| **Healthcare Administrators** | Workflow oversight and operational analytics |

---

## Technology Stack

| Layer | Technologies |
|-------|----------------|
| **Frontend** | React 18, Vite, React Router, Axios, Recharts |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy (async), Pydantic |
| **Database** | PostgreSQL |
| **Vector store** | Qdrant (production) with ChromaDB fallback |
| **Cache** | Redis (optional audit result caching) |
| **ML / NLP** | sentence-transformers, cross-encoder reranking, SapBERT |
| **LLM** | Groq API (structured JSON coding assistance) |
| **Auth** | JWT (access + refresh), bcrypt password hashing |
| **Deployment** | Docker, docker-compose |

---



## Frontend Architecture

The client is a **React + Vite** single-page application with role-gated routing.

| Area | Implementation |
|------|----------------|
| **Coder workspace** | Note upload, human code entry, streaming audit progress, results tabs |
| **Case history** | Searchable case list, status management, reviewer actions |
| **Audit results** | Summary, code comparison, explainability, removed codes, evidence viewer |
| **Analytics** | Revenue and trend views for reviewers and admins |
| **Authentication** | Login, demo sessions, token refresh, session persistence |

**Primary routes:** `/` (coder dashboard), `/case-history`, `/analytics`, `/users`, `/evaluation` (admin).

State is managed through React Context (`AuthContext`, `AuditContext`) with optional `sessionStorage` recovery for in-progress audits.

---

## Backend Architecture

The API layer is **FastAPI** with modular routers and async database access.

| Component | Role |
|-----------|------|
| **`main.py`** | Application entry, CORS, lifespan warmup (DB + RAG engine) |
| **`api/routes.py`** | Streaming audit and feedback endpoints |
| **`api/case_routes.py`** | Case lifecycle and reviewer workflows |
| **`api/auth_routes.py`** | Authentication and user administration |
| **`services/audit_pipeline.py`** | End-to-end orchestration |
| **`agents/`** | Coding logic, auditor, and evidence agents |

On startup, the backend initializes the database and preloads retrieval models so the first audit request does not pay a cold-start penalty.

---


## Retrieval-Augmented Generation (RAG)

| Stage | Technology | Purpose |
|-------|------------|---------|
| **Dense retrieval** | Embedding model (`BAAI/bge-small-en-v1.5`) | Semantic similarity over ICD/CPT/guideline corpora |
| **Sparse retrieval** | BM25 (`FastBM25`) | Lexical matching for exact clinical terms |
| **Hybrid fusion** | Weighted score blend | Combines dense and sparse signals per candidate |
| **Cross-encoder rerank** | `ms-marco-MiniLM-L-6-v2` | Reorders top candidates by query–document relevance |
| **SapBERT validation** | PubMedBERT-derived SapBERT | Biomedical semantic verification of top matches |
| **Anatomy routing** | Region hierarchy maps | Aligns fracture, cardiac, GI, and other domains |

Queries run **per extracted clinical entity**, improving precision versus whole-note embedding search.

---

## Medical Knowledge Ingestion

Knowledge is loaded from structured datasets and guideline corpora into vector collections via `scripts/ingest_guidelines.py`, with safeguards against accidental overwrite of populated indexes.

| Knowledge Collection | Records Ingested |
|---|---:|
| ICD-10 Clinical Codes | 111,738 |
| CPT Procedure Codes | 8,958 |
| Clinical Guidelines | 729 |
| Symptom & Clinical Evidence Mappings | 116 |
| **Total Retrieval Knowledge Entries** | **121,541+** |


## Complete Request Flow

```mermaid
sequenceDiagram
    participant U as Coder (Browser)
    participant F as React Frontend
    participant API as FastAPI /audit
    participant P as AuditPipeline
    participant R as RAG Engine
    participant V as Final Validator
    participant A as Auditor Agent
    participant DB as PostgreSQL

    U->>F: Submit note + human codes
    F->>API: POST /api/v1/audit (SSE)
    API->>P: run_stream()
    P->>P: Entity extraction
    P->>R: Hybrid retrieval per entity
    R-->>P: Ranked candidates
    P->>P: Coding logic + selection
    P->>V: Terminal validation
    V-->>P: Validated ICD/CPT set
    P->>A: Compare vs human codes
    A-->>P: Discrepancies + evidence
    P-->>API: Complete payload
    API->>DB: Persist case (optional)
    API-->>F: Stream step events + result
    F-->>U: Audit results UI
```

---

## API Architecture

All routes are prefixed with **`/api/v1`**.

| Module | Prefix | Purpose |
|--------|--------|---------|
| **Auth** | `/auth` | Login, refresh, user and org management |
| **Audit** | `/audit` | Streaming clinical audit, file upload, feedback |
| **Cases** | `/cases` | Case CRUD, workflow, reviewer actions |
| **Analytics** | `/analytics` | Overview and trend metrics |
| **Evaluation** | `/evaluation` | Admin system evaluation jobs |
| **Health** | `/health` | Liveness and readiness probes |

Interactive documentation: **http://161.118.217.29:8000/docs**

---

## Appendix A — Key File Reference

| File | Responsibility |
|------|----------------|
| `backend/services/audit_pipeline.py` | End-to-end audit orchestration (`run`, `run_stream`) |
| `backend/services/rag_engine.py` | Hybrid retrieval, reranking, Qdrant/Chroma integration |
| `backend/services/selection_engine.py` | Candidate competition and clinical scoring |
| `backend/services/final_validator.py` | Terminal governance and evidence gates |
| `backend/agents/coding_logic.py` | RAG-first coding agent |
| `backend/agents/auditor.py` | Human vs AI discrepancy engine |
| `backend/services/evaluation_engine.py` | Benchmark evaluation runner |
| `backend/api/admin_routes.py` | Admin evaluation API |
| `backend/api/case_routes.py` | Case workflow endpoints |
| `frontend/src/services/api.js` | HTTP client, auth interceptors, SSE audit |

---

## Appendix B — API Route Summary

| Method | Endpoint | Access |
|--------|----------|--------|
| `POST` | `/api/v1/auth/login` | Public |
| `POST` | `/api/v1/auth/refresh` | Authenticated |
| `GET` | `/api/v1/auth/me` | Authenticated |
| `POST` | `/api/v1/audit` | Coder / Admin |
| `POST` | `/api/v1/audit/file` | Coder / Admin |
| `POST` | `/api/v1/feedback` | Authenticated |
| `GET` | `/api/v1/cases` | Authenticated |
| `PATCH` | `/api/v1/cases/{id}/status` | Role-dependent |
| `POST` | `/api/v1/cases/{id}/submit` | Coder |
| `POST` | `/api/v1/cases/{id}/approve` | Reviewer |
| `POST` | `/api/v1/cases/{id}/reject` | Reviewer |
| `GET` | `/api/v1/analytics/overview` | Reviewer / Admin |
| `GET` | `/api/v1/evaluation` | Admin |
| `GET` | `/api/v1/health/live` | Public |

---


---

## Team Project

**Virtusa Jatayu S5 — CodePerfectAuditor - Team(Lucky)**

| | |
|---|---|
| **Program** | Virtusa Jatayu Innovation Challenge |
| **Domain** | Healthcare AI · Revenue Integrity · Clinical Coding |
| **Repository** | JatayuS5-Lucky |

*Built as a team capstone demonstrating  clinical AI engineering—from retrieval infrastructure to governed coding workflows.*

---

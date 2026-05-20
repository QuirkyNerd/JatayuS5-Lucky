# CodePerfectAuditor

**CodePerfectAuditor** is an enterprise-grade, AI-powered clinical coding audit system. It acts as an intelligent co-pilot for Clinical Documentation Improvement (CDI) specialists and medical coders by validating ICD-10 and CPT codes against clinical notes before claim submission. 

By leveraging a highly defensive **5-stage AI pipeline** (deterministic extraction + RAG + LLM reasoning), the system strictly prevents medical hallucinations, identifying missed codes (revenue leakage) and unsupported codes (compliance risk) with high clinical accuracy.

---

## 🏗️ Architecture & Pipeline (v5)

CodePerfectAuditor rejects the "black-box LLM" approach. Instead, it utilizes a heavily constrained, multi-agent orchestrator:

1. **Deterministic Entity Extraction**: A FAANG-grade regex parser maps clinical terms and handles negation/context locally before any AI is invoked.
2. **Entity-Level RAG**: Queries ChromaDB independently *for each extracted condition* (rather than dumping the whole note), ensuring highly relevant ICD/CPT retrieval.
3. **Terminal Evidence Gate**: The `ClinicalReasoningEngine` explicitly drops codes lacking textual grounding and aggressively filters out diagnoses derived from prophylactic contexts (e.g., DVT prophylaxis).
4. **Deterministic Rule Engine**: Encodes static billing guidelines (e.g., ICD-10 hierarchy upgrades for Diabetes + CKD, and symptom exclusion).
5. **Auditor Agent**: Uses Google Gemini (via strict JSON-enforced REST calls) to compare the validated AI code set against human-submitted codes to generate actionable discrepancies.

---

## 🌐 Live Deployment

**Frontend:** [https://codeperfect-audit.vercel.app](https://codeperfect-audit.vercel.app)  
**Backend API:** [https://codeperfect-audit.onrender.com](https://codeperfect-audit.onrender.com)  

*(Note: Production endpoints may require authorized tenant credentials)*

---

## 🛠️ Tech Stack

**Backend:**
- **Framework**: FastAPI (Async Python 3.11)
- **Database**: PostgreSQL (NeonDB) + SQLAlchemy (Async)
- **Vector Store**: ChromaDB (Persistent)
- **Cache / Queue**: Redis (Hiredis)
- **Security**: JWT Auth, Fernet PHI Encryption, Role-Based Access Control (RBAC)

**Frontend:**
- **Framework**: React.js + Vite + TailwindCSS
- **State Management**: React Context API
- **HTTP Client**: Axios with automatic 401 interception

**AI / ML:**
- **LLM**: Google Gemini API (Strict Temperature 0.0 JSON responses)
- **Embeddings**: `all-MiniLM-L6-v2` via Hugging Face `sentence-transformers`

---

## 🚀 Quick Start (Local)

### 1. Prerequisites
Ensure you have Python 3.11+, Node.js 20+, and PostgreSQL running locally or via Docker.

### 2. Clone & Configure
```bash
git clone <repository_url>
cd CodePerfectAuditor
copy .env.example backend\.env
```
*(Ensure `GEMINI_API_KEY` and `DATABASE_URL` are set in `.env`)*

### 3. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Ingest Reference Data into ChromaDB
Run this **once** before starting the server to populate the RAG vector store:
```bash
python ../scripts/ingest_guidelines.py
```

### 5. Start the Backend Server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Start the Frontend
```bash
cd ../frontend
npm install
npm run dev
```

---

## 🐳 Docker Deployment

The system is fully containerized for immediate deployment:
```bash
# In the project root
copy .env.example .env
docker-compose up --build
```
This spins up the FastAPI backend, the React frontend, and a local PostgreSQL instance.

---

## 🧪 Testing

The backend includes a comprehensive `pytest` suite for agent and pipeline logic:
```bash
pytest tests/ -v
```

---

## ⚙️ Core Configuration Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | **required** | Your Gemini API key for LLM orchestration. |
| `DATABASE_URL` | **required** | Async Postgres connection string (e.g., Neon). |
| `PHI_ENCRYPTION_KEY` | **required** | Fernet key for encrypting patient notes at rest. |
| `SECRET_KEY` | **required** | Cryptographic key for signing JWTs. |
| `RAG_TOP_K` | `10` | Number of vector results retrieved per entity. |

---

## 🛡️ Enterprise Readiness & Security

CodePerfectAuditor is designed for healthcare compliance:
- **Data Isolation**: Multi-tenant architecture (Organizations/Branches) with strict data boundaries.
- **Demo Sandboxing**: Hard-partitioned database queries (`is_demo=True`) ensure public demo users never interact with production patient data.
- **PHI Masking & Encryption**: Sensitive identifiers are masked before reaching the LLM and symmetrically encrypted (AES-128) before resting in PostgreSQL.
- **Concurrency Locking**: Audit cases lock for 10 minutes when opened by Reviewers to prevent conflicting manual edits.
- **Financial Analytics**: Live calculation of USD/INR revenue impact using CMS Medicare 2024 schedules.

---

## 📄 Complete Project Documentation
For a deep, exhaustive reverse-engineering report covering the end-to-end data flow, API architecture, database schemas, and prompt engineering strategy, please refer to the `COMPLETE_PROJECT_ANALYSIS.md` located in the root directory.

FROM python:3.11-slim

LABEL maintainer="CodePerfectAuditor"

# ---------- Environment ----------
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    CHROMA_PERSIST_DIR=/app/backend/backend/chroma_db

# ---------- Working Directory ----------
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python Dependencies ----------
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# ---------- Pre-download embedding model ----------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---------- Copy Application ----------
COPY backend/ ./backend/
COPY scripts/ ./scripts/
# Copy prod env file so settings load correctly in container
COPY .env.prod ./backend/.env.prod

# ---------- Runtime Directories ----------
RUN mkdir -p \
    /app/backend/backend/chroma_db \
    /app/backend/data/checkpoints \
    /app/persistent/reports \
    /app/logs \
    /app/.cache/huggingface

# ---------- Expose ----------
EXPOSE 8000

# ---------- Healthcheck ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health/live || exit 1

# ---------- Final Working Directory ----------
WORKDIR /app/backend

# ---------- Start Server ----------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
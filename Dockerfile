# ==========================================
# Multi-Agent Research Platform — Dockerfile
# ==========================================
#
# Single-stage production build.
# ChromaDB runs embedded inside the API process.
#
# Build:
#   docker build -t multi-agent-research .
#
# Run (with docker-compose preferred):
#   docker compose up --build
#
# ==========================================

FROM python:3.14-slim

# ==========================================
# System Dependencies
# ==========================================
# Required for:
#   - chromadb (onnxruntime native bindings)
#   - sentence-transformers / torch
#   - healthcheck (curl)

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Working Directory
# ==========================================

WORKDIR /app

# ==========================================
# Python Dependencies
# ==========================================
# Install requirements before copying source
# to leverage Docker layer caching.

COPY requirements-prod.txt ./
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# ==========================================
# Application Code
# ==========================================

COPY app/ ./app/

# ==========================================
# Non-Root User (Security)
# ==========================================

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ==========================================
# Ports
# ==========================================

EXPOSE 8000

# ==========================================
# Healthcheck
# ==========================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ==========================================
# Entrypoint
# ==========================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

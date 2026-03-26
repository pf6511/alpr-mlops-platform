# ═══════════════════════════════════════════════════════════════════════════════
# ALPR Engine - Dockerfile (LEAD Edition)
# Structure: src/, configs/, pipelines/, airflow/
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Base CPU (HuggingFace Spaces compatible)
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base-cpu

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ───────────────────────────────────────────────────────────────────────────────
# Base GPU (CUDA 11.8)
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# ───────────────────────────────────────────────────────────────────────────────
# Builder stage
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cpu AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ───────────────────────────────────────────────────────────────────────────────
# Production CPU (default - HuggingFace Spaces)
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cpu AS production

WORKDIR /app

# Copier packages Python
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copier l'application (nouvelle structure)
COPY src/ ./src/
COPY configs/ ./configs/
COPY pipelines/ ./pipelines/
COPY models/ ./models/
COPY migrations/ ./migrations/
COPY app.py .
COPY requirements.txt .

# Dossiers de travail
RUN mkdir -p /app/outputs /app/data /app/mlruns

# Variables d'environnement
ENV ENV=production
ENV DB_MODE=sqlite
ENV STORAGE_MODE=local
ENV RAY_MODE=local
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app

# Utilisateur non-root (HuggingFace Spaces)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]

# ───────────────────────────────────────────────────────────────────────────────
# Production GPU (Kubernetes)
# ───────────────────────────────────────────────────────────────────────────────
FROM base-gpu AS production-gpu

WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier l'application (nouvelle structure)
COPY src/ ./src/
COPY configs/ ./configs/
COPY pipelines/ ./pipelines/
COPY airflow/ ./airflow/
COPY models/ ./models/
COPY migrations/ ./migrations/
COPY app.py .

# Dossiers de travail
RUN mkdir -p /app/outputs /app/data /app/mlruns

# Variables d'environnement production
ENV ENV=production
ENV DB_MODE=postgres
ENV STORAGE_MODE=s3
ENV RAY_MODE=serve
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app

# Utilisateur non-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860 8000

CMD ["python", "app.py"]

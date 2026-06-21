# =============================================================================
# Apollo Clinical Intelligence — Dockerfile
# =============================================================================
#
# Multi-stage build: builder installs deps, runtime is a slim final image.
#
# ── Quick Start ───────────────────────────────────────────────────────────────
# Build:
#   docker build -t apollo-clinical-api:latest .
#
# Run (artifacts already pre-built locally):
#   docker run -p 8000:8000 \
#     -e LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1 \
#     -v $(pwd)/final_implementation/artifacts:/app/final_implementation/artifacts \
#     -v $(pwd)/End_Pipeline/Main_Apollo_Catalog.csv:/data/Main_Apollo_Catalog.csv:ro \
#     apollo-clinical-api:latest
#
# See docker-compose.yml for the recommended compose-based workflow.
# =============================================================================

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix directory (easy to copy to runtime)
COPY final_implementation/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir aiofiles python-multipart

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system libraries (libgomp for faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY final_implementation/ ./final_implementation/
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# Create directories
RUN mkdir -p \
    ./final_implementation/artifacts \
    /data \
    /root/.cache/huggingface

# ── Environment defaults (all overridable via -e or docker-compose) ────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_RELOAD=false
ENV API_WORKERS=1
ENV LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1
ENV LM_STUDIO_API_KEY=lm-studio
ENV LM_STUDIO_MODEL=local-model
ENV ARTIFACTS_DIR=/app/final_implementation/artifacts
ENV CATALOG_RAW_PATH=/data/Main_Apollo_Catalog.csv
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── Security: run as non-root ──────────────────────────────────────────────────
RUN useradd -m -u 1001 apolloapi && \
    chown -R apolloapi:apolloapi /app /data
USER apolloapi

EXPOSE 8000

# Health check — waits up to 2 min for startup (FAISS index load takes ~5s)
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]

#!/bin/bash
# =============================================================================
# Apollo Clinical Intelligence — Docker Entrypoint
# =============================================================================
# This script is the container's entrypoint. It:
#   1. Checks whether the FAISS artifacts (index + metadata) already exist.
#   2. If not, runs Stage 1 (semantic text) and Stage 2 (FAISS indexing).
#      This requires the raw catalog CSV to be mounted at $CATALOG_RAW_PATH.
#   3. Starts the Uvicorn API server.
#
# Environment variables consumed:
#   CATALOG_RAW_PATH  — path to Main_Apollo_Catalog.csv (mounted volume)
#   ARTIFACTS_DIR     — directory for generated FAISS artifacts
#   API_HOST / API_PORT / API_WORKERS
# =============================================================================
set -e

ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/final_implementation/artifacts}"
FAISS_INDEX="${ARTIFACTS_DIR}/apollo_fmcg.index"
METADATA_PKL="${ARTIFACTS_DIR}/apollo_fmcg_metadata.pkl"
SEMANTIC_CSV="${ARTIFACTS_DIR}/catalog_with_semantic_text.csv"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Apollo Clinical Intelligence — Container Startup"
echo "═══════════════════════════════════════════════════════"
echo ""

mkdir -p "$ARTIFACTS_DIR"

# ─── Stage 1: Semantic Text Creation ────────────────────────────────────────
if [ ! -f "$SEMANTIC_CSV" ]; then
    echo "[ENTRYPOINT] Stage 1 artifact not found. Running semantic text creation..."
    python -m final_implementation.data_pipeline.semantic_text_creation
    echo "[ENTRYPOINT] Stage 1 complete."
else
    echo "[ENTRYPOINT] Stage 1 artifact found: $(basename $SEMANTIC_CSV)"
fi

# ─── Stage 2: FAISS Index Builder ───────────────────────────────────────────
if [ ! -f "$FAISS_INDEX" ] || [ ! -f "$METADATA_PKL" ]; then
    echo "[ENTRYPOINT] Stage 2 artifacts not found. Building FAISS index..."
    echo "[ENTRYPOINT] WARNING: This will take 20-30 minutes on first run."
    python -m final_implementation.data_pipeline.vector_index_builder
    echo "[ENTRYPOINT] Stage 2 complete."
else
    echo "[ENTRYPOINT] Stage 2 artifacts found: $(basename $FAISS_INDEX), $(basename $METADATA_PKL)"
fi

echo ""
echo "[ENTRYPOINT] Starting Uvicorn API server on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}..."
echo ""

exec uvicorn final_implementation.api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --workers "${API_WORKERS:-1}"

"""
Apollo Clinical NLP Pipeline — Centralized Configuration
=========================================================
All runtime parameters are resolved here using a strict priority order:
  1. Environment variables (production / Docker)
  2. .env file (local development)
  3. Hardcoded defaults (fallback)

This module is imported by every other module in the pipeline, ensuring
a single source of truth for all configuration values.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root paths
# ---------------------------------------------------------------------------
# This file lives at:  final_implementation/config/settings.py
# Project root is:     Apollo_THIT_Solvathon/
_CONFIG_DIR = Path(__file__).resolve().parent          # .../final_implementation/config
_FINAL_IMPL_DIR = _CONFIG_DIR.parent                   # .../final_implementation
_PROJECT_ROOT = _FINAL_IMPL_DIR.parent                 # .../Apollo_THIT_Solvathon

# ---------------------------------------------------------------------------
# Raw Catalog (source data)
# ---------------------------------------------------------------------------
CATALOG_RAW_PATH: Path = Path(
    os.getenv(
        "CATALOG_RAW_PATH",
        str(_PROJECT_ROOT / "End_Pipeline" / "Main_Apollo_Catalog.csv"),
    )
)

# ---------------------------------------------------------------------------
# Artifact storage (generated files — FAISS index, metadata, processed CSV)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: Path = Path(
    os.getenv("ARTIFACTS_DIR", str(_FINAL_IMPL_DIR / "artifacts"))
)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEMANTIC_CATALOG_PATH: Path = ARTIFACTS_DIR / "catalog_with_semantic_text.csv"
FAISS_INDEX_PATH: Path = ARTIFACTS_DIR / "apollo_fmcg.index"
METADATA_PKL_PATH: Path = ARTIFACTS_DIR / "apollo_fmcg_metadata.pkl"

# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
# Dimensionality produced by all-MiniLM-L6-v2
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# FAISS index configuration
# ---------------------------------------------------------------------------
# "flat"   → IndexFlatL2  (exact search, best for <500k vectors)
# "ivf"    → IndexIVFFlat (approximate, best for millions of vectors)
FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "flat")
FAISS_IVF_NLIST: int = int(os.getenv("FAISS_IVF_NLIST", "256"))   # IVF clusters
FAISS_IVF_NPROBE: int = int(os.getenv("FAISS_IVF_NPROBE", "32"))  # Clusters to probe at query time
FAISS_TOP_K_CANDIDATES: int = int(os.getenv("FAISS_TOP_K_CANDIDATES", "50"))
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "10"))

# ---------------------------------------------------------------------------
# LM Studio / MedLlama configuration
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")   # LM Studio ignores this but openai client requires it
LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "local-model")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))     # Deterministic for clinical use
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# FastAPI server configuration
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))  # Keep at 1 for FAISS thread safety
API_TITLE: str = "Apollo Clinical Product Intelligence API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "AI-powered clinical product recommendation engine. "
    "Extracts structured medical intent from doctor-patient dialogues "
    "using JSL MedLlama and retrieves semantically matched products "
    "from the Apollo catalog via FAISS vector search."
)

# ---------------------------------------------------------------------------
# Catalog column definitions (mirrors the known CSV schema)
# ---------------------------------------------------------------------------
CATALOG_TEXT_COLUMNS: list[str] = [
    "name",
    "Product Information",
    "Key Benefits/Uses",
    "Classifier 1",
    "Classifier 2",
    "Classifier 3",
    "Product Type 1",
    "molecules",
    "Pharma Composition",
]

CATALOG_METADATA_COLUMNS: list[str] = [
    "name",
    "price",
    "is_prescription_required",
    "molecules",
    "Pack Form",
    "Product Form",
    "Consume Type",
    "Key Ingredient",
    "Age",
    "Gender",
    "100% vegetarian (Yes/No)",
    "Classifier 1",
    "Classifier 2",
    "Classifier 3",
    "Product Type 1",
    "Key Benefits/Uses",
    "Direction for use/Dosage",
    "Pharma Composition",
]

# ---------------------------------------------------------------------------
# Clinical safety thresholds
# ---------------------------------------------------------------------------
# Minimum vector similarity score (lower L2 distance = more similar)
# Products with distance > this value are dropped from candidates
MAX_VECTOR_DISTANCE: float = float(os.getenv("MAX_VECTOR_DISTANCE", "2.0"))

# Max diversity limits per brand/salt/form in final recommendations
MAX_PER_BRAND: int = int(os.getenv("MAX_PER_BRAND", "2"))
MAX_PER_SALT: int = int(os.getenv("MAX_PER_SALT", "3"))

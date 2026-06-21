"""
Apollo Clinical Pipeline — Retrieval: FAISS Lookup Engine
==========================================================
Loads the pre-built FAISS index and metadata, and provides a
``CatalogLookupEngine`` class that translates a structured clinical intent
dict (produced by the NLP extractor) into a set of candidate products via
approximate-nearest-neighbour search.

Design
------
The lookup proceeds in two stages:
  1. **Vector search** — build a query string from the intent, embed it
     with the same model used at index-build time, and fetch the top-K
     nearest products from FAISS.
  2. **Hard filters** — apply deterministic safety filters to the
     candidates (age group, gender, dietary constraints, Rx/OTC status,
     allergy blacklist, contraindicated ingredients).  Products that fail
     ANY hard filter are unconditionally excluded.

The distinction between vector-soft scoring (handled in ranker.py) and
hard filters (here) is intentional and mirrors clinical decision support
(CDS) best practice: hard rules encode patient safety and must never be
overridden by a similarity score.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from config.settings import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_TOP_K_CANDIDATES,
    METADATA_PKL_PATH,
    MAX_VECTOR_DISTANCE,
)
from utils.logger import AuditLogger, get_logger

log = get_logger(__name__)
audit = AuditLogger()


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_query_text(intent: dict[str, Any]) -> str:
    """
    Convert a structured clinical intent into a natural-language query string
    that can be embedded and compared to product semantic texts.

    We purposefully mimic the structure of our product semantic documents
    (benefits | description | category | molecules) so that the embedding
    space is symmetric — the query vector lands near the relevant product
    vectors.
    """
    parts: list[str] = []

    # Symptoms and conditions are the primary retrieval signal
    parts.extend(intent.get("symptoms", []))
    parts.extend(intent.get("diagnosed_conditions", []))

    # Recommended medications / molecules give a strong product-level signal
    for med in intent.get("recommended_medications", []):
        if med.get("name"):
            parts.append(med["name"])
        if med.get("route"):
            parts.append(med["route"])

    # Preferred dosage forms
    parts.extend(intent.get("preferred_forms", []))

    # Patient demographic context
    if intent.get("patient_age") is not None:
        age = intent["patient_age"]
        if age < 12:
            parts.append("paediatric child baby")
        elif age >= 65:
            parts.append("geriatric elderly senior")
        else:
            parts.append("adult")

    if intent.get("patient_gender"):
        parts.append(intent["patient_gender"])

    # Dietary context
    diets = intent.get("dietary_restrictions", {}) or {}
    if diets.get("diabetic"):
        parts.append("sugar free diabetic")
    if diets.get("vegetarian"):
        parts.append("vegetarian")

    query = " ".join(p.strip() for p in parts if p and p.strip())
    log.debug("query_text_built", extra={"query": query})
    return query


# ---------------------------------------------------------------------------
# Hard Filters
# ---------------------------------------------------------------------------

def _normalise_col(series: pd.Series) -> pd.Series:
    """Lowercase and strip a string column; fill NaN with empty string."""
    return series.fillna("").astype(str).str.lower().str.strip()


def apply_hard_filters(
    candidates: pd.DataFrame,
    intent: dict[str, Any],
    request_id: str = "n/a",
) -> pd.DataFrame:
    """
    Apply deterministic safety filters to a candidate product DataFrame.

    Filters applied (in order):
    1. Allergy / contraindication blacklist — ingredient-level exclusion.
    2. Prescription status — exclude Rx products if prescription_available is False.
    3. Age group suitability.
    4. Gender suitability.
    5. Vegetarian filter — only if patient explicitly requires vegetarian.
    6. Diabetic filter — flag syrup products for manual review rather than
       hard-exclude (sugar-free syrups are valid for diabetics).

    Returns
    -------
    pd.DataFrame
        Filtered candidates; always a subset of the input.
    """
    initial_count = len(candidates)
    filters_applied: list[str] = []

    # ------------------------------------------------------------------
    # 1. Allergy & Contraindication Blacklist
    # ------------------------------------------------------------------
    allergy_terms: list[str] = [
        a.lower().strip()
        for a in (
            list(intent.get("allergies", []))
            + list(intent.get("contraindications", []))
        )
        if a
    ]

    if allergy_terms and "molecules" in candidates.columns:
        mol_col = _normalise_col(candidates["molecules"])
        key_ing_col = _normalise_col(candidates.get("Key Ingredient", pd.Series([""] * len(candidates))))

        def _no_allergy(row_idx: int) -> bool:
            mol_val = mol_col.iloc[row_idx]
            key_val = key_ing_col.iloc[row_idx] if "Key Ingredient" in candidates.columns else ""
            combined = f"{mol_val} {key_val}"
            return not any(term in combined for term in allergy_terms)

        mask = [_no_allergy(i) for i in range(len(candidates))]
        removed = (~pd.Series(mask)).sum()
        candidates = candidates[mask]
        if removed:
            audit.log_safety_filter_event(request_id, int(removed), "allergy_contraindication_blacklist")
            filters_applied.append("allergy_blacklist")

    # ------------------------------------------------------------------
    # 2. Rx / OTC Filter
    # ------------------------------------------------------------------
    if intent.get("prescription_available") is False and "is_prescription_required" in candidates.columns:
        rx_col = _normalise_col(candidates["is_prescription_required"])
        before = len(candidates)
        # Common truthy representations in the catalog
        is_rx = rx_col.isin(["1", "yes", "true", "y"])
        candidates = candidates[~is_rx]
        removed = before - len(candidates)
        if removed:
            audit.log_safety_filter_event(request_id, removed, "prescription_required_but_unavailable")
            filters_applied.append("rx_filter")

    # ------------------------------------------------------------------
    # 3. Age Group Filter
    # ------------------------------------------------------------------
    if intent.get("patient_age") is not None and "Age" in candidates.columns:
        patient_age: int = intent["patient_age"]
        age_col = _normalise_col(candidates["Age"])

        def _age_compatible(age_val: str) -> bool:
            if not age_val or age_val in ("all", "nan", ""):
                return True
            if patient_age < 12:
                return any(kw in age_val for kw in ("child", "baby", "infant", "paediatric", "pediatric", "all"))
            elif patient_age >= 65:
                return any(kw in age_val for kw in ("adult", "geriatric", "elderly", "senior", "all"))
            else:
                return any(kw in age_val for kw in ("adult", "all"))

        before = len(candidates)
        mask = age_col.apply(_age_compatible)
        candidates = candidates[mask]
        removed = before - len(candidates)
        if removed:
            filters_applied.append("age_filter")

    # ------------------------------------------------------------------
    # 4. Gender Filter
    # ------------------------------------------------------------------
    patient_gender = (intent.get("patient_gender") or "").lower().strip()
    if patient_gender and "Gender" in candidates.columns:
        gender_col = _normalise_col(candidates["Gender"])
        before = len(candidates)
        mask = gender_col.apply(
            lambda g: g in ("", "all", "nan") or g == patient_gender or patient_gender == "other"
        )
        candidates = candidates[mask]
        removed = before - len(candidates)
        if removed:
            filters_applied.append("gender_filter")

    # ------------------------------------------------------------------
    # 5. Vegetarian Filter
    # ------------------------------------------------------------------
    diets = intent.get("dietary_restrictions", {}) or {}
    if diets.get("vegetarian") and "100% vegetarian (Yes/No)" in candidates.columns:
        veg_col = _normalise_col(candidates["100% vegetarian (Yes/No)"])
        before = len(candidates)
        candidates = candidates[veg_col.isin(["yes", "y", "1"])]
        removed = before - len(candidates)
        if removed:
            filters_applied.append("vegetarian_filter")

    final_count = len(candidates)
    log.info(
        "hard_filters_applied",
        extra={
            "request_id": request_id,
            "initial": initial_count,
            "final": final_count,
            "removed": initial_count - final_count,
            "filters": filters_applied,
        },
    )
    return candidates


# ---------------------------------------------------------------------------
# Lookup Engine
# ---------------------------------------------------------------------------

class CatalogLookupEngine:
    """
    Manages the FAISS index and metadata and serves product retrieval queries.

    Designed as a singleton for the FastAPI lifespan context — the index and
    metadata are loaded once at startup and held in memory for sub-millisecond
    retrieval during serving.
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: pd.DataFrame | None = None
        self._model: SentenceTransformer | None = None
        self._loaded = False

    def load(self) -> None:
        """
        Load FAISS index, metadata pickle, and embedding model into memory.
        Call this once at application startup.
        """
        if self._loaded:
            return

        # Load FAISS index
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run Stage 2 first: python final_implementation/data_pipeline/vector_index_builder.py"
            )
        log.info("loading_faiss_index", extra={"path": str(FAISS_INDEX_PATH)})
        self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        log.info("faiss_index_loaded", extra={"total_vectors": self._index.ntotal})

        # Load metadata
        if not METADATA_PKL_PATH.exists():
            raise FileNotFoundError(
                f"Metadata pickle not found at {METADATA_PKL_PATH}. Re-run Stage 2."
            )
        with open(METADATA_PKL_PATH, "rb") as fh:
            self._metadata = pickle.load(fh)
        log.info("metadata_loaded", extra={"rows": len(self._metadata)})

        # Load embedding model
        log.info("loading_embedding_model", extra={"model": EMBEDDING_MODEL_NAME})
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        log.info("embedding_model_ready")

        self._loaded = True

    def search(
        self,
        intent: dict[str, Any],
        top_k: int = FAISS_TOP_K_CANDIDATES,
        request_id: str = "n/a",
    ) -> pd.DataFrame:
        """
        Translate a clinical intent dict into a ranked, filtered candidate DataFrame.

        Parameters
        ----------
        intent:     Structured clinical intent from MedLlamaExtractor.
        top_k:      Number of FAISS candidates to retrieve before filtering.
        request_id: Correlation ID for audit logging.

        Returns
        -------
        pd.DataFrame
            Candidates with an added ``vector_distance`` column, filtered
            by hard safety rules, sorted by distance (ascending = more similar).
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call .load() first.")

        t0 = time.perf_counter()

        # 1. Build and embed query
        query_text = build_query_text(intent)
        query_vector = self._model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vector)

        # 2. FAISS search
        distances, indices = self._index.search(query_vector, top_k)
        distances = distances[0]
        indices = indices[0]

        # 3. Retrieve metadata rows
        valid_mask = (indices >= 0) & (distances <= MAX_VECTOR_DISTANCE)
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]

        if len(valid_indices) == 0:
            log.warning("faiss_no_valid_results", extra={"request_id": request_id, "query": query_text})
            return pd.DataFrame()

        candidates = self._metadata.iloc[valid_indices].copy()
        candidates = candidates.reset_index(drop=True)
        candidates["vector_distance"] = valid_distances

        retrieval_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "faiss_search_complete",
            extra={
                "request_id": request_id,
                "candidates_before_filter": len(candidates),
                "retrieval_ms": round(retrieval_ms, 2),
                "query_text": query_text,
            },
        )

        # 4. Apply hard safety filters
        candidates = apply_hard_filters(candidates, intent, request_id=request_id)
        candidates = candidates.sort_values("vector_distance").reset_index(drop=True)

        return candidates


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_engine_instance: CatalogLookupEngine | None = None


def get_lookup_engine() -> CatalogLookupEngine:
    """Return a module-level singleton CatalogLookupEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CatalogLookupEngine()
    return _engine_instance

"""
Apollo Clinical Pipeline — Retrieval: Multi-Factor Ranker & Diversity Engine
=============================================================================
Takes the filtered candidate product DataFrame from the lookup engine and
applies a multi-factor relevance scoring model to produce a final, diverse
ranked list of product recommendations.

Scoring model
-------------
The final score is a weighted linear combination of six clinical and
commercial factors:

  Factor                Weight   Rationale
  ──────────────────    ──────   ──────────────────────────────────────────
  Semantic similarity    0.40    Core FAISS L2 distance (inverted)
  Molecule match         0.25    Exact API match to recommended medications
  Form preference        0.15    Dosage form matches patient/doctor preference
  Age suitability        0.10    Demographic fit signal
  Rx status bonus        0.05    Reward OTC products if prescription is absent
  Vegetarian bonus       0.05    Reward veg products when requested

Diversity enforcement
---------------------
A post-scoring greedy diversity pass ensures the final top-K selection is
not dominated by the same brand or the same active molecule, mirroring the
DiversityManager pattern from the Apollo Recommender Workflow specification.

Explanation generation
----------------------
Each returned product is accompanied by a list of human-readable reasons
explaining why it was recommended, following the ExplanationGenerator pattern
from Phase 5 of the workflow specification.  These reasons are designed for
display in the clinical chat UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from config.settings import FINAL_TOP_K, MAX_PER_BRAND, MAX_PER_SALT
from utils.logger import AuditLogger, get_logger

log = get_logger(__name__)
audit = AuditLogger()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _invert_distance_score(distances: pd.Series) -> pd.Series:
    """
    Convert L2 distance to a similarity score in [0, 1].
    distance=0 → score=1.0 (identical vector)
    distance=max → score~0.0
    We use a Gaussian kernel: score = exp(-distance²/σ²) where σ=1.0.
    """
    return np.exp(-(distances ** 2))


def _molecule_match_score(row: pd.Series, recommended_meds: list[dict]) -> float:
    """
    Score = 1.0 if any recommended molecule appears in the product's
    molecules or Key Ingredient field. Score = 0.0 otherwise.
    """
    if not recommended_meds:
        return 0.5  # Neutral when no molecules are specified (no penalty)

    product_molecules = str(row.get("molecules", "") or "").lower()
    product_key_ing = str(row.get("Key Ingredient", "") or "").lower()
    combined = f"{product_molecules} {product_key_ing}"

    for med in recommended_meds:
        name = (med.get("name") or "").lower().strip()
        if name and name in combined:
            return 1.0
    return 0.0


def _form_preference_score(row: pd.Series, preferred_forms: list[str]) -> float:
    """
    Score = 1.0 if the product's form matches a preferred form.
    Score = 0.5 if no preference is stated (neutral).
    Score = 0.0 if a preference exists but this product doesn't match.
    """
    if not preferred_forms:
        return 0.5

    product_form = str(row.get("Product Form", "") or row.get("Pack Form", "") or "").lower()
    consume_type = str(row.get("Consume Type", "") or "").lower()
    combined = f"{product_form} {consume_type}"

    for form in preferred_forms:
        if form.lower().strip() in combined:
            return 1.0

    return 0.0


def _age_score(row: pd.Series, patient_age: int | None) -> float:
    """
    Reward exact age-group matches.  Neutral (0.5) when age is unknown.
    """
    if patient_age is None:
        return 0.5

    age_val = str(row.get("Age", "") or "").lower().strip()

    if not age_val or age_val in ("all", "nan", ""):
        return 0.7  # Suitable for all ages is a good signal

    if patient_age < 12 and any(kw in age_val for kw in ("child", "baby", "infant", "paediatric", "pediatric")):
        return 1.0
    if patient_age >= 65 and any(kw in age_val for kw in ("geriatric", "elderly", "senior")):
        return 1.0
    if 12 <= patient_age < 65 and "adult" in age_val:
        return 1.0

    return 0.3


def _rx_bonus(row: pd.Series, prescription_available: bool | None) -> float:
    """
    If no prescription is available, reward OTC products.
    If prescription IS available, Rx products are acceptable (neutral).
    """
    is_rx = str(row.get("is_prescription_required", "") or "").lower() in ("1", "yes", "true", "y")
    if prescription_available is False and not is_rx:
        return 1.0  # OTC preferred when no Rx
    if prescription_available is True:
        return 0.7  # Rx acceptable
    return 0.5  # Neutral


def _vegetarian_bonus(row: pd.Series, requires_veg: bool | None) -> float:
    if not requires_veg:
        return 0.5  # Neutral

    is_veg = str(row.get("100% vegetarian (Yes/No)", "") or "").lower().strip() in ("yes", "y", "1")
    return 1.0 if is_veg else 0.3


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------

def generate_reasons(row: pd.Series, intent: dict[str, Any]) -> list[str]:
    """
    Produce 2-4 human-readable clinical reasons for recommending a product.
    """
    reasons: list[str] = []
    recommended_meds = intent.get("recommended_medications", [])
    preferred_forms = intent.get("preferred_forms", [])

    # Molecule match
    product_molecules = str(row.get("molecules", "") or "").lower()
    for med in recommended_meds:
        name = (med.get("name") or "").lower()
        if name and name in product_molecules:
            dose = med.get("dose", "")
            reasons.append(
                f"✓ Contains recommended molecule: {row.get('molecules', '')} "
                + (f"({dose})" if dose else "")
            )
            break

    # Form match
    product_form = str(row.get("Product Form", "") or row.get("Pack Form", "") or "").strip()
    if product_form and preferred_forms:
        for form in preferred_forms:
            if form.lower() in product_form.lower():
                reasons.append(f"✓ Preferred dosage form: {product_form}")
                break

    # Vegetarian
    diets = intent.get("dietary_restrictions", {}) or {}
    if diets.get("vegetarian") and str(row.get("100% vegetarian (Yes/No)", "") or "").lower() in ("yes", "y"):
        reasons.append("✓ 100% vegetarian formulation")

    # Diabetic-safe
    if diets.get("diabetic"):
        form_low = str(row.get("Product Form", "") or "").lower()
        if "syrup" not in form_low:
            reasons.append("✓ Non-syrup formulation suitable for diabetic patients")

    # OTC
    is_rx = str(row.get("is_prescription_required", "") or "").lower() in ("1", "yes", "true", "y")
    if not is_rx:
        reasons.append("✓ Available over-the-counter (no prescription required)")

    # Category / indication
    cat1 = str(row.get("Classifier 1", "") or "").strip().title()
    if cat1:
        reasons.append(f"✓ Indicated for: {cat1}")

    return reasons[:4]  # Cap at 4 reasons for UI readability


# ---------------------------------------------------------------------------
# Main Ranker
# ---------------------------------------------------------------------------

class ClinicalRanker:
    """
    Applies multi-factor scoring and diversity enforcement to a candidate
    DataFrame, returning the final top-K recommended products.
    """

    WEIGHTS: dict[str, float] = {
        "semantic_similarity": 0.40,
        "molecule_match":      0.25,
        "form_preference":     0.15,
        "age_score":           0.10,
        "rx_bonus":            0.05,
        "vegetarian_bonus":    0.05,
    }

    def rank(
        self,
        candidates: pd.DataFrame,
        intent: dict[str, Any],
        final_k: int = FINAL_TOP_K,
        request_id: str = "n/a",
    ) -> list[dict[str, Any]]:
        """
        Score, diversify, and format the top-K recommendations.

        Returns
        -------
        list[dict[str, Any]]
            Ordered list of recommendation dicts, each containing product
            metadata, composite score, and human-readable reasons.
        """
        if candidates.empty:
            log.warning("ranker_received_empty_candidates", extra={"request_id": request_id})
            return []

        recommended_meds = intent.get("recommended_medications", [])
        preferred_forms = intent.get("preferred_forms", [])
        patient_age = intent.get("patient_age")
        prescription_available = intent.get("prescription_available")
        diets = intent.get("dietary_restrictions", {}) or {}
        requires_veg = diets.get("vegetarian")

        # ------------------------------------------------------------------
        # 1. Compute per-row factor scores
        # ------------------------------------------------------------------
        candidates = candidates.copy()
        candidates["_sem_score"]   = _invert_distance_score(candidates["vector_distance"])
        candidates["_mol_score"]   = candidates.apply(_molecule_match_score, axis=1, args=(recommended_meds,))
        candidates["_form_score"]  = candidates.apply(_form_preference_score, axis=1, args=(preferred_forms,))
        candidates["_age_score"]   = candidates.apply(_age_score, axis=1, args=(patient_age,))
        candidates["_rx_bonus"]    = candidates.apply(_rx_bonus, axis=1, args=(prescription_available,))
        candidates["_veg_bonus"]   = candidates.apply(_vegetarian_bonus, axis=1, args=(requires_veg,))

        w = self.WEIGHTS
        candidates["composite_score"] = (
            candidates["_sem_score"]  * w["semantic_similarity"] +
            candidates["_mol_score"]  * w["molecule_match"] +
            candidates["_form_score"] * w["form_preference"] +
            candidates["_age_score"]  * w["age_score"] +
            candidates["_rx_bonus"]   * w["rx_bonus"] +
            candidates["_veg_bonus"]  * w["vegetarian_bonus"]
        )

        candidates = candidates.sort_values("composite_score", ascending=False).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 2. Diversity enforcement (greedy pass)
        # ------------------------------------------------------------------
        selected: list[dict[str, Any]] = []
        brand_counts: dict[str, int] = {}
        salt_counts: dict[str, int] = {}

        for _, row in candidates.iterrows():
            if len(selected) >= final_k:
                break

            brand = str(row.get("name", "unknown") or "").split()[0].lower()  # First word of product name as brand proxy
            mol_raw = str(row.get("molecules", "") or "")
            salt = mol_raw.split(",")[0].strip().lower() if mol_raw else "unknown"

            # Enforce diversity limits
            if brand_counts.get(brand, 0) >= MAX_PER_BRAND:
                continue
            if salt_counts.get(salt, 0) >= MAX_PER_SALT:
                continue

            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            salt_counts[salt] = salt_counts.get(salt, 0) + 1

            # ------------------------------------------------------------------
            # 3. Build output record
            # ------------------------------------------------------------------
            reasons = generate_reasons(row, intent)
            product_record = {
                "rank": len(selected) + 1,
                "name": str(row.get("name", "N/A")),
                "price": row.get("price"),
                "molecules": str(row.get("molecules", "")),
                "product_form": str(row.get("Product Form", "") or row.get("Pack Form", "")),
                "consume_type": str(row.get("Consume Type", "")),
                "classifier_1": str(row.get("Classifier 1", "")),
                "classifier_2": str(row.get("Classifier 2", "")),
                "classifier_3": str(row.get("Classifier 3", "")),
                "is_prescription_required": row.get("is_prescription_required"),
                "age_group": str(row.get("Age", "")),
                "gender": str(row.get("Gender", "")),
                "is_vegetarian": str(row.get("100% vegetarian (Yes/No)", "")),
                "key_benefits": str(row.get("Key Benefits/Uses", ""))[:300],
                "composite_score": round(float(row["composite_score"]), 4),
                "vector_distance": round(float(row["vector_distance"]), 4),
                "reasons": reasons,
            }
            selected.append(product_record)

        audit.log_recommendation_event(
            request_id=request_id,
            recommended_product_names=[p["name"] for p in selected],
            filters_applied=["diversity_enforcement", "composite_scoring"],
            retrieval_time_ms=0.0,  # Timing handled upstream
        )

        log.info(
            "ranking_complete",
            extra={
                "request_id": request_id,
                "candidates_in": len(candidates),
                "recommendations_out": len(selected),
            },
        )
        return selected

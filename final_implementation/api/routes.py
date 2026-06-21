"""
Apollo Clinical Pipeline — API: Route Handlers
==============================================
Implements FastAPI route handlers for the recommendation service.

Routes:
  GET  /health            — Liveness / readiness probe
  POST /v1/recommend      — Core recommendation endpoint
  POST /v1/extract-intent — Standalone NLP extraction (debug / EHR integration)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from api.models import (
    ErrorResponse,
    ExtractionMetadata,
    HealthResponse,
    PipelineMetadata,
    ProductRecommendation,
    RecommendationRequest,
    RecommendationResponse,
)
from config.settings import API_VERSION, LM_STUDIO_BASE_URL, FINAL_TOP_K
from nlp.extractor import ClinicalExtractionError, MedLlamaExtractor, get_extractor
from retrieval.lookup_engine import CatalogLookupEngine, get_lookup_engine
from retrieval.ranker import ClinicalRanker
from utils.logger import AuditLogger, generate_request_id, get_logger

log = get_logger(__name__)
audit = AuditLogger()
router = APIRouter()
ranker = ClinicalRanker()


# ---------------------------------------------------------------------------
# Dependency injectors (enables easy mocking in tests)
# ---------------------------------------------------------------------------

def _get_extractor() -> MedLlamaExtractor:
    return get_extractor()


def _get_engine() -> CatalogLookupEngine:
    return get_lookup_engine()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness probe",
    tags=["System"],
)
async def health_check(engine: CatalogLookupEngine = Depends(_get_engine)) -> HealthResponse:
    index_loaded = engine._loaded
    vector_count = engine._index.ntotal if index_loaded and engine._index else None
    return HealthResponse(
        status="healthy" if index_loaded else "degraded",
        faiss_index_loaded=index_loaded,
        faiss_vector_count=vector_count,
        lm_studio_url=LM_STUDIO_BASE_URL,
        version=API_VERSION,
    )


# ---------------------------------------------------------------------------
# Core recommendation endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/v1/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate clinical product recommendations from a conversation",
    tags=["Recommendations"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request payload"},
        422: {"model": ErrorResponse, "description": "NLP extraction failed"},
        500: {"model": ErrorResponse, "description": "Internal pipeline error"},
        503: {"model": ErrorResponse, "description": "LM Studio unavailable"},
    },
)
async def recommend(
    payload: RecommendationRequest,
    extractor: MedLlamaExtractor = Depends(_get_extractor),
    engine: CatalogLookupEngine = Depends(_get_engine),
) -> RecommendationResponse:
    request_id = generate_request_id()
    overall_start = time.perf_counter()
    llm_time_ms: float | None = None

    log.info(
        "recommendation_request_received",
        extra={
            "request_id": request_id,
            "has_conversation": payload.conversation is not None,
            "has_pre_extracted_intent": payload.extracted_intent is not None,
            "top_k": payload.top_k,
        },
    )

    # ------------------------------------------------------------------
    # Stage 1: Resolve clinical intent
    # ------------------------------------------------------------------
    intent: dict[str, Any]

    if payload.extracted_intent is not None:
        # Bypass LLM — use pre-structured intent
        intent = dict(payload.extracted_intent)
        log.info("using_pre_extracted_intent", extra={"request_id": request_id})
    else:
        # Run MedLlama extraction
        try:
            llm_start = time.perf_counter()
            intent = extractor.extract(payload.conversation, request_id=request_id)  # type: ignore[arg-type]
            llm_time_ms = (time.perf_counter() - llm_start) * 1000
        except ClinicalExtractionError as exc:
            log.error(
                "clinical_extraction_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error_code": "EXTRACTION_FAILED", "message": str(exc), "request_id": request_id},
            )
        except Exception as exc:
            log.error(
                "lm_studio_unavailable",
                extra={"request_id": request_id, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "LM_STUDIO_UNAVAILABLE",
                    "message": f"Could not reach LM Studio: {exc}",
                    "request_id": request_id,
                },
            )

    # ------------------------------------------------------------------
    # Apply user_context overrides (explicit context takes precedence
    # over LLM inference — clinician-provided data is ground truth)
    # ------------------------------------------------------------------
    if payload.user_context:
        ctx = payload.user_context
        if ctx.age is not None:
            intent["patient_age"] = ctx.age
        if ctx.gender is not None:
            intent["patient_gender"] = ctx.gender
        if ctx.prescription_available is not None:
            intent["prescription_available"] = ctx.prescription_available
        if ctx.vegetarian is not None:
            intent.setdefault("dietary_restrictions", {})
            intent["dietary_restrictions"]["vegetarian"] = ctx.vegetarian
        if ctx.diabetic is not None:
            intent.setdefault("dietary_restrictions", {})
            intent["dietary_restrictions"]["diabetic"] = ctx.diabetic

    # ------------------------------------------------------------------
    # Stage 2: Retrieval + filtering
    # ------------------------------------------------------------------
    retrieval_start = time.perf_counter()
    candidates_df = engine.search(intent, top_k=50, request_id=request_id)
    candidates_before = len(candidates_df)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # ------------------------------------------------------------------
    # Stage 3: Multi-factor ranking + diversity
    # ------------------------------------------------------------------
    ranked = ranker.rank(
        candidates=candidates_df,
        intent=intent,
        final_k=payload.top_k,
        request_id=request_id,
    )
    candidates_after = len(ranked)

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    recommendations = [ProductRecommendation(**p) for p in ranked]

    extracted_meta = ExtractionMetadata(
        symptoms=intent.get("symptoms", []),
        diagnosed_conditions=intent.get("diagnosed_conditions", []),
        allergies=intent.get("allergies", []),
        recommended_medications=intent.get("recommended_medications", []),
        current_medications=intent.get("current_medications", []),
        contraindications=intent.get("contraindications", []),
        clinical_notes=intent.get("clinical_notes"),
    )

    total_ms = (time.perf_counter() - overall_start) * 1000

    pipeline_meta = PipelineMetadata(
        request_id=request_id,
        total_processing_time_ms=round(total_ms, 2),
        llm_extraction_time_ms=round(llm_time_ms, 2) if llm_time_ms else None,
        retrieval_time_ms=round(retrieval_ms, 2),
        candidates_before_filter=candidates_before,
        candidates_after_filter=candidates_after,
        faiss_index_size=engine._index.ntotal if engine._index else 0,
    )

    log.info(
        "recommendation_complete",
        extra={
            "request_id": request_id,
            "total_ms": round(total_ms, 2),
            "products_returned": len(recommendations),
        },
    )

    return RecommendationResponse(
        status="success",
        recommendations=recommendations,
        extracted_intent=extracted_meta,
        pipeline_metadata=pipeline_meta,
    )


# ---------------------------------------------------------------------------
# Standalone extraction endpoint (for EHR integration / debugging)
# ---------------------------------------------------------------------------

@router.post(
    "/v1/extract-intent",
    summary="Extract structured clinical intent from a conversation (no retrieval)",
    tags=["NLP"],
)
async def extract_intent_only(
    payload: RecommendationRequest,
    extractor: MedLlamaExtractor = Depends(_get_extractor),
) -> dict[str, Any]:
    """
    Runs only the NLP extraction stage and returns the raw structured intent.
    Useful for EHR systems that manage their own product lookup.
    """
    if not payload.conversation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'conversation' field is required for intent extraction.",
        )
    request_id = generate_request_id()
    try:
        intent = extractor.extract(payload.conversation, request_id=request_id)
    except ClinicalExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"request_id": request_id, "intent": intent}

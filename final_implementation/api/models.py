"""
Apollo Clinical Pipeline — API: Pydantic Request & Response Models
==================================================================
All API I/O is validated via Pydantic v2 models.  Strong type contracts
ensure that malformed requests are rejected at the transport layer before
reaching clinical processing logic — a key requirement for HIPAA-compliant
medical software.

Schema design follows the Apollo Recommender Workflow specification and is
aligned with HL7 FHIR R4 conceptual terminology where applicable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class UserContext(BaseModel):
    """Optional contextual metadata about the patient."""
    age: int | None = Field(None, ge=0, le=130, description="Patient age in years.")
    gender: str | None = Field(None, description="Patient gender (male/female/other).")
    prescription_available: bool | None = Field(
        None,
        description="True if a doctor has issued a prescription in this session.",
    )
    vegetarian: bool | None = Field(None, description="True if patient requires vegetarian products.")
    diabetic: bool | None = Field(None, description="True if patient is diabetic.")

    @field_validator("gender")
    @classmethod
    def normalise_gender(cls, v: str | None) -> str | None:
        if v:
            return v.lower().strip()
        return v


class RecommendationRequest(BaseModel):
    """
    Primary API request payload.

    Clients may supply EITHER a raw conversation (for LLM-driven extraction)
    OR a pre-extracted ``extracted_intent`` dict (for cases where the calling
    system already has structured intent, e.g., from an EHR integration).
    At least one of the two must be provided.
    """

    conversation: str | None = Field(
        None,
        description=(
            "Raw doctor-patient conversation text. "
            "The pipeline will extract clinical intent from this using MedLlama."
        ),
        min_length=10,
    )
    extracted_intent: dict[str, Any] | None = Field(
        None,
        description=(
            "Pre-extracted structured intent dict (bypasses LLM extraction). "
            "Must conform to the MedLlama output schema."
        ),
    )
    user_context: UserContext | None = Field(
        None,
        description="Optional patient context override (takes precedence over LLM-extracted values).",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of product recommendations to return.",
    )

    @field_validator("conversation", "extracted_intent", mode="before")
    @classmethod
    def at_least_one_input(cls, v: object) -> object:
        # Validation happens at model level, not field level, but we declare
        # the validator here for documentation purposes.
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.conversation and not self.extracted_intent:
            raise ValueError(
                "At least one of 'conversation' or 'extracted_intent' must be provided."
            )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ProductRecommendation(BaseModel):
    """A single product recommendation with score and clinical reasons."""
    rank: int
    name: str
    price: float | None = None
    molecules: str | None = None
    product_form: str | None = None
    consume_type: str | None = None
    classifier_1: str | None = None
    classifier_2: str | None = None
    classifier_3: str | None = None
    is_prescription_required: Any | None = None
    age_group: str | None = None
    gender: str | None = None
    is_vegetarian: str | None = None
    key_benefits: str | None = None
    composite_score: float
    vector_distance: float
    reasons: list[str] = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    """Summary of the NLP extraction results."""
    symptoms: list[str] = Field(default_factory=list)
    diagnosed_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    recommended_medications: list[dict[str, Any]] = Field(default_factory=list)
    current_medications: list[dict[str, Any]] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    clinical_notes: str | None = None


class PipelineMetadata(BaseModel):
    """Operational metadata attached to every response for observability."""
    request_id: str
    total_processing_time_ms: float
    llm_extraction_time_ms: float | None = None
    retrieval_time_ms: float
    candidates_before_filter: int
    candidates_after_filter: int
    faiss_index_size: int


class RecommendationResponse(BaseModel):
    """Full API response payload."""
    status: str = "success"
    recommendations: list[ProductRecommendation] = Field(default_factory=list)
    extracted_intent: ExtractionMetadata | None = None
    pipeline_metadata: PipelineMetadata | None = None


class ErrorResponse(BaseModel):
    """Standardised error response payload."""
    status: str = "error"
    error_code: str
    message: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    faiss_index_loaded: bool
    faiss_vector_count: int | None = None
    lm_studio_url: str
    version: str

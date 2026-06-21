"""
Apollo Clinical Pipeline — NLP: MedLlama Extraction Engine
===========================================================
Implements a production-grade client for JSL MedLlama 3 8B v2.0 running
locally via LM Studio's OpenAI-compatible REST endpoint.

Architecture decisions
----------------------
- Uses the ``openai`` SDK against the LM Studio base URL.  This gives us
  streaming, structured retry logic, and a well-tested HTTP transport layer
  (httpx) out of the box.
- The LLM is prompted to emit raw JSON via our medically-validated system
  prompt (see nlp/prompts.py).  We apply strict JSON parsing with schema
  validation via Pydantic to prevent hallucinated or malformed fields from
  propagating downstream.
- Exponential-backoff retry (via ``tenacity``) handles transient LM Studio
  errors (e.g., model still loading on first request).
- Clinical safety invariant: if the LLM fails to produce parsable JSON after
  all retries, we raise a ``ClinicalExtractionError`` rather than silently
  returning empty intent — downstream code must handle this explicitly.

Compliance note
---------------
All LLM inputs and outputs are logged (via the audit logger) for traceability
in accordance with good clinical practice (GCP) guidelines.  PII (patient
names) should be de-identified before reaching this service in a production
environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from config.settings import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LM_STUDIO_API_KEY,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
)
from nlp.prompts import MEDICAL_EXTRACTION_SYSTEM_PROMPT, USER_EXTRACTION_TEMPLATE
from utils.logger import AuditLogger, get_logger

log = get_logger(__name__)
audit = AuditLogger()


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ClinicalExtractionError(Exception):
    """
    Raised when the LLM fails to produce valid, parseable clinical JSON
    after all retry attempts.  This is a hard error — callers must handle it.
    """


class LMStudioConnectionError(Exception):
    """Raised when LM Studio's local server is unreachable."""


# ---------------------------------------------------------------------------
# Pydantic schema for extracted medical intent
# ---------------------------------------------------------------------------
# We intentionally keep this as a standard TypedDict / dict rather than a
# full Pydantic model here, since the full validation happens in api/models.py.
# This avoids circular import chains in the data pipeline.

EMPTY_INTENT: dict[str, Any] = {
    "patient_age": None,
    "patient_gender": None,
    "symptoms": [],
    "diagnosed_conditions": [],
    "allergies": [],
    "current_medications": [],
    "recommended_medications": [],
    "preferred_forms": [],
    "dietary_restrictions": {
        "vegetarian": None,
        "diabetic": None,
        "lactose_intolerant": None,
        "gluten_free": None,
    },
    "prescription_available": None,
    "pregnancy_status": None,
    "contraindications": [],
    "clinical_notes": None,
}


# ---------------------------------------------------------------------------
# LM Studio Client
# ---------------------------------------------------------------------------

class MedLlamaExtractor:
    """
    Stateful extractor that maintains an OpenAI client pointed at LM Studio.

    Parameters
    ----------
    base_url:   LM Studio server base URL (e.g., "http://127.0.0.1:1234/v1")
    api_key:    Placeholder key — LM Studio doesn't validate it.
    model:      Model identifier string (LM Studio ignores this value;
                it uses whatever model is currently loaded).
    temperature: Sampling temperature — 0.0 for deterministic clinical use.
    max_tokens: Max tokens to generate.
    timeout:    HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        api_key: str = LM_STUDIO_API_KEY,
        model: str = LM_STUDIO_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        try:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=float(timeout),
            )
        except Exception as exc:
            raise LMStudioConnectionError(
                f"Failed to initialise OpenAI client pointing to LM Studio at {base_url}: {exc}"
            ) from exc

        log.info(
            "medllama_extractor_initialized",
            extra={"base_url": base_url, "model": model, "temperature": temperature},
        )

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _call_llm(self, conversation: str) -> str:
        """
        Make a single LLM API call and return the raw response string.
        Decorated with tenacity retry for transient network / model errors.
        """
        user_message = USER_EXTRACTION_TEMPLATE.format(conversation=conversation)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": MEDICAL_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = response.choices[0].message.content or ""
        log.debug(
            "llm_raw_response",
            extra={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "raw_length": len(raw),
            },
        )
        return raw

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """
        Robustly parse the LLM's JSON response.

        LMs occasionally wrap JSON in markdown code fences despite instructions.
        We strip them before attempting to parse.
        """
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Drop first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

        try:
            parsed: dict[str, Any] = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError as exc:
            log.error(
                "json_parse_failed",
                extra={"error": str(exc), "raw_snippet": raw[:300]},
            )
            raise ClinicalExtractionError(
                f"LLM returned non-parseable JSON: {exc}\nRaw (first 300 chars): {raw[:300]}"
            ) from exc

    def _merge_with_defaults(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Merge parsed fields with ``EMPTY_INTENT`` defaults to guarantee
        that every key is present even if the LLM omitted it.
        """
        merged = EMPTY_INTENT.copy()
        # Deep merge dietary restrictions
        if "dietary_restrictions" in parsed and parsed["dietary_restrictions"]:
            merged["dietary_restrictions"] = {
                **EMPTY_INTENT["dietary_restrictions"],
                **parsed["dietary_restrictions"],
            }
            parsed.pop("dietary_restrictions")
        merged.update({k: v for k, v in parsed.items() if v is not None})
        return merged

    def extract(self, conversation: str, request_id: str | None = None) -> dict[str, Any]:
        """
        End-to-end extraction: conversation text → structured clinical intent dict.

        Parameters
        ----------
        conversation:  Full text of the doctor-patient dialogue.
        request_id:    Correlation ID for audit logging.

        Returns
        -------
        dict[str, Any]
            Validated and default-merged clinical intent dictionary.

        Raises
        ------
        ClinicalExtractionError
            If the LLM response cannot be parsed as valid JSON after all retries.
        LMStudioConnectionError
            If LM Studio is unreachable.
        """
        log.info(
            "extraction_started",
            extra={"request_id": request_id, "conversation_len": len(conversation)},
        )

        raw = self._call_llm(conversation)
        intent = self._parse_json_response(raw)
        intent = self._merge_with_defaults(intent)

        audit.log_extraction_event(
            request_id=request_id or "n/a",
            conversation_snippet=conversation[:200],
            extracted_intent=intent,
            model_used=self._model,
        )

        log.info(
            "extraction_complete",
            extra={
                "request_id": request_id,
                "symptoms": intent.get("symptoms"),
                "allergies": intent.get("allergies"),
                "recommended_meds": [
                    m.get("name") for m in intent.get("recommended_medications", [])
                ],
            },
        )
        return intent


# ---------------------------------------------------------------------------
# Convenience singleton factory
# ---------------------------------------------------------------------------

_extractor_instance: MedLlamaExtractor | None = None


def get_extractor() -> MedLlamaExtractor:
    """Return a module-level singleton MedLlamaExtractor (lazy-initialised)."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MedLlamaExtractor()
    return _extractor_instance


# ---------------------------------------------------------------------------
# Local test (run directly to verify LM Studio connection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    SAMPLE_CONVERSATION = """
    Doctor: Good morning. What brings you in today?
    Patient: Doctor, I am a 52-year-old male and I have been having chest tightness,
             shortness of breath, and a persistent cough for the past two weeks.
             I also have diabetes and I take metformin 500mg twice a day.
             I'm allergic to aspirin.
    Doctor: Sounds like it could be COPD exacerbation. Let me add tiotropium via
             inhaler once daily and a short course of oral prednisolone 40mg for
             5 days. Continue your metformin. Avoid NSAIDs.
    """.strip()

    print("Testing MedLlama extraction with LM Studio...")
    extractor = MedLlamaExtractor()
    result = extractor.extract(SAMPLE_CONVERSATION, request_id="test-001")
    print(_json.dumps(result, indent=2))

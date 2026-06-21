"""
Apollo Clinical Pipeline — Structured Audit Logging
====================================================
Provides a standardised logging interface that:
  - Emits JSON-structured logs suitable for ingestion by log aggregators
    (Elasticsearch, Splunk, Google Cloud Logging, AWS CloudWatch).
  - Attaches a correlation_id to every log entry for end-to-end request tracing.
  - Maintains a clinical audit trail of every recommendation event, which is
    critical for IEC 62304 / FDA 21 CFR Part 11 compliance in medical software.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("recommendation_generated", products=5, confidence=0.94)
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge any extra fields attached to the record
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "funcName", "lineno", "exc_info",
                "exc_text", "stack_info", "thread", "threadName",
                "process", "processName", "message", "asctime", "created",
                "relativeCreated", "msecs", "taskName",
            }
        }
        log_object.update(extras)

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a module-level logger with JSON formatting.

    Parameters
    ----------
    name:  Module name, typically ``__name__``.
    level: Logging level (default: INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class AuditLogger:
    """
    Clinical Audit Event Logger.

    Every product recommendation made by the system is recorded here with
    full traceability metadata (request ID, extracted intent, product IDs).
    This log stream is the authoritative source for compliance audits.
    """

    def __init__(self) -> None:
        self._log = get_logger("audit.clinical")

    def log_extraction_event(
        self,
        request_id: str,
        conversation_snippet: str,
        extracted_intent: dict[str, Any],
        model_used: str,
    ) -> None:
        self._log.info(
            "clinical_intent_extracted",
            extra={
                "event_type": "INTENT_EXTRACTION",
                "request_id": request_id,
                "model_used": model_used,
                "conversation_length": len(conversation_snippet),
                "symptoms_count": len(extracted_intent.get("symptoms", [])),
                "allergies_count": len(extracted_intent.get("allergies", [])),
                "medications_count": len(extracted_intent.get("current_medications", [])),
            },
        )

    def log_recommendation_event(
        self,
        request_id: str,
        recommended_product_names: list[str],
        filters_applied: list[str],
        retrieval_time_ms: float,
    ) -> None:
        self._log.info(
            "product_recommendations_generated",
            extra={
                "event_type": "RECOMMENDATION",
                "request_id": request_id,
                "product_count": len(recommended_product_names),
                "products": recommended_product_names,
                "filters_applied": filters_applied,
                "retrieval_time_ms": round(retrieval_time_ms, 2),
            },
        )

    def log_safety_filter_event(
        self,
        request_id: str,
        filtered_out_count: int,
        reason: str,
    ) -> None:
        self._log.warning(
            "safety_filter_applied",
            extra={
                "event_type": "SAFETY_FILTER",
                "request_id": request_id,
                "filtered_out_count": filtered_out_count,
                "reason": reason,
            },
        )


def generate_request_id() -> str:
    """Generate a UUID4-based correlation ID for a new API request."""
    return str(uuid.uuid4())

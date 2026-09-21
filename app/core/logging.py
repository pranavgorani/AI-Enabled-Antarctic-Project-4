"""
Structured JSON Logger for Polar Navigator AI
Ensures zero credential leakage, structured event tracking, and request correlation.
"""

import json
import logging
import sys
import os
import re
from datetime import datetime
from typing import Any

SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key|secret|password|token|auth)['\"]?\s*[:=]\s*['\"]?([^'\"&\s]+)", re.IGNORECASE),
]


class ScrubbingJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON and scrubs any credential strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "polar_navigator_ai",
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add any extra contextual attributes passed in kwargs
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "data_source"):
            log_obj["data_source"] = record.data_source
        if hasattr(record, "model_version"):
            log_obj["model_version"] = record.model_version
        if hasattr(record, "event"):
            log_obj["event"] = record.event

        json_str = json.dumps(log_obj)

        # Scrub sensitive credentials
        for pattern in SENSITIVE_PATTERNS:
            json_str = pattern.sub(r"\1=***REDACTED***", json_str)

        return json_str


def setup_logger(name: str = "polar_navigator", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ScrubbingJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logger()

"""Environment-driven runtime settings for the EchoBrief pipeline.

Nothing outside this module reads ``os.environ``; services receive a
``Settings`` instance through their constructors.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, field_validator

_ENV_FIELDS = {
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "WHISPER_MODEL_SIZE": "whisper_model_size",
    "WHISPER_COMPUTE_TYPE": "whisper_compute_type",
}


class Settings(BaseModel):
    """Runtime configuration. Build with ``Settings.from_env()``."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    whisper_model_size: str = "small"
    whisper_compute_type: str = "int8"

    @field_validator("ollama_model", "whisper_model_size", "whisper_compute_type")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _normalize_http_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an http(s) URL, e.g. http://localhost:11434")
        return value

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from an environment mapping (default: ``os.environ``)."""
        source = os.environ if env is None else env
        present = {
            field: source[var] for var, field in _ENV_FIELDS.items() if source.get(var) is not None
        }
        return cls(**present)

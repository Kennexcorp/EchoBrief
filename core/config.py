"""Environment-driven runtime settings for the EchoBrief pipeline.

Nothing outside this module reads ``os.environ``; services receive a
``Settings`` instance through their constructors.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

_ENV_FIELDS = {
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "WHISPER_MODEL_SIZE": "whisper_model_size",
    "WHISPER_COMPUTE_TYPE": "whisper_compute_type",
    "ELEVENLABS_API_KEY": "elevenlabs_api_key",
    "ELEVENLABS_VOICE_ID": "elevenlabs_voice_id",
    "ELEVENLABS_MODEL": "elevenlabs_model",
}


class Settings(BaseModel):
    """Runtime configuration. Build with ``Settings.from_env()``."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    whisper_model_size: str = "small"
    whisper_compute_type: str = "int8"

    # Optional ElevenLabs voice output (F6). Off unless an API key is set, so the
    # default pipeline stays 100% local; audio is the only feature that leaves the machine.
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # "Rachel", a default public voice
    elevenlabs_model: str = "eleven_multilingual_v2"

    @field_validator("ollama_model", "whisper_model_size", "whisper_compute_type")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("elevenlabs_api_key", "elevenlabs_voice_id", "elevenlabs_model")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @property
    def tts_enabled(self) -> bool:
        """Voice output is on only when an ElevenLabs API key is configured."""
        return bool(self.elevenlabs_api_key)

    @field_validator("ollama_base_url")
    @classmethod
    def _normalize_http_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an http(s) URL, e.g. http://localhost:11434")
        return value

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from an environment mapping (default: ``os.environ``).

        When reading the real process environment, a local ``.env`` file is
        loaded first. It is a no-op if none exists, and real environment
        variables always win, so Docker/CI setups are unaffected.
        """
        if env is None:
            load_dotenv()
            source: Mapping[str, str] = os.environ
        else:
            source = env
        present = {
            field: source[var] for var, field in _ENV_FIELDS.items() if source.get(var) is not None
        }
        return cls(**present)

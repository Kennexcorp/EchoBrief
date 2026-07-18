"""Tests for core.config.Settings — env-driven runtime configuration."""

import pytest
from pydantic import ValidationError

from core.config import Settings


class TestDefaults:
    def test_empty_environment_yields_documented_defaults(self) -> None:
        settings = Settings.from_env({})

        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3.1:8b"
        assert settings.whisper_model_size == "small"
        assert settings.whisper_compute_type == "int8"


class TestEnvOverrides:
    def test_reads_all_four_variables(self) -> None:
        settings = Settings.from_env(
            {
                "OLLAMA_BASE_URL": "http://ollama:11434",
                "OLLAMA_MODEL": "qwen2.5:7b",
                "WHISPER_MODEL_SIZE": "base",
                "WHISPER_COMPUTE_TYPE": "float16",
            }
        )

        assert settings.ollama_base_url == "http://ollama:11434"
        assert settings.ollama_model == "qwen2.5:7b"
        assert settings.whisper_model_size == "base"
        assert settings.whisper_compute_type == "float16"

    def test_unrelated_variables_are_ignored(self) -> None:
        settings = Settings.from_env({"PATH": "/usr/bin", "EDITOR": "vim"})

        assert settings.ollama_model == "llama3.1:8b"

    def test_defaults_to_process_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "mistral:7b")

        settings = Settings.from_env()

        assert settings.ollama_model == "mistral:7b"


class TestValidation:
    def test_blank_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings.from_env({"OLLAMA_MODEL": "   "})

    def test_base_url_without_http_scheme_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings.from_env({"OLLAMA_BASE_URL": "ollama:11434"})

    def test_base_url_trailing_slash_is_normalized(self) -> None:
        settings = Settings.from_env({"OLLAMA_BASE_URL": "http://localhost:11434/"})

        assert settings.ollama_base_url == "http://localhost:11434"

"""Tests for core.health — Ollama startup health check."""

from typing import Any

from core.config import Settings
from core.health import check_ollama


def fake_fetch(tags: list[str]):
    def fetch(url: str) -> dict[str, Any]:
        return {"models": [{"name": name} for name in tags]}

    return fetch


def failing_fetch(url: str) -> dict[str, Any]:
    raise OSError("connection refused")


class TestUnreachable:
    def test_reports_exact_url_and_serve_hint(self) -> None:
        settings = Settings.from_env({"OLLAMA_BASE_URL": "http://somehost:11434"})

        status = check_ollama(settings, fetch_json=failing_fetch)

        assert not status.ok
        assert "http://somehost:11434" in status.message
        assert "ollama serve" in status.message


class TestModelMissing:
    def test_reports_exact_pull_command(self) -> None:
        settings = Settings.from_env({"OLLAMA_MODEL": "qwen2.5:7b"})

        status = check_ollama(settings, fetch_json=fake_fetch(["llama3.1:8b"]))

        assert not status.ok
        assert "ollama pull qwen2.5:7b" in status.message


class TestHealthy:
    def test_exact_model_match(self) -> None:
        settings = Settings.from_env({})

        status = check_ollama(settings, fetch_json=fake_fetch(["llama3.1:8b", "mistral:7b"]))

        assert status.ok

    def test_bare_model_name_matches_any_tag(self) -> None:
        settings = Settings.from_env({"OLLAMA_MODEL": "llama3.1"})

        status = check_ollama(settings, fetch_json=fake_fetch(["llama3.1:latest"]))

        assert status.ok

    def test_malformed_tags_response_is_not_ok(self) -> None:
        settings = Settings.from_env({})

        status = check_ollama(settings, fetch_json=lambda url: {"unexpected": True})

        assert not status.ok

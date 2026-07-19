"""Startup health check for the Ollama daemon (reliability NFR).

Failure messages always state the exact URL tried and the exact fix command,
so misconfiguration is self-diagnosing.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.config import Settings

_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    message: str


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _model_matches(wanted: str, available: str) -> bool:
    if wanted == available:
        return True
    # A bare name like "llama3.1" matches any of its tags, e.g. "llama3.1:latest".
    return ":" not in wanted and available.split(":", 1)[0] == wanted


def check_ollama(
    settings: Settings, fetch_json: Callable[[str], Any] = _fetch_json
) -> HealthStatus:
    """Check that the Ollama daemon is reachable and the configured model is pulled."""
    tags_url = f"{settings.ollama_base_url}/api/tags"
    try:
        payload = fetch_json(tags_url)
        available = [model["name"] for model in payload["models"]]
    except (OSError, ValueError, KeyError, TypeError):
        return HealthStatus(
            ok=False,
            message=(
                f"Ollama is not reachable at {settings.ollama_base_url}. "
                "Start it with `ollama serve` (or check OLLAMA_BASE_URL in your .env)."
            ),
        )

    if not any(_model_matches(settings.ollama_model, name) for name in available):
        return HealthStatus(
            ok=False,
            message=(
                f"Model '{settings.ollama_model}' is not available in Ollama at "
                f"{settings.ollama_base_url}. Pull it with `ollama pull {settings.ollama_model}` "
                "(or set OLLAMA_MODEL to one of: " + (", ".join(available) or "none pulled") + ")."
            ),
        )

    return HealthStatus(
        ok=True,
        message=f"Ollama reachable at {settings.ollama_base_url}; "
        f"model '{settings.ollama_model}' is available.",
    )

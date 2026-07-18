"""Insight extraction: transcript text → validated ``Brief`` via a local LLM.

Parse failures trigger exactly one repair-prompt retry; if that also fails,
the caller gets the raw model text instead of an exception (reliability NFR).
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_ollama import ChatOllama
from pydantic import ValidationError

from core.config import Settings
from core.prompts import build_brief_messages, build_repair_messages
from core.schemas import Brief, InsightResult


class ChatModel(Protocol):
    """The single method this engine needs from a chat model."""

    def invoke(self, messages: Any) -> Any: ...


class InsightEngine:
    """Produces an ``InsightResult`` from transcript text using an injected chat model."""

    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model

    def generate_brief(self, transcript: str, user_context: str | None = None) -> InsightResult:
        messages = build_brief_messages(transcript, user_context)
        content = self._invoke(messages)
        brief, error = self._parse(content)
        if brief is not None:
            return InsightResult(brief=brief, raw_text=content)

        retry_messages = build_repair_messages(messages, content, error)
        retry_content = self._invoke(retry_messages)
        brief, _ = self._parse(retry_content)
        return InsightResult(brief=brief, raw_text=retry_content)

    def _invoke(self, messages: Any) -> str:
        response = self._chat_model.invoke(messages)
        return str(response.content)

    def _parse(self, content: str) -> tuple[Brief | None, str]:
        """Parse model output into a Brief; returns (brief, "") or (None, error)."""
        try:
            return Brief.model_validate_json(_strip_code_fences(content)), ""
        except ValidationError as exc:
            return None, str(exc)


def _strip_code_fences(text: str) -> str:
    """Remove a wrapping ``` block if present — models add them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        text = text.strip().removesuffix("```")
    return text.strip()


def create_insight_engine(settings: Settings) -> InsightEngine:
    """Build an engine backed by ChatOllama with Ollama's JSON-schema mode enforced."""
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        format=Brief.model_json_schema(),
        temperature=0.0,
    )
    return InsightEngine(model)

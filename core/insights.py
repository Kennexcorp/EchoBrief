"""Insight extraction: transcript text → validated ``Brief`` via a local LLM.

Parse failures trigger exactly one repair-prompt retry; if that also fails,
the caller gets the raw model text instead of an exception (reliability NFR).
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from core.chunking import DEFAULT_MAX_CHUNK_TOKENS, chunk_segments, estimate_tokens
from core.config import Settings
from core.prompts import (
    build_brief_messages,
    build_chunk_brief_messages,
    build_repair_messages,
    build_synthesis_messages,
)
from core.schemas import Brief, InsightResult, Transcript


class ChatModel(Protocol):
    """The single method this engine needs from a chat model."""

    def invoke(self, messages: Any) -> Any: ...


class InsightEngine:
    """Produces an ``InsightResult`` from a transcript using an injected chat model.

    Transcripts that exceed the chunk budget are map-reduced: each chunk of
    whole segments yields a per-portion brief, and a final synthesis pass
    merges them. Plain-string input is always processed in a single pass.
    """

    def __init__(
        self, chat_model: ChatModel, max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    ) -> None:
        self._chat_model = chat_model
        self._max_chunk_tokens = max_chunk_tokens

    def generate_brief(
        self, transcript: Transcript | str, user_context: str | None = None
    ) -> InsightResult:
        text = transcript.text if isinstance(transcript, Transcript) else transcript
        if isinstance(transcript, str) or estimate_tokens(text) <= self._max_chunk_tokens:
            return self._run(build_brief_messages(text, user_context))

        chunks = chunk_segments(transcript.segments, self._max_chunk_tokens)
        part_notes = []
        for part, chunk in enumerate(chunks, start=1):
            chunk_text = " ".join(segment.text for segment in chunk)
            chunk_result = self._run(build_chunk_brief_messages(chunk_text, part, len(chunks)))
            # A failed chunk still contributes its raw text — synthesis can read prose.
            part_notes.append(
                chunk_result.brief.model_dump_json()
                if chunk_result.brief is not None
                else chunk_result.raw_text
            )
        return self._run(build_synthesis_messages(part_notes, user_context))

    def _run(self, messages: list[BaseMessage]) -> InsightResult:
        """Invoke → parse → one repair retry → raw-text fallback. Never raises."""
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
        # Ollama defaults to a 2-4k context; 8k fits a full chunk (3k tokens)
        # plus the system prompt and JSON response, and stays laptop-friendly.
        num_ctx=8192,
    )
    return InsightEngine(model)

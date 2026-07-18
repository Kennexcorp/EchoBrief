"""Segment-boundary chunking for transcripts that exceed the model context.

Token counts are estimated at ~4 characters per token — close enough for
budgeting chunks without pulling in a tokenizer dependency.
"""

from __future__ import annotations

import math

from core.schemas import TranscriptSegment

# Fits comfortably inside the 8k context configured on the Ollama model,
# leaving room for the system prompt and the JSON response.
DEFAULT_MAX_CHUNK_TOKENS = 3000

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def chunk_segments(
    segments: list[TranscriptSegment], max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
) -> list[list[TranscriptSegment]]:
    """Greedily pack whole segments into chunks of at most ``max_tokens``.

    Segments are never split, so no sentence is cut mid-thought; a single
    segment larger than the budget becomes its own chunk.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0

    for segment in segments:
        segment_tokens = estimate_tokens(segment.text) + 1  # +1 for the joining space
        if current and current_tokens + segment_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(segment)
        current_tokens += segment_tokens

    if current:
        chunks.append(current)
    return chunks

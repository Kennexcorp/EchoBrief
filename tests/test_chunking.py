"""Tests for core.chunking — segment-boundary chunking for long transcripts."""

import pytest

from core.chunking import DEFAULT_MAX_CHUNK_TOKENS, chunk_segments, estimate_tokens
from core.schemas import TranscriptSegment


def segment(text: str, index: int = 0) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=float(index), end=float(index) + 1.0)


class TestEstimateTokens:
    def test_scales_with_text_length(self) -> None:
        assert estimate_tokens("a" * 400) == 100

    def test_rounds_up(self) -> None:
        assert estimate_tokens("abc") == 1

    def test_empty_text_is_zero(self) -> None:
        assert estimate_tokens("") == 0


class TestChunkSegments:
    def test_short_input_yields_single_chunk(self) -> None:
        segments = [segment("Hello there.", 0), segment("Short call.", 1)]

        chunks = chunk_segments(segments, max_tokens=DEFAULT_MAX_CHUNK_TOKENS)

        assert chunks == [segments]

    def test_segments_are_never_split_and_order_is_preserved(self) -> None:
        segments = [segment(f"Sentence number {i} of the call.", i) for i in range(50)]

        chunks = chunk_segments(segments, max_tokens=30)

        flattened = [s for chunk in chunks for s in chunk]
        assert flattened == segments
        assert len(chunks) > 1

    def test_chunks_respect_the_token_budget(self) -> None:
        segments = [segment(f"Sentence number {i} of the call.", i) for i in range(50)]

        chunks = chunk_segments(segments, max_tokens=30)

        for chunk in chunks:
            chunk_text = " ".join(s.text for s in chunk)
            assert estimate_tokens(chunk_text) <= 30

    def test_oversized_single_segment_gets_its_own_chunk(self) -> None:
        oversized = segment("word " * 200, 1)  # ~250 tokens on its own
        segments = [segment("Small intro.", 0), oversized, segment("Small outro.", 2)]

        chunks = chunk_segments(segments, max_tokens=50)

        assert [oversized] in chunks
        flattened = [s for chunk in chunks for s in chunk]
        assert flattened == segments

    def test_empty_segment_list_yields_no_chunks(self) -> None:
        assert chunk_segments([], max_tokens=100) == []

    def test_non_positive_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            chunk_segments([segment("hi")], max_tokens=0)

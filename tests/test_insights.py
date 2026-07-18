"""Tests for core.insights.InsightEngine — parse, retry, fallback behavior."""

import json
from types import SimpleNamespace
from typing import Any

from core.chunking import DEFAULT_MAX_CHUNK_TOKENS, chunk_segments
from core.insights import InsightEngine
from core.schemas import Transcript, TranscriptSegment

VALID_BRIEF_JSON = json.dumps(
    {
        "summary": "The supervisor reviewed the draft and asked for revisions.",
        "key_insights": ["The methodology needs stronger justification."],
        "action_items": [
            {
                "task": "Send the revised methodology chapter",
                "owner": "student",
                "priority": "high",
                "suggested_deadline": "Friday",
                "supporting_quote": "send me the revised methodology chapter by Friday",
            }
        ],
    }
)

PARTIAL_BRIEF_JSON = json.dumps({"summary": "A call happened."})  # missing required fields


class FakeChatModel:
    """Returns canned response contents in order; records every invocation."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls.append(list(messages))
        return SimpleNamespace(content=next(self._responses))


class TestFirstAttemptSuccess:
    def test_valid_json_parses_into_brief(self) -> None:
        model = FakeChatModel([VALID_BRIEF_JSON])
        result = InsightEngine(model).generate_brief("transcript text")

        assert result.succeeded
        assert result.brief is not None
        assert result.brief.action_items[0].priority == "high"
        assert len(model.calls) == 1

    def test_code_fenced_json_is_accepted(self) -> None:
        fenced = f"```json\n{VALID_BRIEF_JSON}\n```"
        result = InsightEngine(FakeChatModel([fenced])).generate_brief("transcript")

        assert result.succeeded

    def test_transcript_reaches_the_model(self) -> None:
        model = FakeChatModel([VALID_BRIEF_JSON])
        InsightEngine(model).generate_brief("the quarterly review transcript")

        human_contents = [m.content for m in model.calls[0] if m.type == "human"]
        assert any("the quarterly review transcript" in content for content in human_contents)


class TestRetry:
    def test_malformed_json_triggers_one_repair_retry(self) -> None:
        model = FakeChatModel(["{ not json", VALID_BRIEF_JSON])
        result = InsightEngine(model).generate_brief("transcript")

        assert result.succeeded
        assert len(model.calls) == 2

    def test_repair_prompt_includes_previous_response(self) -> None:
        model = FakeChatModel(["{ not json", VALID_BRIEF_JSON])
        InsightEngine(model).generate_brief("transcript")

        retry_messages = model.calls[1]
        assert any("{ not json" in str(m.content) for m in retry_messages)

    def test_partial_json_also_triggers_retry(self) -> None:
        model = FakeChatModel([PARTIAL_BRIEF_JSON, VALID_BRIEF_JSON])
        result = InsightEngine(model).generate_brief("transcript")

        assert result.succeeded
        assert len(model.calls) == 2


def long_transcript() -> Transcript:
    """A synthetic transcript comfortably over 8k estimated tokens."""
    text = "The supervisor discussed the methodology chapter in great detail today. " * 2
    segments = [
        TranscriptSegment(text=text, start=float(i), end=float(i) + 1.0) for i in range(300)
    ]
    return Transcript(segments=segments)


class TestChunkedFlow:
    def test_short_transcript_object_is_single_pass(self) -> None:
        transcript = Transcript(segments=[TranscriptSegment(text="Short call.", start=0, end=1)])
        model = FakeChatModel([VALID_BRIEF_JSON])

        result = InsightEngine(model).generate_brief(transcript)

        assert result.succeeded
        assert len(model.calls) == 1

    def test_long_transcript_triggers_map_reduce(self) -> None:
        transcript = long_transcript()
        expected_chunks = chunk_segments(transcript.segments, DEFAULT_MAX_CHUNK_TOKENS)
        assert len(expected_chunks) > 1  # sanity: the fixture really exceeds one chunk

        model = FakeChatModel([VALID_BRIEF_JSON] * (len(expected_chunks) + 1))
        result = InsightEngine(model).generate_brief(transcript)

        assert result.succeeded
        assert len(model.calls) == len(expected_chunks) + 1  # one per chunk + synthesis

    def test_failed_chunk_feeds_raw_text_into_synthesis(self) -> None:
        # Two tiny chunks via a small budget; first chunk fails twice, second is fine.
        transcript = Transcript(
            segments=[
                TranscriptSegment(text="first portion " * 10, start=0, end=1),
                TranscriptSegment(text="second portion " * 10, start=1, end=2),
            ]
        )
        model = FakeChatModel(["garbage", "still garbage", VALID_BRIEF_JSON, VALID_BRIEF_JSON])

        engine = InsightEngine(model, max_chunk_tokens=10)
        result = engine.generate_brief(transcript)

        assert result.succeeded
        assert len(model.calls) == 4  # chunk1 + its retry, chunk2, synthesis
        synthesis_input = " ".join(str(m.content) for m in model.calls[-1])
        assert "still garbage" in synthesis_input

    def test_synthesis_failure_falls_back_to_raw_text(self) -> None:
        transcript = Transcript(
            segments=[
                TranscriptSegment(text="first portion " * 10, start=0, end=1),
                TranscriptSegment(text="second portion " * 10, start=1, end=2),
            ]
        )
        model = FakeChatModel([VALID_BRIEF_JSON, VALID_BRIEF_JSON, "bad", "bad again"])

        result = InsightEngine(model, max_chunk_tokens=10).generate_brief(transcript)

        assert not result.succeeded
        assert result.raw_text == "bad again"


class TestFallback:
    def test_two_failures_fall_back_to_raw_text(self) -> None:
        model = FakeChatModel(["{ not json", "still not json"])
        result = InsightEngine(model).generate_brief("transcript")

        assert not result.succeeded
        assert result.brief is None
        assert result.raw_text == "still not json"
        assert len(model.calls) == 2  # exactly one retry, never more

    def test_fallback_never_raises_on_garbage(self) -> None:
        model = FakeChatModel(["", "\x00\x01 garbage"])
        result = InsightEngine(model).generate_brief("transcript")

        assert not result.succeeded

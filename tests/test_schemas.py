"""Tests for core.schemas — the pipeline's Pydantic contracts."""

from typing import Any

import pytest
from pydantic import ValidationError

from core.schemas import (
    ActionItem,
    Brief,
    InsightResult,
    Transcript,
    TranscriptSegment,
    render_segments,
)


def action_item_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "task": "Send the revised methodology chapter",
        "owner": "student",
        "priority": "high",
        "suggested_deadline": "Friday",
        "supporting_quote": "please get the revised chapter to me by Friday",
    }
    data.update(overrides)
    return data


def brief_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "summary": "The supervisor reviewed the draft and requested revisions.",
        "key_insights": ["The methodology section needs stronger justification."],
        "action_items": [action_item_data()],
    }
    data.update(overrides)
    return data


class TestTranscriptSegment:
    def test_valid_segment_parses(self) -> None:
        segment = TranscriptSegment(text="hello there", start=0.0, end=1.5)

        assert segment.text == "hello there"
        assert segment.start == 0.0
        assert segment.end == 1.5

    def test_negative_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptSegment(text="hello", start=-1.0, end=1.0)

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptSegment(text="hello", start=5.0, end=2.0)

    def test_blank_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptSegment(text="   ", start=0.0, end=1.0)


class TestTranscript:
    def test_text_joins_segments_in_order(self) -> None:
        transcript = Transcript(
            segments=[
                TranscriptSegment(text="First part.", start=0.0, end=1.0),
                TranscriptSegment(text="Second part.", start=1.0, end=2.0),
            ]
        )

        assert transcript.text == "First part. Second part."

    def test_empty_transcript_has_empty_text(self) -> None:
        assert Transcript(segments=[]).text == ""


class TestInsightResult:
    def test_succeeded_when_brief_is_present(self) -> None:
        result = InsightResult(brief=Brief.model_validate(brief_data()), raw_text="...")

        assert result.succeeded

    def test_not_succeeded_when_brief_is_missing(self) -> None:
        result = InsightResult(brief=None, raw_text="unparseable output")

        assert not result.succeeded
        assert result.raw_text == "unparseable output"


class TestActionItem:
    def test_parses_from_llm_style_dict(self) -> None:
        item = ActionItem.model_validate(action_item_data())

        assert item.priority == "high"
        assert item.supporting_quote.startswith("please")

    def test_unknown_priority_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActionItem.model_validate(action_item_data(priority="urgent"))

    def test_missing_supporting_quote_is_rejected(self) -> None:
        data = action_item_data()
        del data["supporting_quote"]

        with pytest.raises(ValidationError):
            ActionItem.model_validate(data)

    def test_blank_task_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActionItem.model_validate(action_item_data(task="  "))


class TestBrief:
    def test_full_brief_parses(self) -> None:
        brief = Brief.model_validate(brief_data())

        assert len(brief.key_insights) == 1
        assert brief.action_items[0].owner == "student"

    def test_action_items_may_be_empty(self) -> None:
        brief = Brief.model_validate(brief_data(action_items=[]))

        assert brief.action_items == []

    def test_blank_summary_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Brief.model_validate(brief_data(summary="   "))

    def test_missing_action_items_field_is_rejected(self) -> None:
        data = brief_data()
        del data["action_items"]

        with pytest.raises(ValidationError):
            Brief.model_validate(data)

    def test_extra_llm_fields_are_ignored(self) -> None:
        brief = Brief.model_validate(brief_data(confidence=0.9))

        assert brief.summary.startswith("The supervisor")

def segment(text: str, start: float, end: float, speaker: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end, speaker=speaker)

class TestRenderSegments:
    def test_unlabelled_segments_render_as_plain_joined_text(self) -> None:
        segments = [segment("Hello there.", 0.0, 2.0), segment("How are you?", 2.0, 4.0)]
        assert render_segments(segments) == "Hello there. How are you?"

    def test_labels_each_speaker_turn(self) -> None:
        segments = [
            segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
            segment("It's progressing well", 2.0, 4.0, "Speaker 1"),
            segment("Good.", 4.0, 5.0, "Speaker 2")
        ]
        assert render_segments(segments) == "Speaker 1: How is the chapter? It's progressing well\nSpeaker 2: Good."

    def test_groups_consecutive_segments_from_the_same_speaker(self) -> None:
        segments = [
            segment("Hello", 0.0, 1.0, "Speaker 1"),
            segment("How are you?", 1.0, 2.0, "Speaker 1"),
            segment("I'm fine", 2.0, 3.0, "Speaker 2"),
            segment("Good.", 3.0, 4.0, "Speaker 1")
        ]
        assert render_segments(segments) == "Speaker 1: Hello How are you?\nSpeaker 2: I'm fine\nSpeaker 1: Good."

    def test_unlabelled_segment_among_labelled_ones_is_marked_unknown(self) -> None:
        segments = [
            segment("How is the chapter?", 0.0, 2.0, "Alice"),
            segment("(inaudible)", 2.0, 3.0),
            segment("I'm fine", 3.0, 4.0, "Bob")
        ]
        assert render_segments(segments) == "Alice: How is the chapter?\nUnknown speaker: (inaudible)\nBob: I'm fine"

    def test_empty_segment_list_renders_as_empty_string(self) -> None:
        assert render_segments([]) == ""

class TestTranscriptText:
    def test_delegates_to_render_segments(self) -> None:
        transcript = Transcript(
            segments=[
                segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
                segment("I'm fine", 3.0, 4.0, "Speaker 2")
            ]
        )
        assert transcript.text == "Speaker 1: How is the chapter?\nSpeaker 2: I'm fine"

    def test_speaker_defaults_to_none(self) -> None:
        assert segment("Hello.", 0.0, 1.0).speaker is None

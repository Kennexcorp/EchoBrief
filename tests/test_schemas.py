"""Tests for core.schemas — the pipeline's Pydantic contracts."""

from typing import Any

import pytest
from pydantic import ValidationError

from core.schemas import ActionItem, Brief, Transcript, TranscriptSegment


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

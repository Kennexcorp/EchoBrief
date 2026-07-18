"""Tests for core.prompts — message construction for the insight engine."""

from core.prompts import build_brief_messages, build_repair_messages


class TestBriefMessages:
    def test_system_then_human_message(self) -> None:
        messages = build_brief_messages("We discussed the draft.")

        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[1].type == "human"

    def test_transcript_is_included_in_human_message(self) -> None:
        messages = build_brief_messages("Please send the chapter by Friday.")

        assert "Please send the chapter by Friday." in messages[1].content

    def test_system_prompt_constrains_to_transcript_content(self) -> None:
        messages = build_brief_messages("anything")

        system = messages[0].content.lower()
        assert "transcript" in system
        assert "do not invent" in system or "only" in system

    def test_user_context_is_included_when_given(self) -> None:
        messages = build_brief_messages("anything", user_context="This was my thesis review.")

        assert "This was my thesis review." in messages[1].content

    def test_braces_in_transcript_are_preserved(self) -> None:
        messages = build_brief_messages("code sample: {x: 1}")

        assert "{x: 1}" in messages[1].content


class TestRepairMessages:
    def test_appends_bad_output_and_repair_instruction(self) -> None:
        original = build_brief_messages("We discussed the draft.")

        messages = build_repair_messages(original, "not json {", "invalid JSON")

        assert len(messages) == len(original) + 2
        assert messages[-2].type == "ai"
        assert "not json {" in messages[-2].content
        assert messages[-1].type == "human"
        assert "invalid JSON" in messages[-1].content
        assert "json" in messages[-1].content.lower()

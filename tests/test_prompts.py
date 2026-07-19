"""Tests for core.prompts — message construction for the insight engine."""

from core.prompts import (
    build_brief_messages,
    build_chunk_brief_messages,
    build_repair_messages,
    build_synthesis_messages,
)


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

    def test_prompt_is_domain_neutral(self) -> None:
        # The product serves any recorded call (coaching, client calls, 1:1s) —
        # the prompt must not assume an academic student/supervisor setting.
        messages = build_brief_messages("anything")

        combined = " ".join(m.content for m in messages).lower()
        assert "student" not in combined
        assert "academic" not in combined


class TestChunkBriefMessages:
    def test_includes_chunk_text_and_part_numbering(self) -> None:
        messages = build_chunk_brief_messages("this portion of the call", part=2, total=5)

        assert "this portion of the call" in messages[-1].content
        combined = " ".join(m.content for m in messages)
        assert "2" in combined
        assert "5" in combined

    def test_keeps_the_transcript_only_constraint(self) -> None:
        messages = build_chunk_brief_messages("anything", part=1, total=2)

        assert "do not invent" in messages[0].content.lower()


class TestSynthesisMessages:
    def test_includes_every_part_in_order(self) -> None:
        messages = build_synthesis_messages(["notes for part one", "notes for part two"])

        human = messages[-1].content
        assert "notes for part one" in human
        assert "notes for part two" in human
        assert human.index("notes for part one") < human.index("notes for part two")

    def test_instructs_merging_one_call(self) -> None:
        messages = build_synthesis_messages(["a", "b"])

        system = messages[0].content.lower()
        assert "merge" in system
        assert "one call" in system or "single" in system

    def test_user_context_is_included_when_given(self) -> None:
        messages = build_synthesis_messages(["a"], user_context="My thesis review.")

        assert "My thesis review." in messages[-1].content


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

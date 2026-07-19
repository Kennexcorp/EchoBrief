"""Tests for core.export — Markdown export of a brief (F5)."""

from core.export import export_markdown
from core.schemas import ActionItem, Brief, InsightResult, Transcript, TranscriptSegment

TRANSCRIPT = Transcript(
    segments=[
        TranscriptSegment(text="We reviewed the draft.", start=0.0, end=2.0),
        TranscriptSegment(text="Send the chapter by Friday.", start=2.0, end=4.0),
    ]
)

BRIEF = Brief(
    summary="The supervisor reviewed the draft and requested revisions.",
    key_insights=["The methodology needs work.", "Results are strong."],
    action_items=[
        ActionItem(
            task="Send the revised chapter",
            owner="student",
            priority="high",
            suggested_deadline="Friday",
            supporting_quote="Send the chapter by Friday.",
        )
    ],
)


class TestStructuredExport:
    def test_contains_all_brief_sections_in_order(self) -> None:
        markdown = export_markdown(InsightResult(brief=BRIEF), TRANSCRIPT)

        assert "## Summary" in markdown
        assert "## Key Insights" in markdown
        assert "## Action Items" in markdown
        assert "## Full Transcript" in markdown
        assert markdown.index("## Summary") < markdown.index("## Full Transcript")

    def test_action_item_details_are_rendered(self) -> None:
        markdown = export_markdown(InsightResult(brief=BRIEF), TRANSCRIPT)

        assert "1. **Send the revised chapter**" in markdown
        assert "high" in markdown
        assert "student" in markdown
        assert "Friday" in markdown
        assert '"Send the chapter by Friday."' in markdown

    def test_insights_become_bullets(self) -> None:
        markdown = export_markdown(InsightResult(brief=BRIEF), TRANSCRIPT)

        assert "- The methodology needs work." in markdown
        assert "- Results are strong." in markdown

    def test_transcript_text_is_included(self) -> None:
        markdown = export_markdown(InsightResult(brief=BRIEF), TRANSCRIPT)

        assert "We reviewed the draft. Send the chapter by Friday." in markdown

    def test_no_action_items_renders_placeholder(self) -> None:
        brief = BRIEF.model_copy(update={"action_items": []})

        markdown = export_markdown(InsightResult(brief=brief), TRANSCRIPT)

        assert "No action items" in markdown


class TestFallbackExport:
    def test_raw_text_and_warning_are_rendered(self) -> None:
        result = InsightResult(brief=None, raw_text="the model said something unparseable")

        markdown = export_markdown(result, TRANSCRIPT)

        assert "the model said something unparseable" in markdown
        assert "parsing failed" in markdown.lower()
        assert "## Full Transcript" in markdown

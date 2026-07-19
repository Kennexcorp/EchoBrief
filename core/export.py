"""Markdown export of a processed call (F5).

Works for both outcomes: a structured ``Brief``, or the raw-text fallback when
parsing failed — the user always gets a usable document.
"""

from __future__ import annotations

from core.schemas import Brief, InsightResult, Transcript


def export_markdown(result: InsightResult, transcript: Transcript) -> str:
    lines = ["# EchoBrief — Call Brief", ""]
    if result.brief is not None:
        lines.extend(_brief_sections(result.brief))
    else:
        lines.extend(
            [
                "> ⚠️ Structured parsing failed — showing the model's raw output.",
                "",
                result.raw_text,
            ]
        )
    lines.extend(["", "## Full Transcript", "", transcript.text, ""])
    return "\n".join(lines)


def _brief_sections(brief: Brief) -> list[str]:
    lines = ["## Summary", "", brief.summary, "", "## Key Insights", ""]
    lines.extend(f"- {insight}" for insight in brief.key_insights)
    lines.extend(["", "## Action Items", ""])
    if not brief.action_items:
        lines.append("_No action items were discussed._")
    for number, item in enumerate(brief.action_items, start=1):
        lines.extend(
            [
                f"{number}. **{item.task}**",
                f"   - Priority: {item.priority} · Owner: {item.owner} · "
                f"Deadline: {item.suggested_deadline}",
                f'   - Supporting quote: "{item.supporting_quote}"',
            ]
        )
    return lines

"""Prompt construction for the insight engine (the F3/F4 contract).

Templates live here — versioned in git, tested, and never built with ad-hoc
f-strings elsewhere. Transcript text is passed as a template *value*, so any
braces it contains are never re-interpreted as template variables.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an academic meeting analyst. You turn transcripts of supervision and \
mentorship calls into a structured brief for the student who was in the call.

Respond with a single JSON object, and nothing else, matching this schema:
- "summary": a 3-5 sentence summary of the call.
- "key_insights": an array of strings — the key insights and feedback the student received.
- "action_items": an array of objects, each with:
    - "task": the action to take, as one imperative sentence
    - "owner": who is responsible (e.g. "student", "supervisor")
    - "priority": exactly one of "high", "medium", "low"
    - "suggested_deadline": the deadline or timeframe mentioned or implied in the call
    - "supporting_quote": a verbatim quote from the transcript this action item is based on

Rules:
- Use ONLY information present in the transcript. Do not invent tasks, \
deadlines, or feedback that were not discussed.
- Every action item MUST include its supporting verbatim quote from the transcript.
- If no action items were discussed, return an empty "action_items" array.
- Keep a neutral, factual tone."""

_USER = """\
Additional context from the student: {user_context}

Transcript of the call:
<transcript>
{transcript}
</transcript>"""

BRIEF_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _USER)])

_REPAIR_INSTRUCTION = """\
Your previous response could not be parsed into the required structure. \
Error: {error}

Respond again with ONLY a single valid JSON object matching the schema from the \
system message — no prose, no code fences, no explanations."""


def build_brief_messages(transcript: str, user_context: str | None = None) -> list[BaseMessage]:
    """Build the system + human message pair asking for a structured brief."""
    context = user_context.strip() if user_context and user_context.strip() else "None provided."
    return BRIEF_PROMPT.format_messages(transcript=transcript, user_context=context)


def build_repair_messages(
    original_messages: list[BaseMessage], bad_response: str, error: str
) -> list[BaseMessage]:
    """Extend the original exchange with the failed response and a repair request."""
    return [
        *original_messages,
        AIMessage(content=bad_response),
        HumanMessage(content=_REPAIR_INSTRUCTION.format(error=error)),
    ]

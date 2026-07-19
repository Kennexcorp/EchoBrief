"""Prompt construction for the insight engine (the F3/F4 contract).

Templates live here — versioned in git, tested, and never built with ad-hoc
f-strings elsewhere. Transcript text is passed as a template *value*, so any
braces it contains are never re-interpreted as template variables.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are a meeting analyst. You turn transcripts of recorded calls — supervision \
meetings, coaching sessions, client check-ins, one-on-ones — into a structured \
brief for the participant who requested it.

Respond with a single JSON object, and nothing else, matching this schema:
- "summary": a 3-5 sentence summary of the call.
- "key_insights": an array of strings — the key insights and feedback the participant received.
- "action_items": an array of objects, each with:
    - "task": the action to take, as one imperative sentence
    - "owner": who is responsible, using the roles heard in the call \
(e.g. "me", "supervisor", "coach", "client")
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
Additional context from the participant: {user_context}

Transcript of the call:
<transcript>
{transcript}
</transcript>"""

BRIEF_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _USER)])

_CHUNK_SYSTEM = (
    _SYSTEM
    + """

You are seeing part {part} of {total} of the transcript. Produce the brief for \
just this portion; it will later be merged with the other parts."""
)

_CHUNK_USER = """\
Part {part} of {total} of the transcript:
<transcript>
{transcript}
</transcript>"""

CHUNK_BRIEF_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CHUNK_SYSTEM), ("human", _CHUNK_USER)]
)

_SYNTHESIS_SYSTEM = (
    _SYSTEM
    + """

You are given structured notes extracted from consecutive portions of one call. \
Merge them into a single brief covering the whole call: deduplicate overlapping \
insights and action items, and keep every supporting_quote verbatim as given."""
)

_SYNTHESIS_USER = """\
Additional context from the participant: {user_context}

Notes from the portions of the call, in order:

{parts}"""

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYNTHESIS_SYSTEM), ("human", _SYNTHESIS_USER)]
)

_REPAIR_INSTRUCTION = """\
Your previous response could not be parsed into the required structure. \
Error: {error}

Respond again with ONLY a single valid JSON object matching the schema from the \
system message — no prose, no code fences, no explanations."""


def build_brief_messages(transcript: str, user_context: str | None = None) -> list[BaseMessage]:
    """Build the system + human message pair asking for a structured brief."""
    context = user_context.strip() if user_context and user_context.strip() else "None provided."
    return BRIEF_PROMPT.format_messages(transcript=transcript, user_context=context)


def build_chunk_brief_messages(chunk_text: str, part: int, total: int) -> list[BaseMessage]:
    """Build the map-step messages: a brief for one portion of a long call."""
    return CHUNK_BRIEF_PROMPT.format_messages(transcript=chunk_text, part=part, total=total)


def build_synthesis_messages(
    part_notes: list[str], user_context: str | None = None
) -> list[BaseMessage]:
    """Build the reduce-step messages: merge per-portion notes into one brief."""
    context = user_context.strip() if user_context and user_context.strip() else "None provided."
    parts = "\n\n".join(
        f"--- Part {i} of {len(part_notes)} ---\n{notes}"
        for i, notes in enumerate(part_notes, start=1)
    )
    return SYNTHESIS_PROMPT.format_messages(parts=parts, user_context=context)


def build_repair_messages(
    original_messages: list[BaseMessage], bad_response: str, error: str
) -> list[BaseMessage]:
    """Extend the original exchange with the failed response and a repair request."""
    return [
        *original_messages,
        AIMessage(content=bad_response),
        HumanMessage(content=_REPAIR_INSTRUCTION.format(error=error)),
    ]

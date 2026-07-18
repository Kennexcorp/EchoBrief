"""Pydantic contracts shared across the pipeline.

``Brief`` is the single output contract between the insight layer and the UI
(F4). Field descriptions double as the JSON schema the LLM is asked to fill,
so keep them written for the model, not just for readers.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Priority = Literal["high", "medium", "low"]


class TranscriptSegment(BaseModel):
    """One Whisper segment with timings in seconds from the start of the audio."""

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class ActionItem(BaseModel):
    """A single follow-up task extracted from the call."""

    task: str = Field(description="The action to take, phrased as one imperative sentence.")
    owner: str = Field(description="Who is responsible, e.g. 'student' or 'supervisor'.")
    priority: Priority = Field(description="Urgency of the task: high, medium, or low.")
    suggested_deadline: str = Field(
        description="Deadline or timeframe as mentioned or implied in the call, e.g. 'Friday'."
    )
    supporting_quote: str = Field(
        description="Verbatim quote from the transcript that this action item is based on."
    )

    @field_validator("task", "supporting_quote")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Brief(BaseModel):
    """The structured brief produced from one call recording."""

    summary: str = Field(description="A 3-5 sentence summary of the call.")
    key_insights: list[str] = Field(
        description="Key insights and feedback the participant received during the call."
    )
    action_items: list[ActionItem] = Field(
        description="Follow-up tasks discussed in the call. Empty if none were discussed."
    )

    @field_validator("summary")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

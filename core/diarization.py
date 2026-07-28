"""Speaker diarization (F9): who spoke when, merged onto Whisper segments.

pyannote runs alongside faster-whisper rather than replacing it, so the two
outputs are joined here by timestamp overlap. The merge is a pure function,
which keeps the interesting logic testable without audio, a GPU, or a token.
"""

from core.schemas import TranscriptSegment
from typing import NamedTuple
class SpeakerTurn(NamedTuple):
    """One span of speech attributed to a speaker, in seconds from audio start."""

    start: float
    end: float
    speaker: str

def _overlap(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    """Return the duration of overlap between a segment and a turn in seconds"""
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))

def _friendly_labels(turns: list[SpeakerTurn]) -> dict[str,str]:
    """Map raw pynnote ids to Speaker 1, Speaker 2 etc."""
    labels: dict[str,str] = {}
    for turn in sorted(turns, key=lambda item: item.start):
        if turn.speaker not in labels:
            labels[turn.speaker] = f"Speaker {len(labels) + 1}"
    return labels

def assign_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    """For each segment, attach the speaker label of the turn it overlaps most. Segments with no overlap get speaker None."""
    if not turns:
        return list(segments)

    labels = _friendly_labels(turns)
    ordered_turns = sorted(turns, key=lambda item: item.start)

    labelled: list[TranscriptSegment] = []
    for segment in segments:
        best_turn: SpeakerTurn | None = None
        best_overlap = 0.0

        for turn in ordered_turns:
            
            overlap = _overlap(segment, turn)
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        speaker = labels[best_turn.speaker] if best_turn is not None else None
        labelled.append(segment.model_copy(update={"speaker": speaker}))
        
    return labelled

    
# tests/test_diarization.py
"""Tests for core.diarization."""

from core.diarization import SpeakerTurn, assign_speakers
from core.schemas import TranscriptSegment


def segment(start: float, end: float, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end)


class TestAssignSpeakers:
    def test_assigns_the_speaker_whose_turn_fully_contains_the_segment(self) -> None:
        segments = [segment(1.0, 2.0)]
        turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
        assert assign_speakers(segments, turns)[0].speaker == "Speaker 1"

    def test_assigns_the_turn_with_the_greatest_overlap(self) -> None:
        # Segment 0-10 overlaps SPEAKER_00 by 3s and SPEAKER_01 by 7s.
        segments = [segment(0.0, 10.0)]
        turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 10.0, "SPEAKER_01")]
        assert assign_speakers(segments, turns)[0].speaker == "Speaker 2"

    def test_ties_resolve_to_the_earlier_starting_turn(self) -> None:
        # Exactly 2s of overlap with each.
        segments = [segment(2.0, 6.0)]
        turns = [SpeakerTurn(0.0, 4.0, "SPEAKER_00"), SpeakerTurn(4.0, 8.0, "SPEAKER_01")]
        assert assign_speakers(segments, turns)[0].speaker == "Speaker 1"

    def test_segment_overlapping_no_turn_keeps_speaker_none(self) -> None:
        segments = [segment(20.0, 25.0)]
        turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
        assert assign_speakers(segments, turns)[0].speaker is None

    def test_touching_boundaries_do_not_count_as_overlap(self) -> None:
        segments = [segment(5.0, 10.0)]
        turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
        assert assign_speakers(segments, turns)[0].speaker is None

    def test_empty_turns_returns_segments_unlabelled(self) -> None:
        segments = [segment(0.0, 1.0), segment(1.0, 2.0)]
        result = assign_speakers(segments, [])
        assert [item.speaker for item in result] == [None, None]

    def test_empty_segments_returns_empty_list(self) -> None:
        assert assign_speakers([], [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]) == []

    def test_labels_are_numbered_by_first_appearance_not_by_raw_id(self) -> None:
        # SPEAKER_07 speaks first, so it must become "Speaker 1".
        segments = [segment(0.0, 1.0), segment(2.0, 3.0)]
        turns = [SpeakerTurn(2.0, 3.0, "SPEAKER_02"), SpeakerTurn(0.0, 1.0, "SPEAKER_07")]
        result = assign_speakers(segments, turns)
        assert [item.speaker for item in result] == ["Speaker 1", "Speaker 2"]

    def test_does_not_mutate_the_input_segments(self) -> None:
        segments = [segment(1.0, 2.0)]
        assign_speakers(segments, [SpeakerTurn(0.0, 5.0, "SPEAKER_00")])
        assert segments[0].speaker is None

    def test_preserves_text_and_timings(self) -> None:
        segments = [segment(1.0, 2.0, text="Send the chapter by Friday.")]
        result = assign_speakers(segments, [SpeakerTurn(0.0, 5.0, "SPEAKER_00")])
        assert result[0].text == "Send the chapter by Friday."
        assert (result[0].start, result[0].end) == (1.0, 2.0)
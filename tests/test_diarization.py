# tests/test_diarization.py
"""Tests for core.diarization."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from core.diarization import apply_diarization
from core.schemas import Transcript
from core.config import Settings
from core.diarization import (
    DiarizationError,
    DiarizationService,
    SpeakerTurn,
    assign_speakers,
    create_diarization_service,
)
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


class FakeAnnotation:
    """Mimics pyannote.core.Annotation for testing"""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Any:
        for start, end, speaker in self.tracks:
            yield SimpleNamespace(start=start, end=end), "_", speaker


class FakeOutput:
    """Mimics the community-1 result objects: diarization is one field of it"""

    def __init__(self, annotation: FakeAnnotation) -> None:
        self.speaker_diarization = annotation


class FakePipeline:
    """Stands in for a loaded pyannote Pipeline: callable, returns a result object."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks
        self.calls: list[str] = []

    def __call__(self, audio: str) -> FakeOutput:
        self.calls.append(audio)
        return FakeOutput(FakeAnnotation(self._tracks))


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "call.mp3"
    path.write_bytes(b"fake audio bytes")
    return path


class TestDiarizationService:
    def test_maps_annotation_tracks_to_speaker_turns(self, audio_file: Path) -> None:
        pipeline = FakePipeline([(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")])
        turns = DiarizationService(pipeline).diarize(audio_file)
        assert turns == [
            SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
            SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
        ]

    def test_passes_the_audio_path_to_the_pipeline(self, audio_file: Path) -> None:
        pipeline = FakePipeline([])
        DiarizationService(pipeline).diarize(audio_file)
        assert pipeline.calls == [str(audio_file)]

    def test_missing_audio_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DiarizationService(FakePipeline([])).diarize(tmp_path / "nope.mp3")


class TestCreateDiarizationService:
    def test_without_a_token_raises_with_the_env_var_and_model_url(self) -> None:
        with pytest.raises(DiarizationError) as exc_info:
            create_diarization_service(Settings())
        message = str(exc_info.value)
        assert "HUGGINGFACE_TOKEN" in message
        assert "pyannote/speaker-diarization-community-1" in message


def transcript_of(*segments: TranscriptSegment) -> Transcript:
    return Transcript(segments=list(segments))


class TestApplyDiarization:
    def test_labels_the_transcript_when_everything_works(self, audio_file: Path) -> None:
        pipeline = FakePipeline([(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")])
        transcript = transcript_of(segment(0.0, 2.0, "Hi."), segment(2.0, 4.0, "Hello."))

        labelled, warning = apply_diarization(
            transcript,
            audio_file,
            Settings(huggingface_token="hf_abc"),
            service_factory=lambda _settings: DiarizationService(pipeline),
        )

        assert warning is None
        assert [item.speaker for item in labelled.segments] == ["Speaker 1", "Speaker 2"]

    def test_returns_transcript_unchanged_with_a_warning_when_unavailable(
        self, audio_file: Path
    ) -> None:
        transcript = transcript_of(segment(0.0, 2.0, "Hi."))

        def unavailable(_settings: Settings) -> DiarizationService:
            raise DiarizationError("Install it with: uv sync --extra diarization")

        result, warning = apply_diarization(
            transcript, audio_file, Settings(), service_factory=unavailable
        )

        assert result is transcript
        assert warning is not None
        assert "uv sync --extra diarization" in warning

    def test_unexpected_failure_degrades_instead_of_raising(self, audio_file: Path) -> None:
        transcript = transcript_of(segment(0.0, 2.0, "Hi."))

        def exploding(_settings: Settings) -> DiarizationService:
            raise RuntimeError("CUDA out of memory")

        result, warning = apply_diarization(
            transcript, audio_file, Settings(huggingface_token="hf_abc"),
            service_factory=exploding,
        )

        assert result is transcript
        assert warning is not None
        assert "CUDA out of memory" in warning
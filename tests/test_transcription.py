"""Tests for core.transcription.TranscriptionService."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.config import Settings
from core.transcription import TranscriptionService, create_transcription_service

FIXTURE_AUDIO = Path(__file__).parent / "fixtures" / "sample_call.mp3"


def whisper_segment(text: str, start: float = 0.0, end: float = 1.0) -> SimpleNamespace:
    """Mimics the attributes of a faster-whisper Segment."""
    return SimpleNamespace(text=text, start=start, end=end)


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel — records calls, yields canned segments."""

    def __init__(self, segments: list[SimpleNamespace]) -> None:
        self._segments = segments
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: str, **options: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": audio, **options})
        info = SimpleNamespace(language="en", duration=26.0)
        return (segment for segment in self._segments), info


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "call.mp3"
    path.write_bytes(b"fake audio bytes")
    return path


class TestTranscriptionService:
    def test_maps_whisper_segments_to_transcript(self, audio_file: Path) -> None:
        model = FakeWhisperModel(
            [
                whisper_segment(" Hello there.", 0.0, 2.0),
                whisper_segment(" Send the chapter by Friday.", 2.0, 5.5),
            ]
        )
        service = TranscriptionService(model)

        transcript = service.transcribe(audio_file)

        assert len(transcript.segments) == 2
        assert transcript.segments[0].text == "Hello there."
        assert transcript.segments[1].start == 2.0
        assert transcript.segments[1].end == 5.5

    def test_full_text_joins_segments_in_order(self, audio_file: Path) -> None:
        model = FakeWhisperModel(
            [whisper_segment(" One."), whisper_segment(" Two."), whisper_segment(" Three.")]
        )

        transcript = TranscriptionService(model).transcribe(audio_file)

        assert transcript.text == "One. Two. Three."

    def test_blank_segments_are_dropped(self, audio_file: Path) -> None:
        model = FakeWhisperModel([whisper_segment("  "), whisper_segment(" Real content.")])

        transcript = TranscriptionService(model).transcribe(audio_file)

        assert [segment.text for segment in transcript.segments] == ["Real content."]

    def test_vad_filter_is_enabled(self, audio_file: Path) -> None:
        model = FakeWhisperModel([whisper_segment("Hi.")])

        TranscriptionService(model).transcribe(audio_file)

        assert model.calls[0]["vad_filter"] is True

    def test_missing_file_raises_without_calling_model(self, tmp_path: Path) -> None:
        model = FakeWhisperModel([whisper_segment("Hi.")])
        missing = tmp_path / "nope.mp3"

        with pytest.raises(FileNotFoundError, match="nope.mp3"):
            TranscriptionService(model).transcribe(missing)

        assert model.calls == []


@pytest.mark.integration
def test_real_faster_whisper_transcribes_bundled_clip() -> None:
    """End-to-end: factory + real tiny model against the bundled sample call."""
    service = create_transcription_service(Settings(whisper_model_size="tiny"))

    transcript = service.transcribe(FIXTURE_AUDIO)

    text = transcript.text.lower()
    assert transcript.segments
    assert "chapter" in text
    assert "friday" in text

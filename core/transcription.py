"""Local speech-to-text: a thin, testable wrapper around faster-whisper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from faster_whisper import WhisperModel

from core.config import Settings
from core.schemas import Transcript, TranscriptSegment


class SpeechToTextModel(Protocol):
    """The single method this service needs from a Whisper implementation."""

    def transcribe(self, audio: str, **options: Any) -> tuple[Iterable[Any], Any]: ...


class TranscriptionService:
    """Turns an audio file into a ``Transcript`` using an injected Whisper model."""

    def __init__(self, model: SpeechToTextModel) -> None:
        self._model = model

    def transcribe(self, audio_path: Path | str) -> Transcript:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        raw_segments, _info = self._model.transcribe(str(path), vad_filter=True)
        segments = [
            TranscriptSegment(text=text, start=segment.start, end=segment.end)
            for segment in raw_segments
            if (text := segment.text.strip())
        ]
        return Transcript(segments=segments)


def create_transcription_service(settings: Settings) -> TranscriptionService:
    """Build a service backed by a real faster-whisper model from settings."""
    model = WhisperModel(
        settings.whisper_model_size,
        device="auto",
        compute_type=settings.whisper_compute_type,
    )
    return TranscriptionService(model)

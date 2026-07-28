"""Speaker diarization (F9): who spoke when, merged onto Whisper segments.

pyannote runs alongside faster-whisper rather than replacing it, so the two
outputs are joined here by timestamp overlap. The merge is a pure function,
which keeps the interesting logic testable without audio, a GPU, or a token.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from core.config import Settings
from core.schemas import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class SpeakerTurn(NamedTuple):
    """One span of speech attributed to a speaker, in seconds from audio start."""

    start: float
    end: float
    speaker: str


def _overlap(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    """Return the duration of overlap between a segment and a turn in seconds"""
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))


def _friendly_labels(turns: list[SpeakerTurn]) -> dict[str, str]:
    """Map raw pynnote ids to Speaker 1, Speaker 2 etc."""
    labels: dict[str, str] = {}
    for turn in sorted(turns, key=lambda item: item.start):
        if turn.speaker not in labels:
            labels[turn.speaker] = f"Speaker {len(labels) + 1}"
    return labels


def assign_speakers(
    segments: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[TranscriptSegment]:
    """Attach to each segment the label of the turn it overlaps most.

    Ties go to the earlier-starting turn, and a segment overlapping no turn
    keeps ``speaker=None``.
    """
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


MODEL_ID = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    """Raised when diarization cannot run. Message states the exact fix."""


class DiarizationModel(Protocol):
    """The single capability this service needs: audio in, result object out."""

    def __call__(self, audio: dict[str, Any]) -> Any: ...


def _decode_audio(path: Path) -> dict[str, Any]:
    """Decode a whole audio file into the in-memory form pyannote accepts.

    Handing pyannote a path makes it seek within the file for each analysis
    window, and a seeked read of an MP3 returns fewer samples than requested
    because the encoder-delay priming is lost. The pipeline asserts an exact
    sample count, so every MP3 fails. Decoding once up front avoids all seeking.

    torchcodec ships with pyannote.audio 4.x, so it is imported lazily here for
    the same reason the pipeline is: this module must stay importable without
    the optional extra.
    """
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(str(path)).get_all_samples()
    return {"waveform": samples.data, "sample_rate": samples.sample_rate}


class DiarizationService:
    """Turns an audio file into speaker turns using an injected pyannote pipeline."""

    def __init__(
        self,
        model: DiarizationModel,
        loader: Callable[[Path], dict[str, Any]] = _decode_audio,
    ) -> None:
        self._model = model
        self._load = loader

    def diarize(self, audio_path: Path | str) -> list[SpeakerTurn]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # community-1 returns a result object; the diarization is one field of it.
        annotation = self._model(self._load(path)).speaker_diarization
        return [
            SpeakerTurn(start=float(span.start), end=float(span.end), speaker=str(speaker))
            for span, _track, speaker in annotation.itertracks(yield_label=True)
        ]


def create_diarization_service(settings: Settings) -> DiarizationService:
    """Build a service backed by a real pyannote pipeline; raises if unavailable.

    pyannote is imported here rather than at module scope so this module stays
    importable when the optional extra is not installed. That is what lets the
    pipeline degrade to an unlabelled transcript instead of failing to start.
    """
    if not settings.diarization_enabled:
        raise DiarizationError(
            "Speaker diarization is off. Set HUGGINGFACE_TOKEN in your .env, and accept "
            f"the model terms at https://hf.co/{MODEL_ID} to enable it."
        )

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "Speaker diarization needs the optional extra, which is not installed. "
            "Install it with: uv sync --extra diarization"
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(MODEL_ID, token=settings.huggingface_token)
    except Exception as exc:
        raise DiarizationError(
            f"Could not load {MODEL_ID}: {exc}. Check that HUGGINGFACE_TOKEN is valid and "
            f"that you have accepted the model terms at https://hf.co/{MODEL_ID}."
        ) from exc

    return DiarizationService(pipeline)


def apply_diarization(
    transcript: Transcript,
    audio_path: Path | str,
    settings: Settings,
    *,
    service_factory: Callable[[Settings], DiarizationService] = create_diarization_service,
) -> tuple[Transcript, str | None]:
    """Label a transcript by speaker, degrading to the original on any failure.

    Returns the transcript (labelled when diarization succeeded, untouched
    otherwise) and a warning message to surface when it did not. Never raises:
    diarization is an enhancement, never a dependency of the main path.
    """
    try:
        turns = service_factory(settings).diarize(audio_path)
    except DiarizationError as exc:
        return transcript, str(exc)
    except Exception as exc:  # noqa: BLE001 - any pyannote failure must degrade, not crash
        # The UI only has room for one line, so keep the traceback in the log:
        # degrading gracefully must not mean destroying the evidence.
        logger.exception("Diarization failed for %s", audio_path)
        return transcript, f"Speaker diarization failed, continuing without labels: {exc}"

    return Transcript(segments=assign_speakers(transcript.segments, turns)), None

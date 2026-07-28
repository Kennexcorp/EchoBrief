# Speaker Diarization (F9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label each transcript segment with its speaker and feed those labels to the LLM, so action-item ownership is grounded in the audio rather than inferred from phrasing.

**Architecture:** pyannote runs alongside faster-whisper (not WhisperX), and the two outputs are merged by timestamp overlap in a pure function. The transcription layer is untouched. Diarization is an optional extra that degrades gracefully to today's exact behavior when unavailable.

**Tech Stack:** Python 3.10+, uv, pyannote.audio (optional extra), Pydantic, pytest, ruff.

**Spec:** [specs/2026-07-27-speaker-diarization.md](specs/2026-07-27-speaker-diarization.md)

## Global Constraints

- **Ask for approval before every `git commit`.** Stage the changes, show what would be committed, and wait. This holds even though each task below ends with a commit step.
- Never add a `Co-Authored-By` trailer to any commit.
- Strict TDD: write the failing test first, watch it fail for the right reason, then write the minimum code to pass. Test and implementation land in the same commit.
- `core/` never imports Streamlit. No business logic in `app/`.
- Unit tests never touch the network, a real Ollama daemon, real Whisper weights, or real pyannote models.
- All public functions and methods carry type hints.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- Every task ends green: `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
- Coverage on `core/` stays >= 80%: `uv run pytest --cov=core --cov-fail-under=80`.
- Model ID is the constant `"pyannote/speaker-diarization-community-1"`. Never an env var.
- pyannote is imported **only** inside `create_diarization_service`, never at module scope.

---

### Task 1: Speaker-aware transcript rendering

Foundation. Adds the `speaker` field and the single renderer that every downstream consumer already flows through. No pyannote involved, so this is fully testable on its own.

**Files:**

- Modify: `core/schemas.py` (add field to `TranscriptSegment:15`, add `render_segments`, change `Transcript.text:41-43`)
- Test: `tests/test_schemas.py`

**Interfaces:**

- Consumes: nothing
- Produces: `TranscriptSegment.speaker: str | None`, `render_segments(segments: list[TranscriptSegment]) -> str`, `Transcript.text` delegating to it

- [X] **Step 1: Write the failing tests**

```python
# tests/test_schemas.py
from core.schemas import Transcript, TranscriptSegment, render_segments


def segment(text: str, start: float, end: float, speaker: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end, speaker=speaker)


class TestRenderSegments:
    def test_unlabelled_segments_render_as_plain_joined_text(self) -> None:
        segments = [segment("Hello there.", 0.0, 2.0), segment("How are you?", 2.0, 4.0)]
        assert render_segments(segments) == "Hello there. How are you?"

    def test_labels_each_speaker_turn(self) -> None:
        segments = [
            segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
            segment("Nearly done.", 2.0, 4.0, "Speaker 2"),
        ]
        assert render_segments(segments) == (
            "Speaker 1: How is the chapter?\nSpeaker 2: Nearly done."
        )

    def test_groups_consecutive_segments_from_the_same_speaker(self) -> None:
        segments = [
            segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
            segment("Any blockers?", 2.0, 4.0, "Speaker 1"),
            segment("Nearly done.", 4.0, 6.0, "Speaker 2"),
        ]
        assert render_segments(segments) == (
            "Speaker 1: How is the chapter? Any blockers?\nSpeaker 2: Nearly done."
        )

    def test_unlabelled_segment_among_labelled_ones_is_marked_unknown(self) -> None:
        segments = [
            segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
            segment("(inaudible)", 2.0, 3.0, None),
        ]
        assert render_segments(segments) == (
            "Speaker 1: How is the chapter?\nUnknown speaker: (inaudible)"
        )

    def test_empty_segment_list_renders_as_empty_string(self) -> None:
        assert render_segments([]) == ""


class TestTranscriptText:
    def test_delegates_to_render_segments(self) -> None:
        transcript = Transcript(
            segments=[
                segment("How is the chapter?", 0.0, 2.0, "Speaker 1"),
                segment("Nearly done.", 2.0, 4.0, "Speaker 2"),
            ]
        )
        assert transcript.text == "Speaker 1: How is the chapter?\nSpeaker 2: Nearly done."

    def test_speaker_defaults_to_none(self) -> None:
        assert segment("Hello.", 0.0, 1.0).speaker is None
```

- [X] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_segments'`

- [X] **Step 3: Write the implementation**

In `core/schemas.py`, add `speaker` to `TranscriptSegment`:

```python
class TranscriptSegment(BaseModel):
    """One Whisper segment with timings in seconds from the start of the audio."""

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str | None = None
```

Add the renderer below the `TranscriptSegment` class and above `Transcript`:

```python
def render_segments(segments: list[TranscriptSegment]) -> str:
    """Join segments into text, prefixing each speaker turn when labels are present.

    Consecutive segments from one speaker are merged into a single turn, so the
    label appears once per turn rather than once per segment. With no labels at
    all the output is the plain space-joined text, keeping behaviour identical
    for transcripts that were never diarized.
    """
    if not any(segment.speaker for segment in segments):
        return " ".join(segment.text for segment in segments)

    turns: list[str] = []
    current_speaker: str | None = None
    for segment in segments:
        speaker = segment.speaker or "Unknown speaker"
        if speaker == current_speaker:
            turns[-1] += f" {segment.text}"
        else:
            turns.append(f"{speaker}: {segment.text}")
            current_speaker = speaker
    return "\n".join(turns)
```

Replace the `Transcript.text` property:

```python
    @property
    def text(self) -> str:
        return render_segments(self.segments)
```

- [X] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. The existing `export.py` and `insights.py` tests must still pass untouched, which proves the no-labels path is unchanged.

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/schemas.py tests/test_schemas.py
git commit -m "feat: add speaker field and speaker-aware transcript rendering"
```

---

### Task 2: Config for diarization

**Files:**

- Modify: `core/config.py` (add to `_ENV_FIELDS:15`, add field, add property)
- Test: `tests/test_config.py`

**Interfaces:**

- Consumes: nothing
- Produces: `Settings.huggingface_token: str`, `Settings.diarization_enabled: bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
class TestDiarizationSettings:
    def test_token_defaults_to_empty_and_feature_is_off(self) -> None:
        settings = Settings.from_env({})
        assert settings.huggingface_token == ""
        assert settings.diarization_enabled is False

    def test_token_from_env_enables_the_feature(self) -> None:
        settings = Settings.from_env({"HUGGINGFACE_TOKEN": "hf_abc123"})
        assert settings.huggingface_token == "hf_abc123"
        assert settings.diarization_enabled is True

    def test_whitespace_only_token_is_stripped_and_leaves_feature_off(self) -> None:
        settings = Settings.from_env({"HUGGINGFACE_TOKEN": "   "})
        assert settings.diarization_enabled is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'huggingface_token'`

- [ ] **Step 3: Write the implementation**

In `core/config.py`, add to `_ENV_FIELDS`:

```python
    "HUGGINGFACE_TOKEN": "huggingface_token",
```

Add the field next to the ElevenLabs block:

```python
    # Optional speaker diarization (F9). Off unless a Hugging Face token is set;
    # pyannote also needs the optional extra: uv sync --extra diarization
    huggingface_token: str = ""
```

Add `huggingface_token` to the existing `_strip` validator's field list, and add the property beside `tts_enabled`:

```python
    @property
    def diarization_enabled(self) -> bool:
        """Diarization is configured only when a Hugging Face token is present.

        This reports configuration intent, not runnability: the optional extra
        must also be installed, which is detected at the pyannote import site.
        """
        return bool(self.huggingface_token)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/config.py tests/test_config.py
git commit -m "feat: add HUGGINGFACE_TOKEN setting for diarization"
```

---

### Task 3: Overlap assignment (the pure logic)

The heart of the feature. No I/O, no pyannote, no audio. If anything in this feature has a bug, it will be here, so this task carries the densest tests.

**Files:**

- Create: `core/diarization.py`
- Test: `tests/test_diarization.py`

**Interfaces:**

- Consumes: `TranscriptSegment` from Task 1
- Produces: `SpeakerTurn(start: float, end: float, speaker: str)` NamedTuple, `assign_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[TranscriptSegment]`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.diarization'`

- [ ] **Step 3: Write the implementation**

```python
# core/diarization.py
"""Speaker diarization (F9): who spoke when, merged onto Whisper segments.

pyannote runs alongside faster-whisper rather than replacing it, so the two
outputs are joined here by timestamp overlap. The merge is a pure function,
which keeps the interesting logic testable without audio, a GPU, or a token.
"""

from __future__ import annotations

from typing import NamedTuple

from core.schemas import TranscriptSegment


class SpeakerTurn(NamedTuple):
    """One span of speech attributed to a speaker, in seconds from audio start."""

    start: float
    end: float
    speaker: str


def _overlap(segment: TranscriptSegment, turn: SpeakerTurn) -> float:
    """Seconds of overlap between a transcript segment and a speaker turn."""
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))


def _friendly_labels(turns: list[SpeakerTurn]) -> dict[str, str]:
    """Map raw pyannote ids (SPEAKER_00) to Speaker 1, 2, ... by first appearance."""
    labels: dict[str, str] = {}
    for turn in sorted(turns, key=lambda item: item.start):
        if turn.speaker not in labels:
            labels[turn.speaker] = f"Speaker {len(labels) + 1}"
    return labels


def assign_speakers(
    segments: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[TranscriptSegment]:
    """Label each segment with the speaker whose turn overlaps it most.

    Segments are never split, matching the rule chunking already follows: a
    segment straddling two speakers goes to whoever holds the majority of it.
    A segment overlapping no turn keeps ``speaker=None`` rather than being
    force-assigned, so unlabelled really means unknown.
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
            # Strictly greater keeps the earliest turn on a tie, since turns are
            # iterated in start order.
            overlap = _overlap(segment, turn)
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        speaker = labels[best_turn.speaker] if best_turn is not None else None
        labelled.append(segment.model_copy(update={"speaker": speaker}))
    return labelled
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/diarization.py tests/test_diarization.py
git commit -m "feat: assign speakers to transcript segments by timestamp overlap"
```

---

### Task 4: DiarizationService and the pyannote factory

**Files:**

- Modify: `core/diarization.py`
- Test: `tests/test_diarization.py`

**Interfaces:**

- Consumes: `SpeakerTurn` from Task 3, `Settings.diarization_enabled` from Task 2
- Produces: `DiarizationError`, `DiarizationModel` Protocol, `DiarizationService.diarize(audio_path: Path | str) -> list[SpeakerTurn]`, `create_diarization_service(settings: Settings) -> DiarizationService`

**API notes** (verified 2026-07-28, do not substitute from memory):

- The auth argument is `token=`. The older `use_auth_token=` is deprecated in pyannote 3.x.
- community-1's pipeline returns a **result object**, not an `Annotation`. The diarization lives at `output.speaker_diarization`, and you call `.itertracks(yield_label=True)` on that. This differs from 3.1, where the pipeline returns the `Annotation` directly, so examples written for 3.1 will mislead you here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diarization.py (append)
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.config import Settings
from core.diarization import (
    DiarizationError,
    DiarizationService,
    create_diarization_service,
)


class FakeAnnotation:
    """Mimics pyannote.core.Annotation's itertracks(yield_label=True)."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Any:
        for start, end, speaker in self._tracks:
            yield SimpleNamespace(start=start, end=end), "_", speaker


class FakeOutput:
    """Mimics the community-1 result object: diarization is one field of it."""

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: FAIL with `ImportError: cannot import name 'DiarizationError'`

- [ ] **Step 3: Write the implementation**

Append to `core/diarization.py`, and add `from pathlib import Path`, `from typing import Any, Protocol` plus `from core.config import Settings` to the imports:

```python
MODEL_ID = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    """Raised when diarization cannot run. Message states the exact fix."""


class DiarizationModel(Protocol):
    """The single capability this service needs: audio path in, result object out."""

    def __call__(self, audio: str) -> Any: ...


class DiarizationService:
    """Turns an audio file into speaker turns using an injected pyannote pipeline."""

    def __init__(self, model: DiarizationModel) -> None:
        self._model = model

    def diarize(self, audio_path: Path | str) -> list[SpeakerTurn]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # community-1 returns a result object; the diarization is one field of it.
        annotation = self._model(str(path)).speaker_diarization
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: PASS

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/diarization.py tests/test_diarization.py
git commit -m "feat: add pyannote-backed diarization service with lazy import"
```

---

### Task 5: The orchestrator that never fails

The single entry point both the CLI and the UI call. This is what keeps business logic out of `app/` and stops the two entry points duplicating error handling.

**Files:**

- Modify: `core/diarization.py`
- Test: `tests/test_diarization.py`

**Interfaces:**

- Consumes: everything from Tasks 1-4
- Produces: `apply_diarization(transcript, audio_path, settings, *, service_factory=create_diarization_service) -> tuple[Transcript, str | None]` returning the (possibly labelled) transcript and an optional warning message

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diarization.py (append)
from core.diarization import apply_diarization
from core.schemas import Transcript


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_diarization'`

- [ ] **Step 3: Write the implementation**

Append to `core/diarization.py`, adding `from collections.abc import Callable` and `Transcript` to the imports:

```python
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
        return transcript, f"Speaker diarization failed, continuing without labels: {exc}"

    return Transcript(segments=assign_speakers(transcript.segments, turns)), None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: PASS

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/diarization.py tests/test_diarization.py
git commit -m "feat: add apply_diarization orchestrator with graceful degradation"
```

---

### Task 6: Carry labels through the chunked path

`Transcript.text` already carries labels after Task 1, so the whole-transcript path and the Markdown export need no changes. The map step is the one place that joins raw segments itself, and if it is missed, labels reach short calls but vanish on long ones.

**Files:**

- Modify: `core/insights.py:56`
- Test: `tests/test_insights.py`

**Interfaces:**

- Consumes: `render_segments` from Task 1
- Produces: nothing new

- [ ] **Step 1: Write the failing test**

```python
# tests/test_insights.py (append to the chunking tests)
def test_chunked_path_sends_speaker_labels_to_the_model() -> None:
    """A long labelled transcript must not lose its labels in the map step."""
    segments = [
        TranscriptSegment(
            text="word " * 400, start=float(i * 10), end=float(i * 10 + 10),
            speaker="Speaker 1" if i % 2 == 0 else "Speaker 2",
        )
        for i in range(8)
    ]
    chat_model = FakeChatModel(responses=[VALID_BRIEF_JSON] * 12)
    engine = InsightEngine(chat_model, max_chunk_tokens=1000)

    engine.generate_brief(Transcript(segments=segments))

    chunk_prompts = "".join(
        str(message.content) for messages in chat_model.calls for message in messages
    )
    assert "Speaker 1:" in chunk_prompts
    assert "Speaker 2:" in chunk_prompts
```

`FakeChatModel` (`tests/test_insights.py:30`) and `VALID_BRIEF_JSON` (`:11`) already exist, and the fake already records every invocation in `self.calls`. No test-helper changes needed.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_insights.py::test_chunked_path_sends_speaker_labels_to_the_model -v`
Expected: FAIL, because the chunk join drops labels

- [ ] **Step 3: Write the implementation**

In `core/insights.py`, import the renderer:

```python
from core.schemas import Brief, InsightResult, Transcript, render_segments
```

Replace line 56:

```python
            chunk_text = render_segments(chunk)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_insights.py -v`
Expected: PASS

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/insights.py tests/test_insights.py
git commit -m "fix: keep speaker labels in the chunked map step"
```

---

### Task 7: Teach the prompt to use speaker labels

**Files:**

- Modify: `core/prompts.py:29-34`
- Test: `tests/test_prompts.py`

**Interfaces:**

- Consumes: nothing
- Produces: nothing new. `_CHUNK_SYSTEM` and `_SYNTHESIS_SYSTEM` both concatenate `_SYSTEM`, so this single edit covers the map and reduce paths too.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py (append)
def test_system_prompt_instructs_the_model_to_use_speaker_labels() -> None:
    messages = build_brief_messages("Speaker 1: Send it by Friday.")
    system = str(messages[0].content)
    assert "Speaker 1" in system
    assert "owner" in system


def test_chunk_and_synthesis_prompts_inherit_the_speaker_instruction() -> None:
    chunk_system = str(build_chunk_brief_messages("text", 1, 2)[0].content)
    synthesis_system = str(build_synthesis_messages(["notes"])[0].content)
    assert "Speaker 1" in chunk_system
    assert "Speaker 1" in synthesis_system
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL on the `"Speaker 1" in system` assertion

- [ ] **Step 3: Write the implementation**

In `core/prompts.py`, add one bullet to the `Rules:` block in `_SYSTEM`, after the "Every action item MUST include its supporting verbatim quote" line:

```
- The transcript may label who is speaking (e.g. "Speaker 1:", "Speaker 2:"). \
When it does, use those labels to set each action item's "owner" instead of \
inferring ownership from phrasing.
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/prompts.py tests/test_prompts.py
git commit -m "feat: instruct the model to use speaker labels for action-item owners"
```

---

### Task 8: Wire diarization into the CLI and the UI

Both entry points call `apply_diarization` and surface the warning. No toggle is exposed: reaching a state where diarization runs already requires installing the extra and setting a token.

**Files:**

- Modify: `core/cli.py`
- Modify: `app/main.py`
- Test: `tests/test_cli.py`, `tests/test_app.py`

**Interfaces:**

- Consumes: `apply_diarization` from Task 5
- Produces: nothing new

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py (append)
def test_cli_labels_the_transcript_and_stays_silent_on_success(
    capsys, audio_file: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "core.cli.apply_diarization",
        lambda transcript, path, settings: (transcript, None),
    )
    exit_code = run_cli([str(audio_file)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "diarization" not in captured.err.lower()


def test_cli_warns_on_stderr_but_still_produces_a_brief(
    capsys, audio_file: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "core.cli.apply_diarization",
        lambda transcript, path, settings: (transcript, "Install: uv sync --extra diarization"),
    )
    exit_code = run_cli([str(audio_file)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "uv sync --extra diarization" in captured.err
    assert "# EchoBrief" in captured.out
```

`run_cli` (`tests/test_cli.py:55`) already injects stub `health_check`, `transcription_factory`, and `engine_factory` defaults, and the `audio_file` fixture already exists at `:48`. No test-helper changes needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `AttributeError: module 'core.cli' has no attribute 'apply_diarization'`

- [ ] **Step 3: Write the implementation**

In `core/cli.py`, add the import:

```python
from core.diarization import apply_diarization
```

Directly after the existing transcription line, add:

```python
    transcript, diarization_warning = apply_diarization(transcript, audio_path, settings)
    if diarization_warning:
        print(diarization_warning, file=sys.stderr)
```

In `app/main.py`, add `diarization` to the `from core import ...` line, and inside the `st.status` block after `_transcribe_upload`, add:

```python
            transcript, diarization_warning = diarization.apply_diarization(
                transcript, audio_path, settings
            )
            if diarization_warning:
                st.warning(diarization_warning)
```

`_transcribe_upload` currently deletes the temp file when its `with` block exits, so it must return the path alongside the transcript for diarization to read the same audio. Change its signature to return `tuple[Transcript, Path]` and move the diarization call inside the `NamedTemporaryFile` context, keeping the privacy guarantee that uploaded audio never outlives the run.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: all pass

- [ ] **Step 5: Request approval, then commit**

```bash
git add core/cli.py app/main.py tests/test_cli.py tests/test_app.py
git commit -m "feat: run diarization from the CLI and the Streamlit UI"
```

---

### Task 9: Packaging and documentation

**Files:**

- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `docs/DESIGN.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**

- Consumes: everything
- Produces: the `diarization` optional extra

- [ ] **Step 1: Add the optional extra**

In `pyproject.toml`, after the `dependencies` list:

```toml
[project.optional-dependencies]
diarization = [
    "pyannote.audio>=4.0",
]
```

The floor is 4.0, not 3.1: `speaker-diarization-community-1` and the `output.speaker_diarization` result shape are both 4.x APIs, so a 3.x resolution would fail at runtime rather than at install. Latest at time of writing is 4.0.7, and its `requires-python >=3.10` matches this project's floor.

Run: `uv sync --extra diarization` and confirm it resolves. Then run `uv sync` alone and confirm the default install still excludes torch.

- [ ] **Step 2: Verify the real pyannote call by hand**

This is the one thing no automated test covers, by deliberate choice in the spec. With a token set and the model terms accepted, run the CLI against `tests/fixtures/sample_call.mp3` and confirm the brief comes back with `Speaker N` labels in the transcript section. Record the result here:

```
Manual verification (date, outcome):
```

- [ ] **Step 3: Update `.env.example`**

```bash
# --- Optional: speaker diarization via pyannote (labels who said what) ---
# Needs the optional extra: uv sync --extra diarization
# Create a token at https://huggingface.co/settings/tokens and accept the model
# terms at https://hf.co/pyannote/speaker-diarization-community-1
# HUGGINGFACE_TOKEN=
```

- [ ] **Step 4: Update `docs/DESIGN.md`**

- Add to the functional requirements table: `| F9 | *(added post-MVP, 2026-07-28)* Transcript segments are labelled by speaker, and those labels are passed to the LLM so action-item ownership is grounded in the audio. Optional: needs the diarization extra and a Hugging Face token. |`
- Add a trade-off matrix row for pyannote over WhisperX, carrying the reasoning from the spec, including why community-1 was chosen over 3.1 (one gated repo instead of two).
- Add a risk matrix row: gated-model friction (likelihood high, impact low, mitigation: feature is optional and degrades to an unlabelled transcript with a message naming the exact fix).
- Amend the privacy NFR to note that pyannote downloads weights from Hugging Face on first run and the token is transmitted there at download time, while inference is fully local. This is the same class of access Whisper weights already use.

- [ ] **Step 5: Update `README.md` and `CLAUDE.md`**

- README roadmap: tick the diarization item and change "WhisperX" to "pyannote".
- README project structure and CLAUDE.md layout: both list `docs/` as `DESIGN.md · model-eval.md`. Add `specs/` and `TODO.md`.
- README: document `uv sync --extra diarization` and `HUGGINGFACE_TOKEN` in the configuration table, including the one-time terms acceptance at https://hf.co/pyannote/speaker-diarization-community-1.
- README: add attribution for `pyannote/speaker-diarization-community-1`, which is CC-BY-4.0 and so carries an attribution requirement, unlike the MIT-licensed 3.1.

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest --cov=core --cov-fail-under=80`
Expected: all pass, coverage on `core/` >= 80%

- [ ] **Step 7: Request approval, then commit**

```bash
git add pyproject.toml uv.lock .env.example docs/DESIGN.md README.md CLAUDE.md
git commit -m "feat: add diarization optional extra and document F9"
```

---

## Resolved: which pyannote model (settled 2026-07-28)

The plan originally flagged an open question about whether `speaker-diarization-community-1` was ungated. It is not. Checked against the Hugging Face API:

| Repo                                             | Gated    | Ships weights             |
| ------------------------------------------------ | -------- | ------------------------- |
| `pyannote/speaker-diarization-3.1`             | `auto` | No, it is a manifest only |
| `pyannote/segmentation-3.0` (pulled in by 3.1) | `auto` | Yes                       |
| `pyannote/speaker-diarization-community-1`     | `auto` | Yes, all of them          |

Both pipelines require a token and a terms acceptance, and both return HTTP 401 on file downloads without one. community-1 was chosen anyway because 3.1 requires accepting terms on **two** gated pages, and forgetting the second produces a load failure that does not name what is missing. community-1 needs one acceptance.

Costs accepted: the result object shape differs (`output.speaker_diarization`, handled in Task 4), and the licence is CC-BY-4.0 rather than MIT, so attribution is required (handled in Task 9).

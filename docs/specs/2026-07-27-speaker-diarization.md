# Speaker Diarization (F9) — Design

**Date:** 2026-07-27
**Status:** Approved, ready for implementation planning
**Requirement ID:** F9

## Problem

EchoBrief produces a single undifferentiated wall of transcript text. In the flagship use case, a supervision call, the most decision-relevant fact about any sentence is *who said it*. Without that, the `owner` field on every action item is inferred from conversational phrasing rather than grounded in the audio, and a confidently wrong owner is a trust failure: the user cannot tell whether "send the revised chapter by Friday" is a task assigned to them or one they assigned to someone else.

Diarization labels each transcript segment with its speaker and feeds those labels to the LLM, so attribution is grounded rather than guessed.

## Scope

Speaker labels are produced, displayed, exported, **and passed to the LLM**. The prompt is updated so the model uses labels when setting `owner`.

Explicitly out of scope:

- Mapping speakers to real identities ("me" vs "supervisor"). Labels remain `Speaker 1`, `Speaker 2`. This is the natural follow-up if labels prove useful in practice.
- Per-speaker talk-time analytics.
- Word-level forced alignment.
- Diarization of the F8 voice output.
- Any user-facing control to toggle the feature. See "Configuration" for why.

## Decision: pyannote alongside faster-whisper, not WhisperX

The README roadmap previously named WhisperX. That is superseded by this document.

| Option | Verdict |
|---|---|
| **pyannote + faster-whisper, merged by timestamp overlap** | **Chosen.** Additive. Transcription is untouched, so its tests and the published benchmark numbers stay valid. The merge is a pure function, testable without audio, GPU, or token. |
| WhisperX replaces the transcription layer | Rejected. Bundles faster-whisper + wav2vec2 alignment + pyannote into one pipeline. The genuine win is word-level timestamps placing boundaries mid-sentence, but that mainly matters for overlapping speech, which is not the failure mode of a two-person call. Cost is rewriting a working, covered layer and invalidating the measured results in `docs/model-eval.md`. The chosen approach does not block adding alignment later if precision proves insufficient. |
| LLM infers speakers from text alone | Rejected. Zero new dependencies, but it contradicts the "use ONLY information present in the transcript" constraint in `prompts.py`. Unreliable labels feeding attribution are worse than no labels. |

## Architecture

### New module: `core/diarization.py`

- **`SpeakerTurn`** — a `NamedTuple` of `(start, end, speaker)`. It lives here, not in `schemas.py`, because `schemas.py` holds contracts *between layers*; this type is produced and consumed inside this one module and never crosses a boundary.
- **`DiarizationService`** — wraps an injected diarization model; `diarize(audio_path) -> list[SpeakerTurn]`. A class rather than a plain function so the UI can cache the loaded model with `@st.cache_resource`, matching how the transcription service is already handled.
- **`assign_speakers(segments, turns) -> list[TranscriptSegment]`** — pure function, no I/O. All non-trivial logic lives here.
- **`create_diarization_service(settings)`** — the factory, and the **only** place `pyannote` is imported. The import sits inside the function body so `core.diarization` remains importable when the optional extra is not installed. This import placement is load-bearing for graceful degradation, not stylistic.
- **`DiarizationError`** — carries remediation text, mirroring `SpeechError`.

The narrow `DiarizationModel` Protocol declares only the single call the service needs, keeping test fakes trivial.

The model ID `pyannote/speaker-diarization-community-1` is a module-level constant, not an env var. Unlike `WHISPER_MODEL_SIZE`, which exists because `model-eval.md` documents an accuracy/speed tradeoff users act on, there is no comparable tradeoff to expose here. Precedent: `_TTS_ENDPOINT` in `speech.py`.

**Why community-1 rather than 3.1** (verified against the Hugging Face API, 2026-07-28): both are `gated: auto`, so neither avoids the token or the terms acceptance. But `speaker-diarization-3.1` ships **no weights of its own**. It is a manifest pointing at two other repos, one of which (`pyannote/segmentation-3.0`) is *also* gated, so a 3.1 user must accept terms on two separate pages and missing the second produces a load-time failure that does not name the page they missed. community-1 bundles segmentation, embedding, and PLDA weights in a single repo: one acceptance, no hidden second gate.

Two consequences accepted: its result object exposes the diarization at `output.speaker_diarization` rather than being an `Annotation` directly, and its licence is CC-BY-4.0 rather than MIT, which carries an attribution requirement to note in the README.

### Schema changes (`core/schemas.py`)

`TranscriptSegment` gains `speaker: str | None = None`. Defaulted, so every existing call site and test stays valid.

`schemas.py` also gains the `render_segments` function described under "Speaker-aware text rendering", and `Transcript.text` becomes a one-line delegation to it.

### Data flow

```
audio ─┬─> TranscriptionService ──> segments (text, start, end)
       │                                    │
       └─> DiarizationService ───> turns ───┤
                                            v
                                    assign_speakers()
                                            │
                                            v
                            segments (text, start, end, speaker)
```

### Assignment rules

- Each segment takes the speaker whose turn overlaps it most **by duration**.
- Segments are never split, matching the rule `chunking.py` already follows.
- A segment overlapping no turn keeps `speaker=None` rather than being force-assigned.
- Ties resolve to the earlier-starting turn, so the result is deterministic.
- Raw pyannote IDs (`SPEAKER_00`) are remapped to `Speaker 1`, `Speaker 2`, ordered by first appearance in the call.

## Speaker-aware text rendering

Transcript text is currently rendered in three places:

- `insights.py:49` — whole-transcript path
- `insights.py:56` — the per-chunk join in the map step
- `export.py:24` — the Markdown full-transcript section

If labels reach some but not all of these, the LLM receives labels on short calls and silently loses them on chunked long calls. That inconsistency is hard to diagnose and easy to prevent.

**One shared renderer, `render_segments(segments: list[TranscriptSegment]) -> str`, in `schemas.py`.** It groups consecutive segments from the same speaker into a single labeled turn, emitting `Speaker 1:` once per turn rather than once per segment, which serves both readability and token budget.

It lives in `schemas.py` beside the models because `Transcript.text` already does exactly this job today and simply delegates to it:

```python
@property
def text(self) -> str:
    return render_segments(self.segments)
```

That delegation means **two of the three call sites need no change at all**. `insights.py:49` and `export.py:24` both go through `Transcript.text` and pick up labels for free. Only `insights.py:56`, which joins a raw `list[TranscriptSegment]` for a chunk and has no `Transcript` to call, changes to invoke `render_segments(chunk)` directly.

When every segment has `speaker=None`, the renderer returns exactly today's plain joined text, so behavior with the extra absent is byte-for-byte unchanged.

**`chunking.py` is not modified.** Label prefixes add a few characters per speaker turn against a 3000-token chunk budget inside an 8k context, which disappears within both the existing headroom and the error bar of the 4-chars-per-token estimator.

## Configuration

One new env var:

| Env var | Default | Purpose |
|---|---|---|
| `HUGGINGFACE_TOKEN` | `""` | Gated-model access. Presence enables the feature. |

A `diarization_enabled` property derives from token presence, exactly like `tts_enabled`. Note the distinction: `diarization_enabled` reports **configuration intent** (a token is set) and lives in `config.py`, which must not import pyannote. Whether the feature can actually run additionally requires the extra to be installed, detected at the import site in `create_diarization_service`. Both conditions are reported separately so the remediation message names the right fix.

Packaging: an optional extra, installed with `uv sync --extra diarization`. The default install and the four-command quickstart are untouched, so the student path never grows a torch download or an account signup.

**No toggle is exposed.** Diarization runs whenever it is available. Reaching that state already requires two deliberate opt-ins (installing the extra, setting a token), so a third control would be a knob for a situation that cannot arise: a user who wants it off unsets the token.

## Error handling

Two mechanisms, not four. In every case the user still receives a complete brief, unlabeled.

1. **Availability check**, before running. Returns a reason string when the extra is missing (naming `uv sync --extra diarization`) or the token is unset (naming the env var and the gated-model URL whose terms must be accepted).
2. **`try/except` around the diarization step**, covering everything that fails at runtime: a rejected token, an inaccessible model, or any pyannote exception. Warns with the underlying failure and proceeds.

Diarization is strictly an enhancement, never a dependency of the main path.

## Testing

**Unit only.** No network, no models, no GPU.

- Overlap assignment: exact overlap, partial overlap, ties, zero overlap, segment spanning two turns, empty turn list, empty segment list
- ID remapping order by first appearance
- Renderer: consecutive-same-speaker grouping, speaker changes, all-`None` plain-text fallback, mixed `None` and labeled segments
- `DiarizationService` with a fake model injected through the constructor
- Both error mechanisms above

**No pyannote integration test.** It would need a token and manually accepted gated terms, so it could never run in CI and would run approximately never in practice: maintenance cost without a signal. The logic that can actually break (overlap assignment, the renderer) is fully covered by fast unit tests. The tradeoff accepted is that the shape of our pyannote call is not proven by automated test; it will be verified by hand once during implementation.

Coverage gate on `core/` remains at 80%.

## Documentation updates

- `docs/DESIGN.md`: add F9 to the functional requirements table; add a trade-off matrix row recording the pyannote-over-WhisperX decision; add a risk row for gated-model friction.
- `docs/DESIGN.md` privacy NFR: note explicitly that pyannote downloads weights from Hugging Face on first run and the token is transmitted there at download time, while **inference is fully local**. This is the same class of network access Whisper weights already use, but it must be stated rather than left implicit.
- `README.md`: tick the roadmap item and correct "WhisperX" to "pyannote".
- `README.md` and `CLAUDE.md`: both describe `docs/` as `DESIGN.md · model-eval.md`, which no longer includes `docs/specs/`. Correct in passing.
- `.env.example`: add `HUGGINGFACE_TOKEN`.

## Definition of done

1. Every behavior developed red-green-refactor; test and implementation in the same commit.
2. `ruff check` and `ruff format --check` pass; public API type-hinted.
3. Coverage on `core/` still >= 80%.
4. Both error mechanisms verified by tests.
5. `core/` still imports no Streamlit; no logic added to `app/`.
6. Docs updated as above.

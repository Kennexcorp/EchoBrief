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

## Decision: pyannote alongside faster-whisper, not WhisperX

The README roadmap previously named WhisperX. That is superseded by this document.

| Option | Verdict |
|---|---|
| **pyannote + faster-whisper, merged by timestamp overlap** | **Chosen.** Additive. Transcription is untouched, so its tests and the published benchmark numbers stay valid. The merge is a pure function, testable without audio, GPU, or token. |
| WhisperX replaces the transcription layer | Rejected. Bundles faster-whisper + wav2vec2 alignment + pyannote into one pipeline. The genuine win is word-level timestamps placing boundaries mid-sentence, but that mainly matters for overlapping speech, which is not the failure mode of a two-person call. Cost is rewriting a working, covered layer and invalidating the measured results in `docs/model-eval.md`. A does not block adding alignment later if precision proves insufficient. |
| LLM infers speakers from text alone | Rejected. Zero new dependencies, but it contradicts the "use ONLY information present in the transcript" constraint in `prompts.py`. Unreliable labels feeding attribution are worse than no labels. |

## Architecture

### New module: `core/diarization.py`

Three units with distinct responsibilities:

1. **`DiarizationService`** — wraps an injected diarization model; `diarize(audio_path) -> list[SpeakerTurn]`. Structurally identical to `TranscriptionService`.
2. **`assign_speakers(segments, turns) -> list[TranscriptSegment]`** — pure function, no I/O. All non-trivial logic lives here.
3. **`create_diarization_service(settings)`** — the factory, and the **only** place `pyannote` is imported. The import sits inside the function body so `core.diarization` remains importable when the optional extra is not installed. This import placement is what makes graceful degradation possible; it is load-bearing, not stylistic.

A `DiarizationError` carries remediation text, mirroring `SpeechError`.

The narrow `DiarizationModel` Protocol declares only the single call the service needs, keeping test fakes trivial (interface segregation).

### Schema changes (`core/schemas.py`)

- `TranscriptSegment` gains `speaker: str | None = None`. Defaulted, so every existing call site and test stays valid.
- New `SpeakerTurn` model: `start: float`, `end: float`, `speaker: str`, with the same `end >= start` validation as `TranscriptSegment`.

Contracts live in `schemas.py`, not in the new module.

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

If labels reach some but not all of these, the LLM receives labels on short calls and loses them on chunked long calls. That inconsistency is hard to diagnose and easy to prevent.

**One shared renderer, `render_segments(segments) -> str`, is used by all three.** It groups consecutive segments from the same speaker into a single labeled turn, emitting `Speaker 1:` once per turn rather than once per segment. This serves both readability and token budget.

When every segment has `speaker=None`, the renderer returns exactly today's plain joined text, so behavior with the extra absent is byte-for-byte unchanged.

Chunk token estimates in `chunking.py` account for the label prefix so labeled chunks cannot silently overflow the context budget.

## Configuration

Following the F8 pattern:

| Env var | Default | Purpose |
|---|---|---|
| `HUGGINGFACE_TOKEN` | `""` | Gated-model access. Presence enables the feature. |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Model ID. |

A `diarization_enabled` property derives from token presence, exactly like `tts_enabled`. Note the distinction: `diarization_enabled` reports **configuration intent** (a token is set) and lives in `config.py`, which must not import pyannote. Whether the feature can actually run additionally requires the extra to be installed, which is detected in `create_diarization_service` at the import site. Both conditions are reported separately so the remediation message can name the right fix.

Packaging: an optional extra, installed with `uv sync --extra diarization`. The default install and the four-command quickstart are untouched, so the student path never grows a torch download or an account signup.

Diarization runs by default **when available**, since it improves the default output. It can be skipped with `--no-diarize` on the CLI or a checkbox in the UI, because it adds real wall-clock time.

## Error handling

Four failure modes, none fatal. In every case the user still receives a complete brief.

| Condition | Behavior |
|---|---|
| Extra not installed | Unlabeled transcript; message names `uv sync --extra diarization` |
| No `HUGGINGFACE_TOKEN` | Unlabeled transcript; message names the env var and the gated-model URL whose terms must be accepted |
| Token rejected / model inaccessible | Unlabeled transcript; message includes the exact HTTP failure |
| Diarization raises mid-run | Warn, proceed unlabeled |

Diarization is strictly an enhancement and never a dependency of the main path.

## Testing

**Unit** (no network, no models, no GPU):

- Overlap assignment: exact overlap, partial overlap, ties, zero overlap, segment spanning two turns, empty turn list, empty segment list
- ID remapping order by first appearance
- Renderer: consecutive-same-speaker grouping, speaker changes, all-`None` plain-text fallback, mixed `None` and labeled segments
- `DiarizationService` via a fake model injected through the constructor
- One test per degradation path in the table above

**Integration:** one `@pytest.mark.integration` test running real pyannote on the bundled clip, skipped when the extra or token is absent, so CI stays green without secrets.

Coverage gate on `core/` remains at 80%.

## Documentation updates

- `docs/DESIGN.md`: add F9 to the functional requirements table; add a trade-off matrix row recording the pyannote-over-WhisperX decision; add a risk row for gated-model friction.
- `docs/DESIGN.md` privacy NFR: note explicitly that pyannote downloads weights from Hugging Face on first run and the token is transmitted there at download time, while **inference is fully local**. This is the same class of network access Whisper weights already use, but it must be stated rather than left implicit.
- `README.md`: tick the roadmap item and correct "WhisperX" to "pyannote".
- `.env.example`: add both new variables.

## Definition of done

1. Every behavior developed red-green-refactor; test and implementation in the same commit.
2. `ruff check` and `ruff format --check` pass; public API type-hinted.
3. Coverage on `core/` still >= 80%.
4. All four degradation paths verified by tests.
5. `core/` still imports no Streamlit; no logic added to `app/`.
6. DESIGN.md, README.md, and `.env.example` updated as above.

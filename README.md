# EchoBrief 🎙️→✅

**Turn recorded supervisor calls into summaries, insights, and tracked next steps — 100% locally.** No cloud APIs, no per-token billing, no audio ever leaving your machine. Built on faster-whisper + Ollama + Streamlit.

<!-- Badges: fill in your repo path -->
![CI](https://github.com/Kennexcorp/echobrief/actions/workflows/ci.yml/badge.svg)
![Coverage](PLACEHOLDER_COVERAGE_BADGE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

<!-- 📸 DEMO GOES HERE — this is the single most important element on the page.
     Record a 15–30s GIF: upload file → progress states → rendered brief → download.
     Tools: LICEcap, Kap, or `ffmpeg` screen capture. Keep it under 10 MB. -->
![Demo](docs/demo.gif)

## What it does

Students lose the value of mentor and supervisor calls because feedback and action items live inside un-reviewed audio recordings. EchoBrief takes an uploaded call recording (`.mp3`, `.wav`, `.m4a`, `.opus`, `.ogg` — WhatsApp voice notes work out of the box) and produces:

1. **A full transcript** (local Whisper inference — expandable panel)
2. **A 3–5 sentence summary** of the call
3. **Key insights & feedback received**
4. **Numbered action items** with priority and suggested deadlines — each backed by a quoted snippet from the transcript so you can verify nothing was hallucinated

Export the whole brief as Markdown and drop it into your notes.

**Why local?** Supervisor calls contain unreleased research, grades, and personal feedback. This pipeline makes zero network calls beyond `localhost` — the privacy guarantee is architectural, not a policy promise.

---

## Quickstart

### Path A — uv + Ollama (recommended)

Prereqs: [uv](https://docs.astral.sh/uv/getting-started/installation/), [ffmpeg](https://ffmpeg.org/download.html), [Ollama](https://ollama.com/download). No Python install needed — uv fetches the right version automatically.

```bash
git clone https://github.com/Kennexcorp/echobrief && cd echobrief
uv sync
ollama pull llama3.1:8b
uv run streamlit run app/main.py
```

First transcription downloads the Whisper `small` model (~500 MB) from the Hugging Face Hub automatically.

### Path B — Docker (one command, fully reproducible)

Prereqs: Docker + Compose.

```bash
git clone https://github.com/Kennexcorp/echobrief && cd echobrief
docker compose up
```

This stands up both the app and an Ollama service on a shared network, with model weights persisted in named volumes (nothing multi-GB is baked into the image). Using Ollama on your host instead? Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.

### Configuration

Copy `.env.example` → `.env`:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://ollama:11434` under compose |
| `OLLAMA_MODEL` | `llama3.1:8b` | any locally pulled chat model |
| `WHISPER_MODEL_SIZE` | `small` | `base` for low-RAM machines, `medium`+ for GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on GPU |

---

## Results

<!-- ⚠️ REPLACE EVERY PLACEHOLDER WITH A MEASURED NUMBER BEFORE PUBLISHING.
     These are the numbers to capture during the build — see "How to measure" below each. -->

| Metric | Result |
|---|---|
| 30-min call transcription (4-core CPU, `small`/INT8) | **X min Y s** <!-- time.perf_counter() around model.transcribe(); use one real recording --> |
| Same audio, vanilla `openai-whisper` baseline | **X min Y s (N× slower)** <!-- run once for the comparison claim --> |
| End-to-end: upload → rendered brief (30-min call) | **X min** |
| Structured-output validity (first attempt, N=20 runs) | **Z %** <!-- count Pydantic parse successes across repeated runs on 3 sample transcripts --> |
| Structured-output validity (with 1 retry) | **Z %** |
| Unit test coverage (pipeline layer) | **N %** <!-- from pytest-cov; gate is ≥80% in CI --> |
| Model comparison (llama3.1:8b vs qwen2.5:7b vs mistral:7b) | see [docs/model-eval.md](docs/model-eval.md) <!-- score each on 2–3 transcripts: JSON validity, faithfulness, action-item recall --> |

---

## How it works

```
Upload (Streamlit) → faster-whisper (CTranslate2, INT8) → transcript
    → LangChain ChatPromptTemplate → ChatOllama (localhost:11434)
    → Pydantic-validated structured brief → render + Markdown export
```

<!-- Optionally paste the full ASCII/Mermaid architecture diagram from docs/DESIGN.md -->

### Why these choices

| Choice | Over | Because |
|---|---|---|
| **faster-whisper** | vanilla `openai-whisper`, HF `transformers` pipeline | CTranslate2 runtime is ~4× faster at identical accuracy with INT8 quantization — the difference between "usable on a student laptop" and not. Models still auto-download from the HF Hub. |
| **`langchain-ollama` (ChatOllama)** | raw `requests` / `ollama` client | Versioned prompt templates, `.with_structured_output()` + Pydantic enforcement of the brief schema, and one-line model portability. Accepted trade-off: heavier dependency than the bare client for a single-chain app. |
| **uv-first, Docker-second packaging** | pip + venv, Docker-only | The target user is a non-technical student; `uv sync` is one command with no venv-activation step (the classic Windows tripwire), installs Python itself, and pins everything via `uv.lock` — reproducibility on the light path too. The compose path exists for full-stack reproducibility and reviewers. Packaging matched to the user, not to fashion. |
| **Ollama** | llama.cpp server, vLLM | Simplest cross-platform install and model management story for the audience. |

Full trade-off analysis, requirements, and risk matrix: [docs/DESIGN.md](docs/DESIGN.md)

---

## Engineering challenges

<!-- Write these in PAST TENSE once solved — this is your STAR "Action" section.
     Keep each to 3–5 sentences: problem → why it's hard → what you did → measured outcome. -->

**Hour-long transcripts vs. context windows.** A 60-minute call produces a transcript well beyond a 7–8B model's usable context. I chunk on Whisper segment boundaries (so no sentence is split mid-thought), summarize each chunk, then run a final synthesis pass over the chunk summaries — classic map-reduce. <!-- add: measured token counts and how many chunks a 60-min call yields -->

**Making a local model tell the truth.** Small local models happily invent plausible action items. Two guardrails: the system prompt restricts the model to transcript content only, and every action item is rendered with its supporting transcript quote, making hallucinations immediately visible to the user. <!-- add: observed hallucination rate before/after, if measured -->

**Guaranteeing parseable output.** The UI depends on structured data, but LLM JSON is unreliable. The brief schema is a Pydantic model enforced through LangChain's structured output against Ollama's schema mode, with one automatic repair-prompt retry on parse failure and graceful fallback to raw text. <!-- add: your measured validity % -->

**Testing without a GPU in CI.** GitHub Actions runners can't run a live Ollama model, so the pipeline layer is tested against mocked LLM responses (valid, malformed, and partial JSON cases) plus a bundled 30-second audio clip for the transcription integration test.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` on startup health check | Ollama isn't running — `ollama serve`, or check `OLLAMA_BASE_URL` |
| Model not found error | `ollama pull llama3.1:8b` (or whatever `OLLAMA_MODEL` is set to) |
| `ffmpeg not found` (Path A) | Install ffmpeg and ensure it's on PATH; Path B bundles it |
| Docker app can't reach host Ollama | `OLLAMA_BASE_URL=http://host.docker.internal:11434` |
| Very slow / out of memory | `WHISPER_MODEL_SIZE=base`; close other apps; INT8 is already the default |

---

## Roadmap

- [ ] Speaker diarization — who said what (WhisperX)
- [ ] Call history dashboard with search across past briefs
- [ ] Calendar / task-manager export (ICS, Todoist)
- [ ] Multilingual transcription & translation
- [ ] Live transcription during calls

## Project structure

```
app/            Streamlit UI
core/           transcription.py · insights.py · prompts.py · schemas.py
tests/          unit + integration (mocked LLM)
docs/           DESIGN.md (full planning doc) · model-eval.md · demo.gif
pyproject.toml, uv.lock, Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## License

MIT — see [LICENSE](LICENSE).

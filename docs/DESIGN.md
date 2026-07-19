# EchoBrief — Turning Supervisor Calls into Actionable Next Steps
### A Local, Privacy-First Speech-to-Insight Pipeline (Whisper + Ollama + Streamlit)

> **The Hook:** *"EchoBrief — converting recorded mentorship calls into structured summaries, actionable insights, and tracked next steps using fully local ML inference (faster-whisper + Ollama)."*

---

## Phase 1: Inception & Problem Alignment

### The Problem
Students and early-career professionals regularly have review calls with supervisors, mentors, and advisors. These calls are dense with feedback, corrections, and follow-up tasks — but the value decays fast:

- Students take incomplete notes (or none) while trying to stay engaged in the conversation.
- Action items get mentioned verbally, scattered across a 30–60 minute call, and forgotten within days.
- Recordings exist but nobody re-listens to a full hour of audio to extract three to-dos.
- Cloud transcription/summarization tools raise privacy concerns — supervisor calls often contain unreleased research, grades, personal feedback, or institutional information that should never leave the student's machine.

### Target Audience
- **Primary:** University students (undergrad/postgrad) with recurring supervisor or thesis-advisor meetings.
- **Secondary:** Interns, mentees in professional development programs, research assistants.

> **Post-build broadening (2026-07-19):** the pipeline proved fully domain-neutral, so prompts and copy were generalized to any recorded call (coaching sessions, client check-ins, 1:1s). The student use case remains the flagship narrative; specialization happens through the optional user-context field rather than per-domain modes.

### User Constraints
- Consumer-grade hardware (laptop, likely no dedicated GPU) — the pipeline must run acceptably on CPU.
- Non-technical users — a single upload-and-read workflow, no CLI knowledge required.
- Zero cloud dependency — all inference (speech-to-text and LLM) runs locally.
- Common audio formats from phone/laptop recordings: `.mp3`, `.wav`, `.m4a`, plus `.opus`/`.ogg` (WhatsApp voice notes).

### Scope Boundaries

| In-Scope (MVP) | Out-of-Scope (v1) |
|---|---|
| Upload a single pre-recorded audio file | Live/real-time transcription during the call |
| Local speech-to-text via faster-whisper | Speaker diarization (who said what) |
| Structured LLM output: summary, insights, next steps | Calendar/task-manager integrations (Todoist, Google Tasks) |
| Local Ollama endpoint as the only LLM backend | Cloud LLM fallback (OpenAI/Anthropic APIs) |
| Download/export the brief as Markdown | Multi-call history dashboard and search |
| English-language calls | Multilingual transcription & translation |

Explicitly deferring diarization, live transcription, and history tracking keeps the MVP shippable in weeks, not months — and each deferred item becomes a credible roadmap entry in the final case study.

---

## Phase 2: Specifications & Guardrails

### Functional Requirements

| ID | Requirement |
|---|---|
| F1 | User can upload an audio file (`.mp3`, `.wav`, `.m4a`, `.opus`, `.ogg`) up to 60 minutes / ~200 MB via the Streamlit UI. |
| F2 | System transcribes the audio locally and displays the full transcript in an expandable panel. |
| F3 | Transcript is injected into a structured prompt template and sent to a local Ollama model. |
| F4 | LLM returns a structured brief: **(a)** 3–5 sentence summary, **(b)** key insights/feedback received, **(c)** numbered actionable next steps with suggested priority. |
| F5 | User can download the full brief (transcript + summary + actions) as a Markdown file. |
| F6 | User can select the Ollama model from a dropdown of locally available models. |
| F7 | Clear progress indicators for each stage (transcribing → summarizing → done). |

### Non-Functional Requirements

| Category | Guardrail |
|---|---|
| **Performance** | Transcription of a 30-min call completes in ≤ 10 minutes on a 4-core CPU using faster-whisper `small` with INT8 quantization; ≤ 2 minutes on GPU. |
| **Privacy** | No network calls leave localhost. Audio files and transcripts are processed in a temp directory and deleted on session end. |
| **Reliability** | Graceful degradation: if Ollama is unreachable, the user still receives the raw transcript plus a clear remediation message. |
| **Output validity** | LLM output must parse into the expected structure ≥ 95% of runs (enforced via structured output / JSON schema, with one automatic retry on parse failure). |
| **Portability** | Runs on Windows, macOS, Linux with `uv sync` + a running Ollama daemon — uv provisions Python and pins all dependencies via `uv.lock`. Documented model pull commands. |
| **Testability** | ≥ 80% unit-test coverage on the pipeline layer (transcription wrapper, prompt builder, output parser). |

---

## Phase 3: Architecture with Justification

### System Context & Data Flow

```
┌────────────┐   audio file    ┌──────────────────────────────┐
│  Student   │ ──────────────► │  Streamlit App (UI layer)    │
└────────────┘                 └──────────────┬───────────────┘
                                              │ file path
                                              ▼
                               ┌──────────────────────────────┐
                               │ Transcription Service        │
                               │ faster-whisper (CTranslate2) │
                               │ model: small / base, INT8    │
                               └──────────────┬───────────────┘
                                              │ transcript (text + segments)
                                              ▼
                               ┌──────────────────────────────┐
                               │ Insight Engine               │
                               │ LangChain: ChatPromptTemplate│
                               │  → ChatOllama                │
                               │  → structured output parser  │
                               └──────────────┬───────────────┘
                                              │ HTTP (localhost:11434)
                                              ▼
                               ┌──────────────────────────────┐
                               │ Ollama daemon                │
                               │ e.g. llama3.1:8b / qwen2.5   │
                               └──────────────┬───────────────┘
                                              │ structured brief (JSON)
                                              ▼
                               ┌──────────────────────────────┐
                               │ Renderer + Markdown exporter │
                               └──────────────────────────────┘
```

### Tech Stack Evaluation Matrix

| Layer | Chosen | Alternatives Considered | Justification (the trade-off) |
|---|---|---|---|
| **Speech-to-Text** | `faster-whisper` | `openai-whisper` (vanilla), HF `transformers` ASR pipeline, `whisper.cpp` | faster-whisper reimplements Whisper on CTranslate2 and is up to ~4x faster than vanilla Whisper at the same accuracy on commodity x86 CPUs, with lower memory use and INT8 quantization — critical for our CPU-only student hardware constraint. (Post-build measurement: on Apple Silicon the speed advantage did not reproduce — matched-settings decoding was parity; the memory, install-size, and VAD advantages held. See README Results.) Vanilla Whisper is too slow on CPU; the transformers pipeline is a strong GPU option but slower than CTranslate2 in this deployment profile. Models auto-download from the Hugging Face Hub, so we still benefit from the HF ecosystem without importing the heavier `transformers` stack. |
| **STT model size** | `small` (default), `base` (low-RAM fallback) | `medium`, `large-v3`, `distil-large-v3` | `small` is the accuracy/speed sweet spot for clear single-speaker or two-speaker call audio on CPU. `large-v3` gains marginal accuracy on clean meeting audio but is not viable on 8 GB laptops. Configurable via env variable. |
| **LLM runtime** | Ollama (localhost:11434) | llama.cpp server, LM Studio, vLLM | Given as a project constraint — and a good one: simplest install story for students, model management (`ollama pull`), OpenAI-compatible-ish API, cross-platform. |
| **LLM orchestration** | `langchain-ollama` (`ChatOllama`) + LangChain prompt templates & output parsing | Raw `requests` to Ollama REST API, `ollama` Python client | LangChain is justified here for three concrete reasons: (1) `ChatPromptTemplate` gives versioned, testable prompt construction instead of f-string spaghetti; (2) Ollama's JSON-schema-constrained decoding (`format=` the Pydantic schema) enforces our F4 output contract, with parsing/retry/fallback kept in our own unit-tested code rather than hidden inside LangChain; (3) swapping `ChatOllama` for any other chat model later is a one-line change, future-proofing the roadmap. Trade-off acknowledged: LangChain adds dependency weight for what is currently a single-chain app — if the project never grows beyond one prompt, the raw `ollama` client would be leaner. We accept the weight for the structured-output and portability wins. |
| **UI** | Streamlit | Gradio, Flask + React | Given as a constraint; also correct for the audience — `st.file_uploader`, `st.status`, and `st.download_button` map 1:1 onto our user flow with near-zero frontend code. |
| **Audio handling** | `ffmpeg` (system) + faster-whisper's built-in decoding | `pydub`, `librosa` | faster-whisper decodes common formats directly (via PyAV/ffmpeg), so we avoid an extra preprocessing dependency. `pydub` only added if we later need trimming/normalization. |
| **Packaging & distribution** | `uv` + host-installed Ollama (primary path); Docker + docker-compose (secondary path) | pip + venv, Docker-only, uv-only | Deliberate dual-path decision matched to the audience. **Primary:** non-technical students get the lightest install — `uv sync` plus the one-click Ollama installer. uv beats pip + venv on exactly this audience's failure modes: no separate Python install (uv provisions it), no venv-activation step (the classic Windows tripwire), and `uv.lock` makes the light path reproducible, not just the Docker path. Docker Desktop would add friction for exactly the users we're targeting. **Secondary:** a Dockerfile (app) + `docker-compose.yml` (app + `ollama/ollama` service, shared network, named volume for model weights) gives reviewers and power users a one-command reproducible stand-up of the full architecture, eliminating the ffmpeg/CTranslate2 "works on my machine" class of issues entirely. Trade-offs accepted: image stays lean by mounting a Whisper model cache volume instead of baking multi-GB weights into the image; GPU passthrough (NVIDIA Container Toolkit) documented but not required. |

### Prompt Design (the F3/F4 contract)

The Insight Engine sends a system + user message pair:

- **System:** role definition ("You are an academic meeting analyst…"), output schema (JSON with `summary`, `key_insights[]`, `action_items[]` where each action has `task`, `owner`, `priority`, `suggested_deadline`), tone constraints, instruction to only use information present in the transcript (anti-hallucination guardrail).
- **User:** the transcript, wrapped in delimiters, plus optional user-supplied context (e.g., "this was my thesis progress review").
- **Long-call handling:** transcripts exceeding the model's context window are chunked by Whisper segment boundaries, map-reduce summarized (chunk summaries → final synthesis pass).

---

## Phase 4: Timeline & Risk Mitigation

### Milestone Tracker

| Phase | Deliverable | Target |
|---|---|---|
| **P1 — Core Engine** | Transcription service wrapper + Insight Engine as pure-Python modules with unit tests; CLI entry point proving the pipeline end-to-end | Week 1–2 |
| **P2 — UI Integration** | Streamlit app: upload → progress states → rendered brief → Markdown export; model selector; error surfaces | Week 3 |
| **P3 — Hardening & Ship** | Structured-output retry logic, long-audio chunking, coverage report, GitHub Actions CI (lint + test), Dockerfile + docker-compose secondary install path (app + Ollama services), README case study, demo GIF | Week 4 |

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM returns malformed/unparseable output | Medium | High (breaks F4) | Structured output via Ollama's JSON/schema mode through LangChain; Pydantic validation; one automatic retry with a "fix your JSON" repair prompt; fall back to displaying raw LLM text. |
| CPU transcription too slow for long calls | Medium | Medium | Default to `small` INT8; enable faster-whisper's built-in VAD filter to skip silence; show honest ETA in UI; document GPU path. |
| Transcript exceeds LLM context window | High (60-min calls) | High | Segment-boundary chunking + map-reduce summarization; tested against a synthetic 60-min transcript. |
| Local model hallucinates action items not discussed | Medium | High (trust) | Prompt constraint "only from transcript"; render each action item with a quoted supporting snippet from the transcript so users can verify. |
| Ollama daemon not running / model not pulled | High (first-run) | Low | Startup health check against `localhost:11434`; friendly setup instructions rendered in-app, including exact `ollama pull` command. |
| Poor audio quality (phone speaker recording) | Medium | Medium | Document recording best practices; expose Whisper `beam_size` and model-size settings for a "high accuracy" mode. |
| Dockerized app can't reach Ollama (localhost vs container networking) | High (for Docker users) | Medium | `OLLAMA_BASE_URL` is env-driven, never hardcoded: compose sets `http://ollama:11434` (service name); docs cover `http://host.docker.internal:11434` for host-installed Ollama. Startup health check surfaces the exact URL it tried, so misconfiguration is self-diagnosing. |
| Docker image bloat from bundled model weights | Medium | Low | Never bake weights into the image: named volume for Ollama models, mounted HF cache volume for Whisper models; first-run download documented with sizes. |

---

## Phase 5: Execution Standards (the Evidence Layer)

- **Repo structure:** `app/` (Streamlit), `core/` (transcription.py, insights.py, prompts.py, schemas.py), `tests/`, `pyproject.toml` + `uv.lock`, `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- **Dependency management:** uv with `pyproject.toml` as the single manifest (runtime deps + a `dev` group for pytest/pytest-cov/ruff) and `uv.lock` committed for reproducible installs everywhere — local, CI, and Docker all install from the same lock.
- **Containerization:** multi-stage Dockerfile (builder runs `uv sync --frozen --no-dev` + installs ffmpeg → slim runtime image); compose defines `app` + `ollama` services on a shared network with a named volume for model weights; CI builds the image on every push to `main` so the tested code and the shipped image are the same artifact.
- **Git hygiene:** conventional commits — `feat: add faster-whisper transcription wrapper`, `feat: structured-output parsing with pydantic schema`, `test: chunking behaviour for >8k-token transcripts`.
- **Testing:** unit tests for prompt builder, output parser (valid/invalid/partial JSON), and chunker; integration test with a bundled 30-second sample clip; mock Ollama responses so CI never needs a live model.
- **CI/CD:** GitHub Actions on every push — `astral-sh/setup-uv` + `uv sync --frozen`, then `ruff` lint and `pytest` with coverage gate (≥ 80%), badge in README. (Deployment stays local by design — the "ship" artifact is a reproducible install, which itself is the differentiator.)
- **Config:** `.env.example` with `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `WHISPER_MODEL_SIZE`, `WHISPER_COMPUTE_TYPE`.

---

## Phase 6: Final Case Study Narrative (STAR skeleton)

- **Situation:** Students lose the value of supervisor calls because feedback and tasks live in un-reviewed audio; cloud tools are a privacy non-starter for academic content.
- **Task:** Build a fully local pipeline converting a call recording into a verified, structured action brief on consumer hardware.
- **Action:** Chose faster-whisper over vanilla Whisper for ~4x CPU inference speedup with INT8 quantization; used LangChain's `ChatOllama` with Pydantic-validated structured output to guarantee a machine-parseable brief; designed a map-reduce chunking strategy for hour-long transcripts; enforced quality with mocked-LLM CI at 80%+ coverage.
- **Result (targets to measure and report):** e.g. *"Transcribed 30-min calls in X min on a 4-core CPU (Y× faster than vanilla Whisper baseline); structured-output validity of Z% across N test calls; end-to-end call-to-brief time under M minutes."* Replace with your real measured numbers — the measurement itself is part of the story.
- **Developer Empathy Block:** two clearly-labeled install paths — **Path A (recommended for students):** 4-command quickstart (clone → `uv sync` → `ollama pull llama3.1:8b` → `uv run streamlit run app/main.py`); **Path B (reproducible/reviewer):** `docker compose up` standing up the full app + Ollama stack in one command. Plus `.env.example`, troubleshooting table (Ollama not running, ffmpeg missing, low RAM, `host.docker.internal` networking), and roadmap: speaker diarization (WhisperX), call history dashboard, calendar export, multilingual support. Case-study framing: *"Primary UX optimized for non-technical users; containerized path provided for reproducibility — packaging matched to the user, not to fashion."*

---

## Appendix: Dependency Shortlist

Declared in `pyproject.toml`, locked in `uv.lock` (managed with uv):

```
# [project.dependencies]
streamlit
faster-whisper          # STT — CTranslate2 runtime, models pulled from HF Hub
langchain
langchain-ollama        # ChatOllama, OllamaEmbeddings (if RAG added later)
langchain-core
pydantic                # output schemas
# [dependency-groups] dev
pytest, pytest-cov, ruff
# system: uv, ffmpeg, ollama daemon
# optional path B: Docker Desktop / Docker Engine + Compose
```

**Open question to resolve before P1:** default Ollama model. Candidates: `llama3.1:8b` (strong general summarization), `qwen2.5:7b` (strong instruction-following/JSON), `mistral:7b` (lighter). Benchmark all three on 2–3 sample transcripts and record the comparison in the case study — that mini-evaluation is itself portfolio evidence.

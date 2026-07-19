"""Benchmark harness for the README Results table. Dev tool — not shipped code.

Usage:
    uv run python scripts/benchmark.py transcribe PATH [--model-size small]
    uv run python scripts/benchmark.py e2e PATH
    uv run python scripts/benchmark.py validity [--runs 20]
    uv run --with openai-whisper python scripts/benchmark.py baseline PATH

CPU-bound commands (transcribe, e2e, baseline) should run one at a time so
timings aren't skewed by contention.
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter
from typing import Any

from core.chunking import DEFAULT_MAX_CHUNK_TOKENS, chunk_segments, estimate_tokens
from core.config import Settings
from core.export import export_markdown
from core.insights import InsightEngine, create_chat_model, create_insight_engine
from core.transcription import create_transcription_service

SHORT_TRANSCRIPT = (
    "Thanks for joining today's supervision meeting. I reviewed your draft over the weekend, "
    "and the results section is looking much stronger now. However, the methodology chapter "
    "still needs a clearer justification for your sampling approach. Please send me the "
    "revised methodology chapter by Friday. Also remember to upload your ethics approval "
    "form to the portal. Next month we will start planning the conference presentation."
)

MEDIUM_TRANSCRIPT = (
    "Supervisor: Let's go through your progress report. Your research questions are clear "
    "now, which the committee appreciated. The survey analysis was thorough, but I want to "
    "see confidence intervals reported alongside every point estimate, not just the "
    "significant ones. Student: I can add those this week. Supervisor: Good. The bigger "
    "issue is chapter four — your discussion doesn't yet connect the findings back to the "
    "framework you set up in chapter two. That link is what examiners look for. Student: "
    "So I should restructure around the three themes? Supervisor: Exactly, one section per "
    "theme. Also, book a session with the statistics consultant before you run the final "
    "regression models; I want the assumptions checked independently. Student: I'll email "
    "them today. Supervisor: Two deadlines to note: the revised chapter four is due on the "
    "fifteenth, and your conference abstract must be submitted by the end of the month. "
    "The abstract is high priority — this conference matters for your examiners' network. "
    "Student: Understood. Supervisor: Finally, well done on the ethics amendment; it was "
    "approved without changes. Keep your research journal up to date, it will pay off when "
    "you write the methodology defence."
)

LONG_TRANSCRIPT = (
    MEDIUM_TRANSCRIPT
    + " "
    + (
        "Supervisor: One more area — your literature review. It currently reads as a summary "
        "rather than an argument. I marked six places where you describe a study but never say "
        "why it matters to your question. Rework those into critical engagement: what did they "
        "get wrong, what gap remains, and how does your design address it. Student: That's "
        "fair, I wrote most of it in first year. Supervisor: It shows, and that's normal. "
        "Allocate two full weeks to it after chapter four. Do not start the rewrite before the "
        "conference abstract is in. Student: Should I send you sections as I go? Supervisor: "
        "Send the first rewritten section as a sample before doing the rest, so we catch style "
        "issues early. Also consider presenting the pilot results at the departmental seminar "
        "in March — low stakes, good practice for the viva. Student: I'd like that. "
        "Supervisor: I'll put your name down. Last thing: your funding report is due to the "
        "graduate school next quarter; start collecting the training records now because "
        "chasing certificates in the final week is miserable. Student: Noted, I'll make a "
        "checklist. Supervisor: Excellent. Same time in two weeks."
    )
)

VALIDITY_TRANSCRIPTS = [SHORT_TRANSCRIPT, MEDIUM_TRANSCRIPT, LONG_TRANSCRIPT]


class CountingModel:
    """Wraps the real chat model and counts invocations (1 = first attempt, 2 = retried)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.invocations = 0

    def invoke(self, messages: Any) -> Any:
        self.invocations += 1
        return self._inner.invoke(messages)


def cmd_transcribe(args: argparse.Namespace) -> None:
    settings = Settings.from_env().model_copy(update={"whisper_model_size": args.model_size})

    t0 = perf_counter()
    service = create_transcription_service(settings)
    t1 = perf_counter()
    transcript = service.transcribe(args.audio)
    t2 = perf_counter()

    audio_seconds = transcript.segments[-1].end if transcript.segments else 0.0
    tokens = estimate_tokens(transcript.text)
    chunks = len(chunk_segments(transcript.segments, DEFAULT_MAX_CHUNK_TOKENS))
    print(f"model:            {settings.whisper_model_size} / {settings.whisper_compute_type}")
    print(f"model load:       {t1 - t0:.1f}s")
    print(f"transcription:    {t2 - t1:.1f}s ({_mmss(t2 - t1)})")
    print(f"audio duration:   {_mmss(audio_seconds)}")
    print(f"speed:            {audio_seconds / (t2 - t1):.1f}x realtime")
    print(
        f"transcript:       {len(transcript.segments)} segments, ~{tokens} tokens, "
        f"{chunks} chunk(s) at {DEFAULT_MAX_CHUNK_TOKENS} tokens/chunk"
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("Run with: uv run --with openai-whisper python scripts/benchmark.py baseline …")

    t0 = perf_counter()
    model = whisper.load_model(args.model_size)
    t1 = perf_counter()
    result = model.transcribe(args.audio, fp16=False)
    t2 = perf_counter()

    audio_seconds = result["segments"][-1]["end"] if result["segments"] else 0.0
    print(f"vanilla whisper {args.model_size} (fp32 CPU)")
    print(f"model load:       {t1 - t0:.1f}s")
    print(f"transcription:    {t2 - t1:.1f}s ({_mmss(t2 - t1)})")
    print(f"speed:            {audio_seconds / (t2 - t1):.1f}x realtime")


def cmd_e2e(args: argparse.Namespace) -> None:
    settings = Settings.from_env()

    t0 = perf_counter()
    transcript = create_transcription_service(settings).transcribe(args.audio)
    t1 = perf_counter()
    result = create_insight_engine(settings).generate_brief(transcript)
    export_markdown(result, transcript)
    t2 = perf_counter()

    print(f"transcription:    {t1 - t0:.1f}s")
    print(f"brief generation: {t2 - t1:.1f}s (structured: {result.succeeded})")
    print(f"end-to-end:       {t2 - t0:.1f}s ({_mmss(t2 - t0)})")


def cmd_briefs(args: argparse.Namespace) -> None:
    """One brief per sample transcript — for qualitative review (faithfulness, recall)."""
    settings = Settings.from_env()
    if args.model:
        settings = settings.model_copy(update={"ollama_model": args.model})

    for name, transcript in zip(["short", "medium", "long"], VALIDITY_TRANSCRIPTS, strict=True):
        engine = create_insight_engine(settings)
        t0 = perf_counter()
        result = engine.generate_brief(transcript)
        elapsed = perf_counter() - t0
        print(f"\n=== {name} transcript · {settings.ollama_model} · {elapsed:.1f}s ===")
        if result.brief is not None:
            print(result.brief.model_dump_json(indent=2))
        else:
            print(f"STRUCTURED PARSING FAILED. Raw output:\n{result.raw_text}")


def cmd_validity(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if args.model:
        settings = settings.model_copy(update={"ollama_model": args.model})
    first_attempt_ok = 0
    with_retry_ok = 0

    for run in range(args.runs):
        transcript = VALIDITY_TRANSCRIPTS[run % len(VALIDITY_TRANSCRIPTS)]
        model = CountingModel(create_chat_model(settings))
        result = InsightEngine(model).generate_brief(transcript)
        if result.succeeded and model.invocations == 1:
            first_attempt_ok += 1
        if result.succeeded:
            with_retry_ok += 1
        print(
            f"run {run + 1:>2}/{args.runs}: attempts={model.invocations} ok={result.succeeded}",
            flush=True,
        )

    print(f"\nmodel: {settings.ollama_model}, runs: {args.runs}")
    print(f"first-attempt validity: {100 * first_attempt_ok / args.runs:.0f}%")
    print(f"with one retry:         {100 * with_retry_ok / args.runs:.0f}%")


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}m {seconds % 60:02.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_transcribe = sub.add_parser("transcribe", help="time faster-whisper on an audio file")
    p_transcribe.add_argument("audio")
    p_transcribe.add_argument("--model-size", default="small")
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_baseline = sub.add_parser("baseline", help="time vanilla openai-whisper (same audio)")
    p_baseline.add_argument("audio")
    p_baseline.add_argument("--model-size", default="small")
    p_baseline.set_defaults(func=cmd_baseline)

    p_e2e = sub.add_parser("e2e", help="time the full pipeline: audio → markdown brief")
    p_e2e.add_argument("audio")
    p_e2e.set_defaults(func=cmd_e2e)

    p_validity = sub.add_parser("validity", help="structured-output validity over N runs")
    p_validity.add_argument("--runs", type=int, default=20)
    p_validity.add_argument("--model", help="override OLLAMA_MODEL for model comparison")
    p_validity.set_defaults(func=cmd_validity)

    p_briefs = sub.add_parser("briefs", help="one brief per sample transcript (for eval)")
    p_briefs.add_argument("--model", help="override OLLAMA_MODEL for model comparison")
    p_briefs.set_defaults(func=cmd_briefs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

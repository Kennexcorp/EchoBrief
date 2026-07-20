"""Command-line entry point: audio file → Markdown brief, end to end.

Progress and warnings go to stderr; the brief itself goes to stdout (or a
file via --output), so the output stays pipeable.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from core.config import Settings
from core.export import export_markdown
from core.health import HealthStatus, check_ollama
from core.insights import InsightEngine, create_insight_engine
from core.speech import SpeechClient, SpeechError, create_speech_client, synthesize_brief
from core.transcription import TranscriptionService, create_transcription_service


def main(
    argv: list[str] | None = None,
    *,
    health_check: Callable[[Settings], HealthStatus] = check_ollama,
    transcription_factory: Callable[[Settings], TranscriptionService] = (
        create_transcription_service
    ),
    engine_factory: Callable[[Settings], InsightEngine] = create_insight_engine,
    speech_factory: Callable[[Settings], SpeechClient] = create_speech_client,
) -> int:
    parser = argparse.ArgumentParser(
        prog="echobrief",
        description="Turn a recorded call into a structured Markdown brief, fully locally.",
    )
    parser.add_argument("audio", help="path to the recording (.mp3, .wav, .m4a)")
    parser.add_argument("--context", help="optional context, e.g. 'thesis progress review'")
    parser.add_argument("--output", "-o", help="write the brief to this file instead of stdout")
    parser.add_argument(
        "--speak",
        metavar="MP3_PATH",
        help="also read the brief aloud, saving spoken audio to this MP3 "
        "(opt-in; needs ELEVENLABS_API_KEY, the only feature that leaves the machine)",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()

    status = health_check(settings)
    if not status.ok:
        print(status.message, file=sys.stderr)
        return 1

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    print("Transcribing (this can take a while on CPU)...", file=sys.stderr)
    transcript = transcription_factory(settings).transcribe(audio_path)

    print(
        f"Transcribed {len(transcript.segments)} segments. "
        f"Generating brief with {settings.ollama_model}...",
        file=sys.stderr,
    )
    result = engine_factory(settings).generate_brief(transcript, user_context=args.context)
    if not result.succeeded:
        print(
            "Warning: structured parsing failed; exporting the raw model output.",
            file=sys.stderr,
        )

    markdown = export_markdown(result, transcript)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"Brief written to {args.output}", file=sys.stderr)
    else:
        print(markdown)

    if args.speak:
        if result.brief is None:
            print(
                "Skipping audio: no structured brief to read aloud (structured parsing failed).",
                file=sys.stderr,
            )
            return 1
        try:
            audio = synthesize_brief(result.brief, speech_factory(settings))
        except SpeechError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        Path(args.speak).write_bytes(audio)
        print(f"Spoken brief written to {args.speak}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

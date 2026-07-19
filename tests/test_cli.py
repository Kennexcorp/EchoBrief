"""Tests for core.cli — the end-to-end command-line entry point."""

from pathlib import Path

import pytest

from core.cli import main
from core.health import HealthStatus
from core.schemas import Brief, InsightResult, Transcript, TranscriptSegment

TRANSCRIPT = Transcript(
    segments=[TranscriptSegment(text="Send the chapter by Friday.", start=0.0, end=2.0)]
)

RESULT = InsightResult(
    brief=Brief(
        summary="The supervisor requested a revised chapter.",
        key_insights=["Deadline discipline matters."],
        action_items=[],
    )
)


class FakeTranscriptionService:
    def transcribe(self, audio_path: Path) -> Transcript:
        return TRANSCRIPT


class FakeEngine:
    def __init__(self, result: InsightResult = RESULT) -> None:
        self._result = result
        self.received_context: str | None = None

    def generate_brief(self, transcript: Transcript, user_context: str | None = None):
        self.received_context = user_context
        return self._result


def healthy(settings, **kwargs) -> HealthStatus:
    return HealthStatus(ok=True, message="ok")


def unhealthy(settings, **kwargs) -> HealthStatus:
    return HealthStatus(ok=False, message="Ollama is not reachable at http://localhost:11434")


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "call.mp3"
    path.write_bytes(b"fake audio")
    return path


def run_cli(argv: list[str], **overrides) -> int:
    defaults = {
        "health_check": healthy,
        "transcription_factory": lambda settings: FakeTranscriptionService(),
        "engine_factory": lambda settings: FakeEngine(),
    }
    defaults.update(overrides)
    return main(argv, **defaults)


class TestHealthGate:
    def test_unreachable_ollama_exits_nonzero_with_message(
        self, audio_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run_cli([str(audio_file)], health_check=unhealthy)

        assert code == 1
        assert "not reachable" in capsys.readouterr().err


class TestHappyPath:
    def test_prints_markdown_brief_to_stdout(
        self, audio_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run_cli([str(audio_file)])

        captured = capsys.readouterr()
        assert code == 0
        assert "## Summary" in captured.out
        assert "The supervisor requested a revised chapter." in captured.out

    def test_output_flag_writes_file_instead(
        self, audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_file = tmp_path / "brief.md"

        code = run_cli([str(audio_file), "--output", str(out_file)])

        assert code == 0
        assert "## Summary" in out_file.read_text()
        assert "## Summary" not in capsys.readouterr().out

    def test_context_flag_reaches_the_engine(self, audio_file: Path) -> None:
        engine = FakeEngine()

        code = run_cli(
            [str(audio_file), "--context", "thesis review"],
            engine_factory=lambda settings: engine,
        )

        assert code == 0
        assert engine.received_context == "thesis review"


class TestBadInput:
    def test_missing_audio_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = run_cli([str(tmp_path / "nope.mp3")])

        assert code == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_fallback_result_still_exports_with_warning(
        self, audio_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fallback = InsightResult(brief=None, raw_text="raw model text")

        code = run_cli(
            [str(audio_file)],
            engine_factory=lambda settings: FakeEngine(result=fallback),
        )

        captured = capsys.readouterr()
        assert code == 0
        assert "raw model text" in captured.out
        assert "parsing failed" in captured.err.lower()

"""Smoke tests for the Streamlit UI via streamlit's AppTest.

File uploads can't be scripted through AppTest, so the full processing flow
is covered by the core-layer tests and the CLI; here we verify the page
renders, the health gate blocks correctly, and the model selector works.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.health import HealthStatus
from core.schemas import Brief, InsightResult, Transcript, TranscriptSegment

APP_PATH = str(Path(__file__).parent.parent / "app" / "main.py")

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


@pytest.fixture
def healthy_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub .env loading so tests never pick up a developer's real key.
    monkeypatch.setattr("core.config.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "core.health.check_ollama",
        lambda settings, **kwargs: HealthStatus(ok=True, message="ok"),
    )
    monkeypatch.setattr(
        "core.health.list_models",
        lambda settings, **kwargs: ["llama3.1:8b", "qwen2.5:7b"],
    )


class TestHealthyApp:
    def test_renders_without_exception(self, healthy_ollama: None) -> None:
        at = AppTest.from_file(APP_PATH).run(timeout=15)

        assert not at.exception
        assert "EchoBrief" in at.title[0].value

    def test_model_selector_lists_available_models(self, healthy_ollama: None) -> None:
        at = AppTest.from_file(APP_PATH).run(timeout=15)

        assert at.selectbox[0].options == ["llama3.1:8b", "qwen2.5:7b"]
        assert at.selectbox[0].value == "llama3.1:8b"  # configured default pre-selected


class TestVoiceOutput:
    def _run_with_brief(self) -> AppTest:
        at = AppTest.from_file(APP_PATH)
        at.session_state["result"] = RESULT
        at.session_state["transcript"] = TRANSCRIPT
        return at.run(timeout=15)

    def test_read_aloud_button_shown_when_key_set(
        self, healthy_ollama: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")

        at = self._run_with_brief()

        assert not at.exception
        assert any("Read the brief aloud" in button.label for button in at.button)

    def test_hint_shown_and_no_button_without_key(
        self, healthy_ollama: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        at = self._run_with_brief()

        assert not at.exception
        assert not any("Read the brief aloud" in button.label for button in at.button)
        assert any("ELEVENLABS_API_KEY" in caption.value for caption in at.caption)


class TestUnhealthyApp:
    def test_health_failure_shows_remediation_and_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "core.health.check_ollama",
            lambda settings, **kwargs: HealthStatus(
                ok=False, message="Ollama is not reachable at http://localhost:11434"
            ),
        )

        at = AppTest.from_file(APP_PATH).run(timeout=15)

        assert not at.exception
        assert any("not reachable" in err.value for err in at.error)
        assert not at.selectbox  # page stopped before the main UI

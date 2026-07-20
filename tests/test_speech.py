"""Tests for core.speech — the optional ElevenLabs voice-output feature.

No test here makes a network call: the pure ``brief_to_script`` is exercised
directly, and the CLI path is driven with an injected fake client.
"""

import urllib.error

import pytest

from core.config import Settings
from core.schemas import ActionItem, Brief
from core.speech import (
    ElevenLabsClient,
    SpeechError,
    brief_to_script,
    create_speech_client,
    synthesize_brief,
)

BRIEF = Brief(
    summary="The supervisor asked for a revised chapter and clearer figures.",
    key_insights=["Deadline discipline matters", "Figures need captions"],
    action_items=[
        ActionItem(
            task="Send the revised chapter",
            owner="me",
            priority="high",
            suggested_deadline="Friday",
            supporting_quote="Please get the chapter to me by Friday.",
        )
    ],
)


class TestBriefToScript:
    def test_includes_summary_insights_and_action_items(self) -> None:
        script = brief_to_script(BRIEF)

        assert "The supervisor asked for a revised chapter" in script
        assert "Deadline discipline matters." in script
        assert "Send the revised chapter." in script
        assert "Owner: me." in script

    def test_omits_supporting_quotes_from_audio(self) -> None:
        # Quotes live in the text brief; they would make the audio unlistenable.
        assert "Please get the chapter to me by Friday" not in brief_to_script(BRIEF)

    def test_states_when_there_are_no_action_items(self) -> None:
        brief = BRIEF.model_copy(update={"action_items": []})

        assert "No action items were discussed." in brief_to_script(brief)


class TestConfigGate:
    def test_tts_disabled_without_api_key(self) -> None:
        assert Settings().tts_enabled is False

    def test_tts_enabled_with_api_key(self) -> None:
        assert Settings(elevenlabs_api_key="sk-test").tts_enabled is True

    def test_create_client_raises_when_disabled(self) -> None:
        with pytest.raises(SpeechError, match="Voice output is off"):
            create_speech_client(Settings())

    def test_create_client_builds_elevenlabs_client_when_enabled(self) -> None:
        client = create_speech_client(Settings(elevenlabs_api_key="sk-test"))

        assert isinstance(client, ElevenLabsClient)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestElevenLabsClient:
    def test_posts_and_returns_audio_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["key"] = request.headers.get("Xi-api-key")
            captured["body"] = request.data
            return FakeResponse(b"ID3-mp3-bytes")

        monkeypatch.setattr("core.speech.urllib.request.urlopen", fake_urlopen)
        client = ElevenLabsClient(api_key="sk-test", voice_id="voice123", model="eleven_x")

        audio = client.synthesize("hello world")

        assert audio == b"ID3-mp3-bytes"
        assert captured["url"].endswith("/text-to-speech/voice123")
        assert captured["key"] == "sk-test"
        assert b"hello world" in captured["body"]

    def test_empty_text_raises_before_any_request(self) -> None:
        client = ElevenLabsClient(api_key="sk-test", voice_id="v", model="m")

        with pytest.raises(SpeechError, match="Nothing to synthesize"):
            client.synthesize("   ")

    def test_bad_api_key_gives_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr("core.speech.urllib.request.urlopen", fake_urlopen)
        client = ElevenLabsClient(api_key="bad", voice_id="v", model="m")

        with pytest.raises(SpeechError, match="ELEVENLABS_API_KEY"):
            client.synthesize("hi")

    def test_quota_exceeded_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

        monkeypatch.setattr("core.speech.urllib.request.urlopen", fake_urlopen)
        client = ElevenLabsClient(api_key="k", voice_id="v", model="m")

        with pytest.raises(SpeechError, match="quota exceeded"):
            client.synthesize("hi")

    def test_other_http_error_includes_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        def fake_urlopen(request, timeout=0):
            raise urllib.error.HTTPError(
                request.full_url, 500, "Server Error", {}, io.BytesIO(b"boom")
            )

        monkeypatch.setattr("core.speech.urllib.request.urlopen", fake_urlopen)
        client = ElevenLabsClient(api_key="k", voice_id="v", model="m")

        with pytest.raises(SpeechError, match="HTTP 500"):
            client.synthesize("hi")

    def test_unreachable_api_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request, timeout=0):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("core.speech.urllib.request.urlopen", fake_urlopen)
        client = ElevenLabsClient(api_key="k", voice_id="v", model="m")

        with pytest.raises(SpeechError, match="Could not reach"):
            client.synthesize("hi")


class RecordingClient:
    def __init__(self) -> None:
        self.spoken: str | None = None

    def synthesize(self, text: str) -> bytes:
        self.spoken = text
        return b"mp3"


class TestSynthesizeBrief:
    def test_renders_then_delegates_to_client(self) -> None:
        client = RecordingClient()

        audio = synthesize_brief(BRIEF, client)

        assert audio == b"mp3"
        assert client.spoken is not None
        assert "Summary." in client.spoken

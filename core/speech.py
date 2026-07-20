"""Optional spoken-audio output for a brief (F6), via the ElevenLabs API.

This is the one feature that leaves the machine. It stays off unless an
``ELEVENLABS_API_KEY`` is set, so the default pipeline remains 100% local. The
brief-to-script step is a pure function (network-free, fully unit-tested); the
HTTP call is isolated behind ``SpeechClient`` and injected, matching how the
rest of the pipeline is wired.

Uses stdlib ``urllib`` for the request, so this adds no third-party dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from core.config import Settings
from core.schemas import Brief

_TTS_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TIMEOUT_SECONDS = 60


class SpeechError(RuntimeError):
    """Raised when audio synthesis cannot be completed. Message states the fix."""


def brief_to_script(brief: Brief) -> str:
    """Render a ``Brief`` as plain narration text, ready to be spoken aloud.

    Markdown and verbatim quotes are dropped: this is meant for the ear, so the
    supporting quotes (which live in the text/Markdown brief) are left out to
    keep the audio listenable.
    """
    lines = ["Here is your EchoBrief for this call.", "", "Summary.", brief.summary]

    if brief.key_insights:
        lines += ["", "Key insights."]
        lines += [insight.rstrip(".") + "." for insight in brief.key_insights]

    lines += ["", "Action items."]
    if not brief.action_items:
        lines.append("No action items were discussed.")
    else:
        for number, item in enumerate(brief.action_items, start=1):
            lines.append(
                f"{number}. {item.task.rstrip('.')}. "
                f"Owner: {item.owner}. "
                f"Priority: {item.priority}. "
                f"Suggested deadline: {item.suggested_deadline}."
            )
    return "\n".join(lines)


class SpeechClient(Protocol):
    """The single capability the CLI/UI needs: text in, MP3 bytes out."""

    def synthesize(self, text: str) -> bytes: ...


class ElevenLabsClient:
    """Turns text into MP3 audio through the ElevenLabs text-to-speech API."""

    def __init__(self, api_key: str, voice_id: str, model: str) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise SpeechError("Nothing to synthesize: the script text is empty.")

        url = _TTS_ENDPOINT.format(voice_id=self._voice_id)
        body = json.dumps(
            {
                "text": text,
                "model_id": self._model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            if exc.code in (401, 403):
                raise SpeechError(
                    "ElevenLabs rejected the API key (HTTP "
                    f"{exc.code}). Check ELEVENLABS_API_KEY in your .env."
                ) from exc
            if exc.code == 429:
                raise SpeechError(
                    "ElevenLabs quota exceeded (HTTP 429). The free tier resets monthly; "
                    "wait or upgrade the plan."
                ) from exc
            raise SpeechError(f"ElevenLabs request failed (HTTP {exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise SpeechError(
                f"Could not reach the ElevenLabs API: {exc.reason}. Check your connection."
            ) from exc


def create_speech_client(settings: Settings) -> SpeechClient:
    """Build a speech client from settings; raises if voice output is not configured."""
    if not settings.tts_enabled:
        raise SpeechError(
            "Voice output is off. Set ELEVENLABS_API_KEY in your .env to enable it "
            "(free tier: https://elevenlabs.io)."
        )
    return ElevenLabsClient(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_model,
    )


def synthesize_brief(brief: Brief, client: SpeechClient) -> bytes:
    """Convenience: render a brief to a script and synthesize it to MP3 bytes."""
    return client.synthesize(brief_to_script(brief))

"""EchoBrief Streamlit UI: a thin presentation layer over the core pipeline.

No business logic lives here: this file renders widgets, hands files to the
core services, and displays what they return.
"""

import tempfile
from pathlib import Path

import streamlit as st

from core import health, insights, transcription
from core.config import Settings
from core.export import export_markdown
from core.schemas import InsightResult, Transcript

ACCEPTED_TYPES = ["mp3", "wav", "m4a", "opus", "ogg"]


def main() -> None:
    st.set_page_config(page_title="EchoBrief", page_icon="🎙️")
    st.title("🎙️ EchoBrief")
    st.caption(
        "Turn recorded calls into summaries, insights, and tracked next steps, "
        "100% locally. No audio ever leaves your machine."
    )

    settings = Settings.from_env()

    status = health.check_ollama(settings)
    if not status.ok:
        st.error(status.message)
        st.markdown(
            "**Setup:** install [Ollama](https://ollama.com/download), start it, "
            f"then run `ollama pull {settings.ollama_model}` and refresh this page."
        )
        st.stop()

    models = health.list_models(settings)
    default_index = models.index(settings.ollama_model) if settings.ollama_model in models else 0
    chosen_model = st.selectbox("Ollama model", models, index=default_index)
    settings = settings.model_copy(update={"ollama_model": chosen_model})

    uploaded = st.file_uploader(
        "Upload a call recording",
        type=ACCEPTED_TYPES,
        help="Up to ~60 minutes / 200 MB. The file is processed locally and deleted afterwards.",
    )
    user_context = st.text_input(
        "Optional context", placeholder="e.g. thesis progress review with my supervisor"
    )

    if uploaded is not None and st.button("Generate brief", type="primary"):
        with st.status("Processing...", expanded=True) as progress:
            st.write("Transcribing locally. This can take a while on CPU...")
            transcript = _transcribe_upload(uploaded, settings)
            st.write(
                f"Transcribed {len(transcript.segments)} segments. "
                f"Generating brief with {settings.ollama_model}..."
            )
            result = insights.create_insight_engine(settings).generate_brief(
                transcript, user_context=user_context or None
            )
            progress.update(label="Done", state="complete", expanded=False)
        st.session_state["transcript"] = transcript
        st.session_state["result"] = result

    if "result" in st.session_state:
        _render_result(st.session_state["result"], st.session_state["transcript"])


def _transcribe_upload(uploaded, settings: Settings) -> Transcript:
    service = _cached_transcription_service(
        settings.whisper_model_size, settings.whisper_compute_type
    )
    suffix = Path(uploaded.name).suffix or ".mp3"
    # NamedTemporaryFile deletes on close, so uploaded audio never outlives the run (privacy NFR).
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        return service.transcribe(tmp.name)


@st.cache_resource
def _cached_transcription_service(model_size: str, compute_type: str):
    """Keep the loaded Whisper model across reruns; keyed on the settings that matter."""
    return transcription.create_transcription_service(
        Settings(whisper_model_size=model_size, whisper_compute_type=compute_type)
    )


def _render_result(result: InsightResult, transcript: Transcript) -> None:
    if result.succeeded and result.brief is not None:
        brief = result.brief
        st.subheader("Summary")
        st.write(brief.summary)

        st.subheader("Key Insights")
        for insight in brief.key_insights:
            st.markdown(f"- {insight}")

        st.subheader("Action Items")
        if not brief.action_items:
            st.markdown("_No action items were discussed._")
        for number, item in enumerate(brief.action_items, start=1):
            st.markdown(f"**{number}. {item.task}**")
            st.caption(
                f"Priority: {item.priority} · Owner: {item.owner} · "
                f"Deadline: {item.suggested_deadline}"
            )
            st.markdown(f"> {item.supporting_quote}")
    else:
        st.warning("Structured parsing failed. Showing the model's raw output.")
        st.text(result.raw_text)

    with st.expander("Full transcript"):
        st.write(transcript.text)

    st.download_button(
        "⬇️ Download brief (Markdown)",
        data=export_markdown(result, transcript),
        file_name="echobrief.md",
        mime="text/markdown",
    )


main()

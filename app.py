import io
import os
import tempfile
import hashlib
from datetime import datetime

import requests
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


TITLE = "VoiceGuard AI"
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
ASR_ID = "openai/whisper-tiny"


def tinyfish_search(query):
    """Search the web using the configured TinyFish API key."""
    try:
        api_key = st.secrets.get("TINYFISH_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("TINYFISH_API_KEY")

    if not api_key:
        st.error(
            "TinyFish API key nahi mili. Project folder mein "
            ".streamlit/secrets.toml check karo."
        )
        return

    try:
        response = requests.get(
            "https://api.search.tinyfish.ai",
            params={"query": query},
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Support common response shapes.
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results and isinstance(data, dict):
            results = data.get("data", [])

        if not results:
            st.info("Koi search result nahi mila, ya API response ka format alag hai.")
            with st.expander("API response (debug)"):
                st.json(data)
            return

        for item in results:
            if not isinstance(item, dict):
                st.write(item)
                continue

            title = item.get("title") or item.get("name") or "Untitled result"
            snippet = (
                item.get("snippet")
                or item.get("description")
                or item.get("content")
                or ""
            )
            url = item.get("url") or item.get("link") or ""

            st.markdown(f"### {title}")
            if snippet:
                st.write(snippet)
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.markdown(f"[Open source]({url})")
            st.divider()

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        st.error(f"TinyFish API HTTP error: {status}")
        if exc.response is not None:
            st.code(exc.response.text[:1500])
    except requests.RequestException as exc:
        st.error(f"Internet/API connection error: {exc}")
    except ValueError:
        st.error("API ne valid JSON response nahi diya.")
    except Exception as exc:
        st.error(f"Search failed: {exc}")


st.set_page_config(page_title=TITLE, page_icon="🛡️", layout="wide")

st.title("🛡️ VoiceGuard AI")
st.caption("Synthetic-speech screening assistant · English interface")

FEATURES = [
    "AI Voice Detection",
    "Real vs Fake Score Meter",
    "Clear Screening Result",
    "Audio Characteristics",
    "Waveform Visualization",
    "Speech-to-Text Transcript",
    "PDF Report + TinyFish Web Search",
]
with st.expander("✨ 7 Project Features", expanded=True):
    feature_cols = st.columns(2)
    for index, feature in enumerate(FEATURES):
        feature_cols[index % 2].markdown(f"**{index + 1}.** {feature}")

if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None


with st.sidebar:
    st.header("Privacy & limitations")
    st.write(
        "Audio is processed by this app. Temporary decoding files are deleted "
        "after processing, but this does not guarantee zero retention by a "
        "hosting provider, browser, operating system, or network."
    )
    st.write(
        "Do not upload sensitive audio or recordings you lack permission to "
        "analyze. Model scores are not calibrated probabilities, identity "
        "verification, or forensic proof."
    )
    if st.button("Clear session history"):
        st.session_state.history = []
        st.session_state.last = None
        st.success("History cleared for this session.")


@st.cache_resource(show_spinner=False)
def load_detector():
    extractor = AutoFeatureExtractor.from_pretrained(DETECTOR_ID)
    model = AutoModelForAudioClassification.from_pretrained(DETECTOR_ID)
    model.eval()
    return extractor, model


@st.cache_resource(show_spinner=False)
def load_asr():
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=ASR_ID,
        device=device,
    )


def metrics(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "Duration (seconds)": round(len(y) / sr, 2),
        "Sample rate (Hz)": int(sr),
        "RMS energy": round(float(np.sqrt(np.mean(y**2))), 6),
        "Zero-crossing rate": round(
            float(np.mean(librosa.feature.zero_crossing_rate(y=y))), 6
        ),
        "Spectral centroid (Hz)": round(
            float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 2
        ),
        "Spectral rolloff (Hz)": round(
            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 2
        ),
        "Mean MFCC variance": round(float(np.mean(np.var(mfcc, axis=1))), 4),
    }


def make_pdf(result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("VoiceGuard AI — Audio Screening Report", styles["Title"]),
        Paragraph("Created: " + result["created_at"], styles["Normal"]),
        Paragraph("File: " + result["filename"], styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Model output (not proof)", styles["Heading2"]),
        Paragraph(
            "Scores are raw model outputs, not calibrated probabilities. "
            "This report cannot verify identity or prove that audio is genuine "
            "or synthetic.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    rows = [["Model label", "Score (%)"]]
    rows += [
        [item["label"], f'{item["score"] * 100:.2f}%']
        for item in result.get("scores", [])
    ]
    score_table = Table(rows, colWidths=[330, 120])
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story += [
        score_table,
        Spacer(1, 12),
        Paragraph("Audio characteristics", styles["Heading2"]),
    ]
    rows2 = [["Metric", "Value"]] + [
        [key, str(value)] for key, value in result.get("metrics", {}).items()
    ]
    metrics_table = Table(rows2, colWidths=[280, 170])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    transcript = (
        result.get("transcript") or "Not generated."
    ).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story += [
        metrics_table,
        Spacer(1, 12),
        Paragraph("Transcript", styles["Heading2"]),
        Paragraph(transcript, styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            "False positives and false negatives are possible. Results may not "
            "generalize across languages, devices, codecs, or synthesis methods. "
            "Do not rely on this report alone for important decisions.",
            styles["BodyText"],
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


uploaded = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
)

if uploaded:
    raw = uploaded.getvalue()
    file_id = hashlib.sha256(raw).hexdigest()[:12]
    st.audio(raw)
    st.caption(
        f"File size: {len(raw) / 1024:.1f} KB · Session file ID: {file_id}"
    )

    col1, col2 = st.columns(2)
    detect = col1.button(
        "Analyze voice",
        type="primary",
        use_container_width=True,
    )
    transcribe = col2.button(
        "Generate transcript (optional)",
        use_container_width=True,
    )

    if detect:
        temp_path = None
        try:
            with st.spinner(
                "Loading detector and analyzing audio. "
                "First run downloads model files..."
            ):
                suffix = os.path.splitext(uploaded.name)[1] or ".audio"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(raw)
                    temp_path = temp_file.name

                y, sr = librosa.load(
                    temp_path,
                    sr=16000,
                    mono=True,
                    duration=20,
                )

                if len(y) < int(sr * 0.5):
                    st.error("Audio is too short. Please upload at least 0.5 seconds.")
                else:
                    extractor, model = load_detector()
                    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")

                    with torch.no_grad():
                        logits = model(**inputs).logits
                        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                    labels = getattr(model.config, "id2label", {}) or {}
                    scores = [
                        {
                            "label": str(
                                labels.get(
                                    i,
                                    labels.get(str(i), f"Class {i}"),
                                )
                            ),
                            "score": float(probability),
                        }
                        for i, probability in enumerate(probabilities)
                    ]
                    scores.sort(key=lambda item: item["score"], reverse=True)

                    result = {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "file_id": file_id,
                        "model_id": DETECTOR_ID,
                        "scores": scores,
                        "metrics": metrics(y, sr),
                        "transcript": "",
                    }
                    st.session_state.last = result
                    st.session_state.history.insert(0, result.copy())
                    st.success("Analysis complete.")

        except Exception as exc:
            st.error(
                "Analysis failed. Check internet access for the first model "
                "download and review the terminal error."
            )
            st.exception(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if transcribe:
        temp_path = None
        try:
            with st.spinner(
                "Loading Whisper Tiny. First use downloads a separate model..."
            ):
                suffix = os.path.splitext(uploaded.name)[1] or ".audio"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(raw)
                    temp_path = temp_file.name

                transcript = load_asr()(temp_path).get("text", "").strip()

                if (
                    st.session_state.last
                    and st.session_state.last.get("file_id") == file_id
                ):
                    st.session_state.last["transcript"] = transcript
                    for item in st.session_state.history:
                        if item.get("file_id") == file_id:
                            item["transcript"] = transcript
                else:
                    result = {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "file_id": file_id,
                        "model_id": "Whisper Tiny transcript only",
                        "scores": [],
                        "metrics": {},
                        "transcript": transcript,
                    }
                    st.session_state.last = result
                    st.session_state.history.insert(0, result.copy())

                st.success(
                    "Transcript generated. Speech recognition can make mistakes."
                )

        except Exception as exc:
            st.error(
                "Transcript generation failed. Check internet access and the "
                "terminal error."
            )
            st.exception(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


if st.session_state.last:
    result = st.session_state.last
    st.divider()
    st.header("Results & explanation")

    if result.get("scores"):
        st.subheader("Model score breakdown")
        for score_item in result["scores"]:
            st.write(
                f'**{score_item["label"]}** — {score_item["score"] * 100:.2f}%'
            )
            st.progress(max(0.0, min(1.0, score_item["score"])))

        top_score = result["scores"][0]
        st.info(
            f'Top model label: **{top_score["label"]}**. '
            "This is the model’s strongest class, not certainty."
        )

        if any(
            word in top_score["label"].lower()
            for word in ("fake", "spoof", "synthetic", "generated", "deepfake", "ai")
        ):
            st.warning(
                "Top label appears to indicate synthetic/fake/spoof speech. "
                "Treat it as a warning, not proof."
            )
        else:
            st.info(
                "Top label does not appear to indicate synthetic speech. "
                "That does not prove the audio is genuine."
            )
    else:
        st.info("Transcript-only result: detection was not run for this item.")

    st.subheader("Audio characteristics")
    if result.get("metrics"):
        columns = st.columns(3)
        for index, (key, value) in enumerate(result["metrics"].items()):
            columns[index % 3].metric(key, value)
    else:
        st.caption("Run Analyze voice to calculate audio characteristics.")

    if uploaded and uploaded.name == result["filename"]:
        try:
            waveform, waveform_sr = librosa.load(
                io.BytesIO(uploaded.getvalue()),
                sr=8000,
                mono=True,
                duration=20,
            )
            figure, axis = plt.subplots(figsize=(10, 2.5))
            axis.plot(
                np.arange(len(waveform)) / waveform_sr,
                waveform,
                linewidth=0.6,
            )
            axis.set_xlabel("Time (seconds)")
            axis.set_ylabel("Amplitude")
            axis.set_title("Waveform (first 20 seconds)")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
        except Exception:
            st.caption("Waveform could not be plotted.")

    st.subheader("Transcript")
    if result.get("transcript"):
        st.text_area(
            "Recognized speech (may contain errors)",
            result["transcript"],
            height=130,
        )
    else:
        st.caption("No transcript yet. Use Generate transcript (optional).")

    st.download_button(
        "Download PDF report",
        data=make_pdf(result),
        file_name="voiceguard_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


st.divider()
st.header("Session analysis history")

if st.session_state.history:
    for item in st.session_state.history:
        with st.expander(f'{item["created_at"]} — {item["filename"]}'):
            st.write("Model:", item.get("model_id", "N/A"))
            for score_item in item.get("scores", []):
                st.write(
                    f'{score_item["label"]}: {score_item["score"] * 100:.2f}%'
                )
            if item.get("metrics"):
                st.json(item["metrics"])
            if item.get("transcript"):
                st.write("Transcript:", item["transcript"])
else:
    st.caption("No analyses in this session yet.")

st.caption(
    "Research/demo only. No speaker identity verification or forensic authenticity claim is made."
)

st.divider()
st.header("🌐 TinyFish Web Search")
st.caption(
    "Search the web for information related to AI voice detection. "
    "Search results do not prove whether audio is genuine or AI-generated."
)

query = st.text_input(
    "What do you want to search for?",
    placeholder="AI voice cloning detection",
)

if st.button("Search with TinyFish"):
    if query.strip():
        with st.spinner("Searching the web..."):
            tinyfish_search(query.strip())
    else:
        st.warning("Please enter a search query first.")
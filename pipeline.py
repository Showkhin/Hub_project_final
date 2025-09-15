import os
import tempfile
import pandas as pd
import streamlit as st
from typing import List, Callable, Optional
import time

# --- Helpers ---
from ui_helpers import confirmation_modal, show_csv
from oci_helpers import download_blob, upload_cloud_csv, load_cloud_csv, delete_objects
from ml_models import get_audio_emotion_model, get_whisper_model, get_text_emotion_classifier
from utils import preprocess_audio, apply_vad, decide_final_emotion, clean_with_ollama
import extract_ola  # automatically updates final_emotion_ensemble

# --- NER Setup (spaCy) ---
import spacy

_SPACY_NLP = None
def get_spacy(nlp_name="en_core_web_sm"):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load(nlp_name)
        except Exception:
            from spacy.cli import download
            download(nlp_name)
            _SPACY_NLP = spacy.load(nlp_name)
    return _SPACY_NLP

# === Main Pipeline ===
def run_emotion_pipeline_for_files(
    object_names: List[str],
    progress_cb: Optional[Callable[[float], None]] = None,
    whisper_model_size: str = "large"
) -> None:

    def _p(x):
        if progress_cb:
            progress_cb(x)

    tmpdir = tempfile.mkdtemp(prefix="wavs_")
    local_paths = []
    for name in object_names:
        data = download_blob(name)
        local_path = os.path.join(tmpdir, os.path.basename(name))
        with open(local_path, "wb") as f:
            f.write(data)
        local_paths.append(local_path)

    # --- Load previous CSVs ---
    wav_cols = ["filename", "emotion", "confidence (%)"]
    wav_prev = load_cloud_csv("predicted_emotions.csv", wav_cols)

    trans_cols = ["filename", "transcription", "transcription_cleaned"]
    trans_prev = load_cloud_csv("transcription_output.csv", trans_cols)

    text_cols = ["filename", "emotion", "confidence_percent", "transcription_cleaned"]
    text_prev = load_cloud_csv("transcription_output_with_emotion.csv", text_cols)

    # --- Step 1: Audio emotion ---
    model = get_audio_emotion_model()
    new_rows = []
    for lp in local_paths:
        bn = os.path.basename(lp)
        if (not wav_prev.empty) and (bn in set(wav_prev["filename"].values)):
            continue
        out_prob, score, index, label = model.classify_file(lp)
        raw_label = label[0].lower() if isinstance(label, (list, tuple)) else str(label).lower()
        key = raw_label[:3]
        emotion = {
            "ang": "Anger", "dis": "Disgust", "fea": "Fear",
            "hap": "Happy", "neu": "Neutral", "sad": "Sad",
            "cal": "Calm", "sur": "Surprised",
        }.get(key, "Unknown")
        conf = float(score) * 100
        new_rows.append([bn, emotion, f"{conf:.2f}"])
    wav_new = pd.DataFrame(new_rows, columns=wav_cols)
    wav_all = pd.concat([wav_prev, wav_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("predicted_emotions.csv", wav_all)
    _p(0.25)

    # --- Step 2: Whisper transcription ---
    whisper_model = get_whisper_model(model_size=whisper_model_size)
    new_rows_t = []

    for lp in local_paths:
        bn = os.path.basename(lp)
        if not trans_prev.empty and (bn in set(trans_prev["filename"].values)):
            existing = trans_prev[trans_prev["filename"] == bn]
            if not existing.empty and str(existing.iloc[0]["transcription"]).strip():
                continue

        cleaned = preprocess_audio(lp)
        vad_path = apply_vad(cleaned)
        result = whisper_model.transcribe(vad_path, language="en", task="transcribe")
        transcription = (result.get("text") or "").strip()
        transcription_cleaned = clean_with_ollama(transcription)

        try:
            os.remove(cleaned)
            os.remove(vad_path)
        except Exception:
            pass

        new_rows_t.append([bn, transcription, transcription_cleaned])

    trans_new = pd.DataFrame(new_rows_t, columns=trans_cols)
    trans_all = pd.concat([trans_prev, trans_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("transcription_output.csv", trans_all)
    _p(0.50)

    # --- Step 3: Text emotion ---
    clf = get_text_emotion_classifier()
    text_label_map = {
        "anger": "Anger", "disgust": "Disgust", "fear": "Fear",
        "joy": "Happy", "neutral": "Neutral", "sadness": "Sad",
        "calm": "Calm", "surprise": "Surprised",
    }
    new_rows_text = []
    for _, row in trans_all.iterrows():
        bn = row["filename"]
        if (not text_prev.empty) and (bn in set(text_prev["filename"].values)):
            continue

        text = row["transcription_cleaned"] or row["transcription"] or ""
        if not text.strip():
            continue

        scores = clf(text)[0]
        best_label, best_score = None, 0.0
        for item in scores:
            lbl = item["label"].lower()
            score = float(item["score"])
            if lbl in text_label_map and score > best_score:
                best_score = score
                best_label = text_label_map[lbl]
        if best_label is None:
            best_label, best_score = "Calm", 1.0

        new_rows_text.append({
            "filename": bn,
            "emotion": best_label,
            "confidence_percent": f"{best_score * 100:.2f}%",
            "transcription_cleaned": text
        })

    text_newdf = pd.DataFrame(new_rows_text)
    text_all = pd.concat([text_prev, text_newdf]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("transcription_output_with_emotion.csv", text_all)
    _p(0.75)

    # --- Step 4: Ensemble for processed_incidents_with_emotion.csv ---
    ens_cols = ["filename", "emotion", "transcription"]
    ens_prev = load_cloud_csv("processed_incidents_with_emotion.csv", ens_cols)
    new_rows_ens = []
    for bn in set(wav_all["filename"].values).union(set(text_all["filename"].values)):
        if not ens_prev.empty and (bn in set(ens_prev["filename"].values)):
            continue
        a_row = wav_all[wav_all["filename"] == bn]
        t_row = text_all[text_all["filename"] == bn]
        a_em = a_row["emotion"].values[0] if not a_row.empty else "neutral"
        t_em = t_row["emotion"].values[0] if not t_row.empty else "neutral"
        transcription = t_row["transcription_cleaned"].values[0] if not t_row.empty else ""
        final_emotion = decide_final_emotion(a_em, t_em)
        new_rows_ens.append({
            "filename": bn,
            "emotion": final_emotion,
            "transcription": transcription
        })

    ens_new = pd.DataFrame(new_rows_ens)
    ens_all = pd.concat([ens_prev, ens_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("processed_incidents_with_emotion.csv", ens_all)
    _p(0.95)

    # --- Step 5: Automatically generate final_emotion_ensemble.csv ---
    try:
        extract_ola.process_and_upload()
        _p(1.0)
        st.session_state.page = "Results"
        st.rerun()

    except Exception as e:
        st.error(f"Failed to generate final ensemble: {e}")

    # --- Cleanup uploaded files ---
    delete_objects(object_names)

# --- Progress UI ---
def show_progress_ui() -> Callable[[float], None]:
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_text = st.empty()

    def progress_callback(progress: float, step_name: str = ""):
        p = min(max(progress, 0.0), 1.0)
        progress_bar.progress(p)
        status_text.markdown(f"**Progress:** {p*100:.0f}%")
        if step_name:
            step_text.markdown(f"**Step:** {step_name}")
        time.sleep(0.05)
    return progress_callback

# --- Streamlit entry point ---
if __name__ == "__main__":
    st.title("Emotion Analysis Pipeline")
    uploaded_files = st.file_uploader("Upload WAV files", accept_multiple_files=True, type=["wav"])
    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix="uploaded_wavs_")
        object_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            object_names.append(file_path)

        cb = show_progress_ui()
        run_emotion_pipeline_for_files(object_names=object_names, progress_cb=cb)
        st.success("Processing complete! ✅ Final Ensemble updated automatically.")

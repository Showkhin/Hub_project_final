# pipeline.py
import os
import tempfile
import pandas as pd
import streamlit as st
from typing import List, Callable, Optional
import time
import logging

from ui_helpers import confirmation_modal, show_csv
from oci_helpers import download_blob, upload_cloud_csv, load_cloud_csv, delete_objects
from ml_models import get_audio_emotion_model, get_whisper_model, get_text_emotion_classifier
from utils import (
    preprocess_audio,
    clean_with_ollama,
    tta_audio_paths,
    weighted_majority,
    llm_text_emotion,
    decide_final_emotion,
)
import extract_ola  # auto updates final_emotion_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_emotion_pipeline_for_files(
    object_names: List[str],
    progress_cb: Optional[Callable[[float], None]] = None,
    whisper_model_size: str = "large-v2"
) -> None:

    def _p(x, step=""):
        if progress_cb:
            progress_cb(x, step)

    tmpdir = tempfile.mkdtemp(prefix="wavs_")
    local_paths = []
    for name in object_names:
        data = download_blob(name)
        local_path = os.path.join(tmpdir, os.path.basename(name))
        with open(local_path, "wb") as f:
            f.write(data)
        local_paths.append(local_path)
        logging.info(f"Downloaded {name} to {local_path}")

    # Load previous CSVs
    wav_cols = ["filename", "emotion", "confidence (%)"]
    wav_prev = load_cloud_csv("predicted_emotions.csv", wav_cols)

    trans_cols = ["filename", "transcription", "transcription_cleaned"]
    trans_prev = load_cloud_csv("transcription_output.csv", trans_cols)

    text_cols = ["filename", "emotion", "confidence_percent", "transcription_cleaned"]
    text_prev = load_cloud_csv("transcription_output_with_emotion.csv", text_cols)

    # Step 1: Audio emotion (A + B: normalization & TTA)
    _p(0.05, "Preparing audio")
    model = get_audio_emotion_model()
    new_rows = []
    for lp in local_paths:
        bn = os.path.basename(lp)
        if (not wav_prev.empty) and (bn in set(wav_prev["filename"].values)):
            logging.info(f"Skipping audio emotion for previously processed file {bn}")
            continue

        # A: normalize/denoise/VAD trimming inside preprocess_audio
        prepared = preprocess_audio(lp, do_vad=True, vad_aggr=2)

        # B: small TTA set
        variants = tta_audio_paths(prepared)

        labels, scores = [], []
        for p in variants:
            try:
                out_prob, score, index, label = model.classify_file(p)
                raw_label = label[0].lower() if isinstance(label, (list, tuple)) else str(label).lower()
                # Map common prefixes
                key = raw_label[:3]
                emotion = {
                    "ang": "Anger", "dis": "Disgust", "fea": "Fear",
                    "hap": "Happy", "neu": "Neutral", "sad": "Sad",
                    "cal": "Calm", "sur": "Surprised",
                }.get(key, raw_label.capitalize())
                labels.append(emotion)
                scores.append(float(score))
            except Exception as e:
                logging.warning(f"TTA variant failed for {bn}: {e}")

        if not labels:
            # fallback
            labels, scores = ["Neutral"], [0.4]

        final_label, conf_weighted = weighted_majority(labels, scores)
        conf_pct = f"{conf_weighted * 100:.2f}"
        new_rows.append([bn, final_label, conf_pct])
        logging.info(f"[Audio TTA] {bn}: {final_label} ({conf_pct}%)")

        # cleanup temp variants except prepared
        for p in variants:
            if p != prepared:
                try:
                    os.remove(p)
                except Exception:
                    pass
        try:
            if prepared != lp:
                os.remove(prepared)
        except Exception:
            pass

    wav_new = pd.DataFrame(new_rows, columns=wav_cols)
    wav_all = pd.concat([wav_prev, wav_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("predicted_emotions.csv", wav_all)
    _p(0.25, "Audio emotions done")

    # Step 2: Whisper transcription (A: cleaned inputs used above already)
    _p(0.30, "Transcribing")
    whisper_model = get_whisper_model(model_size=whisper_model_size)
    new_rows_t = []
    for lp in local_paths:
        bn = os.path.basename(lp)
        if not trans_prev.empty and (bn in set(trans_prev["filename"].values)):
            existing = trans_prev[trans_prev["filename"] == bn]
            if not existing.empty and str(existing.iloc[0]["transcription"]).strip():
                logging.info(f"Skipping transcription for already processed file {bn}")
                continue

        # Reuse normalization for transcription input (no VAD to keep context)
        prepared_no_vad = preprocess_audio(lp, do_vad=False)
        result = whisper_model.transcribe(prepared_no_vad, language="en", task="transcribe")
        transcription = (result.get("text") or "").strip()
        transcription_cleaned = clean_with_ollama(transcription)

        logging.info(f"Transcription for {bn}: {transcription_cleaned[:120]}...")
        try:
            if prepared_no_vad != lp:
                os.remove(prepared_no_vad)
        except Exception:
            pass

        new_rows_t.append([bn, transcription, transcription_cleaned])

    trans_new = pd.DataFrame(new_rows_t, columns=trans_cols)
    trans_all = pd.concat([trans_prev, trans_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("transcription_output.csv", trans_all)
    _p(0.50, "Transcriptions saved")

    # Step 3: Text emotion classification (transformer + D: LLM pseudo-label)
    _p(0.55, "Text emotions")
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
            logging.info(f"Skipping text emotion for already processed file {bn}")
            continue

        text = row["transcription_cleaned"] or row["transcription"] or ""
        if not text.strip():
            continue

        # transformer predictions
        scores = clf(text)[0]
        best_label, best_score = None, 0.0
        for item in scores:
            lbl = item["label"].lower()
            score = float(item["score"])
            if lbl in text_label_map and score > best_score:
                best_score = score
                best_label = text_label_map[lbl]

        if best_label is None:
            best_label, best_score = "Neutral", 0.4

        # D: LLM pseudo-label for text
        llm_lbl, llm_conf = llm_text_emotion(text)

        # If both agree, modestly boost confidence
        if llm_lbl == best_label:
            best_score = min(1.0, (best_score + llm_conf) / 2 + 0.1)
        else:
            # choose the higher-confidence one
            if llm_conf > best_score:
                best_label, best_score = llm_lbl, llm_conf

        new_rows_text.append({
            "filename": bn,
            "emotion": best_label,
            "confidence_percent": f"{best_score * 100:.2f}%",
            "transcription_cleaned": text
        })
        logging.info(f"[Text+LLM] {bn}: {best_label} ({best_score:.2f})")

    text_newdf = pd.DataFrame(new_rows_text)
    text_all = pd.concat([text_prev, text_newdf]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("transcription_output_with_emotion.csv", text_all)
    _p(0.75, "Text emotions saved")

    # Step 4: Final ensemble with confidence + F (selective acceptance)
    _p(0.80, "Final ensemble")
    ens_cols = ["filename", "emotion", "transcription", "needs_review", "final_confidence"]
    ens_prev = load_cloud_csv("processed_incidents_with_emotion.csv", ens_cols)
    new_rows_ens = []

    # Build quick lookup dicts
    wav_map = wav_all.set_index("filename").to_dict(orient="index")
    text_map = text_all.set_index("filename").to_dict(orient="index")
    for bn in set(wav_all["filename"].values).union(set(text_all["filename"].values)):
        if not ens_prev.empty and (bn in set(ens_prev["filename"].values)):
            logging.info(f"Skipping ensemble for already processed file {bn}")
            continue

        a_em = "Neutral"
        a_conf = 0.4
        if bn in wav_map:
            a_em = wav_map[bn].get("emotion", "Neutral")
            try:
                a_conf = float(str(wav_map[bn].get("confidence (%)", "0")).replace("%", "")) / 100.0
            except Exception:
                a_conf = 0.4

        t_em = "Neutral"
        t_conf = 0.4
        transcription = ""
        if bn in text_map:
            t_em = text_map[bn].get("emotion", "Neutral")
            try:
                t_conf = float(str(text_map[bn].get("confidence_percent", "0%")).replace("%", "")) / 100.0
            except Exception:
                t_conf = 0.4
            transcription = text_map[bn].get("transcription_cleaned", "") or ""

        # D: Use LLM as a soft source too (already computed above for text step, but recompute if missing)
        llm_lbl, llm_conf = llm_text_emotion(transcription) if transcription else ("Neutral", 0.4)

        final_label, final_conf, needs_review = decide_final_emotion(
            a_em, t_em, a_conf, t_conf, llm_lbl, llm_conf
        )

        new_rows_ens.append({
            "filename": bn,
            "emotion": final_label,
            "transcription": transcription,
            "needs_review": "yes" if needs_review else "no",
            "final_confidence": f"{final_conf:.2f}"
        })
        logging.info(f"[Final] {bn}: {final_label} ({final_conf:.2f}) review={needs_review}")

    ens_new = pd.DataFrame(new_rows_ens)
    ens_all = pd.concat([ens_prev, ens_new]).drop_duplicates(subset="filename", keep="last")
    upload_cloud_csv("processed_incidents_with_emotion.csv", ens_all)
    _p(0.92, "Final ensemble saved")

    # Step 5: Generate final emotion ensemble with extract_ola (fields + fuzzy names, date normalization)
    try:
        extract_ola.process_and_upload()
        _p(1.0, "Final CSV enriched")
        st.session_state.page = "Results"
        st.rerun()
    except Exception as e:
        st.error(f"Failed to generate final ensemble: {e}")

    # Cleanup uploaded files (cloud objects) if desired
    delete_objects(object_names)

import os
import io
import math
import tempfile
import subprocess
import contextlib
import wave
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr
import webrtcvad
import librosa

import requests
from rapidfuzz import fuzz, process

# ================
# Optional spaCy (kept; not strictly used here)
# ================
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ============
# Constants
# ============
KNOWN_NAMES = ["Alice", "Bob", "Charlie", "John Smith"]  # replace with org dataset
OLLAMA_URL_GENERATE = os.getenv("OLLAMA_URL_GENERATE", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL_TEXT = os.getenv("OLLAMA_MODEL_TEXT", "gemma2")  # for grammar clean & emotion
AUDIO_TARGET_SR = 16000
AUDIO_TARGET_WIDTH = 2  # 16-bit PCM
VAD_FRAME_MS = 30


# =====================================
# Audio utilities (A: normalize/denoise/VAD)
# =====================================
def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def ensure_wav_pcm16_mono_16k(input_path: str) -> str:
    """
    Convert any audio to mono, 16-bit PCM WAV @ 16kHz via ffmpeg (if available);
    otherwise fallback to librosa+soundfile.
    """
    base, _ = os.path.splitext(input_path)
    out_path = base + "_16k_mono.wav"

    if _ffmpeg_available():
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(AUDIO_TARGET_SR),
            "-f", "wav",
            "-acodec", "pcm_s16le",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        y, sr = librosa.load(input_path, sr=AUDIO_TARGET_SR, mono=True)
        sf.write(out_path, y, AUDIO_TARGET_SR, subtype="PCM_16")

    return out_path


def _remove_dc_and_rms_normalize(y: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    """
    Remove DC offset and normalize RMS to a target level.
    """
    if y.size == 0:
        return y
    y = y - np.mean(y)  # DC
    rms = np.sqrt(np.mean(np.square(y))) + 1e-8
    gain = target_rms / rms
    y = y * gain
    # prevent clipping
    y = np.clip(y, -0.999, 0.999)
    return y


def denoise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except Exception:
        return y


def apply_vad_pcm_bytes(pcm16_bytes: bytes, sr: int = 16000, aggressiveness: int = 2) -> bytes:
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * VAD_FRAME_MS / 1000) * 2  # 16-bit
    frames = [pcm16_bytes[i:i + frame_size] for i in range(0, len(pcm16_bytes), frame_size)]
    voiced = bytearray()
    for f in frames:
        if len(f) < frame_size:
            continue
        if vad.is_speech(f, sr):
            voiced += f
    return bytes(voiced)


def trim_silence_with_vad(wav_path: str, aggressiveness: int = 2) -> str:
    """
    Read wav (assumed mono/16k/16-bit), VAD-trim, return new file.
    If no voiced frames, return original.
    """
    with wave.open(wav_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if num_channels != 1 or sampwidth != 2 or framerate != 16000:
        # safety: convert first
        fixed = ensure_wav_pcm16_mono_16k(wav_path)
        return trim_silence_with_vad(fixed, aggressiveness=aggressiveness)

    voiced = apply_vad_pcm_bytes(pcm, sr=16000, aggressiveness=aggressiveness)
    if not voiced:
        return wav_path

    out_path = wav_path.replace(".wav", "_vad.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(voiced)
    return out_path


def preprocess_audio(input_path: str,
                     do_vad: bool = True,
                     vad_aggr: int = 2,
                     target_rms: float = 0.03) -> str:
    """
    A: Normalize pipeline — convert → remove DC → RMS normalize → mild denoise → optional VAD.
    Returns path to cleaned WAV.
    """
    # 1) Convert format
    fixed = ensure_wav_pcm16_mono_16k(input_path)

    # 2) Load for DC/RMS + denoise
    y, sr = sf.read(fixed)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = _remove_dc_and_rms_normalize(y, target_rms=target_rms)
    y = denoise(y, sr)

    cleaned = fixed.replace(".wav", "_cleaned.wav")
    sf.write(cleaned, y, sr, subtype="PCM_16")

    # 3) Optional VAD trim
    if do_vad:
        return trim_silence_with_vad(cleaned, aggressiveness=vad_aggr)
    return cleaned


# =====================================
# B: Test-time augmentation (TTA)
# =====================================
def _variant_speed(y: np.ndarray, rate: float) -> np.ndarray:
    # librosa.effects.time_stretch expects float32
    y32 = y.astype(np.float32, copy=False)
    try:
        return librosa.effects.time_stretch(y32, rate)
    except Exception:
        return y


def _variant_pitch(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    y32 = y.astype(np.float32, copy=False)
    try:
        return librosa.effects.pitch_shift(y32, sr=sr, n_steps=n_steps)
    except Exception:
        return y


def _variant_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    factor = 10 ** (gain_db / 20.0)
    z = y * factor
    return np.clip(z, -0.999, 0.999)


def save_temp_wav(y: np.ndarray, sr: int) -> str:
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def tta_audio_paths(original_wav_16k: str) -> List[str]:
    """
    Build small set of variants ±2% speed, ±1 semitone, ±2 dB.
    Returns list including the original path first.
    """
    paths = [original_wav_16k]

    y, sr = sf.read(original_wav_16k)
    if y.ndim > 1:
        y = y.mean(axis=1)

    variants = []

    # speed
    variants.append(_variant_speed(y, 1.02))
    variants.append(_variant_speed(y, 0.98))
    # pitch
    variants.append(_variant_pitch(y, sr, 1.0))
    variants.append(_variant_pitch(y, sr, -1.0))
    # gain
    variants.append(_variant_gain(y, +2.0))
    variants.append(_variant_gain(y, -2.0))

    for v in variants:
        paths.append(save_temp_wav(v, sr))
    return paths


def weighted_majority(labels: List[str], scores: List[float]) -> Tuple[str, float]:
    """
    Combine labels with confidence scores (0..1). Returns (label, combined_confidence).
    """
    bucket: Dict[str, float] = {}
    for lbl, sc in zip(labels, scores):
        bucket[lbl] = bucket.get(lbl, 0.0) + float(sc)
    best = max(bucket.items(), key=lambda x: x[1])
    total = sum(bucket.values()) + 1e-8
    return best[0], best[1] / total


# =====================================
# D: LLM helpers (cleanup + emotion from text)
# =====================================
def clean_with_ollama(text: str) -> str:
    """
    Grammar & cleanup using Ollama /api/generate.
    """
    if not text or not text.strip():
        return text or ""
    payload = {
        "model": OLLAMA_MODEL_TEXT,
        "prompt": f"Clean this transcription for grammar and remove obvious filler words, without changing meaning:\n\n{text}",
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL_GENERATE, json=payload, timeout=120)
        r.raise_for_status()
        return (r.json().get("response") or "").strip() or text
    except Exception:
        return text


EMO_LABELS_CANON = {
    "anger": "Anger",
    "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "happy": "Happy",
    "sad": "Sad",
    "sadness": "Sad",
    "calm": "Calm",
    "neutral": "Neutral",
    "surprise": "Surprised",
    "surprised": "Surprised",
}


def llm_text_emotion(text: str) -> Tuple[str, float]:
    """
    Ask Ollama to infer emotion and a confidence 0..1 (rough).
    Returns (label, confidence). Fails gracefully to ("Neutral", 0.4)
    """
    if not text or not text.strip():
        return "Neutral", 0.4

    prompt = (
        "You are an emotion rater. From the text, infer the single best emotion label among: "
        "Anger, Disgust, Fear, Happy, Neutral, Sad, Calm, Surprised.\n"
        "Return a STRICT JSON object with keys 'label' and 'confidence' (0..1).\n"
        "Example: {\"label\": \"Happy\", \"confidence\": 0.76}\n\n"
        f"Text:\n{text}"
    )
    payload = {"model": OLLAMA_MODEL_TEXT, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL_GENERATE, json=payload, timeout=120)
        r.raise_for_status()
        raw = (r.json().get("response") or "").strip()
        # try parse JSON
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            # try to locate json substring
            import re, json
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return "Neutral", 0.4
        lbl = str(data.get("label", "")).strip().lower()
        conf = float(data.get("confidence", 0.4))
        canon = EMO_LABELS_CANON.get(lbl, "Neutral")
        conf = max(0.0, min(1.0, conf))
        return canon, conf
    except Exception:
        return "Neutral", 0.4


# =====================================
# Fuzzy name matching (unchanged)
# =====================================
def fuzzy_match_name(candidate: str, threshold: float = 0.75):
    if not candidate:
        return None, 0.0
    best = process.extractOne(candidate, KNOWN_NAMES, scorer=fuzz.ratio)
    if best and best[1] / 100 >= threshold:
        return best[0], best[1] / 100
    return None, 0.0


# =====================================
# G: (light) Final decision helper (uses confidence)
# =====================================
def decide_final_emotion(audio_label: str,
                         text_label: str,
                         audio_conf: float,
                         text_conf: float,
                         llm_label: Optional[str] = None,
                         llm_conf: Optional[float] = None) -> Tuple[str, float, bool]:
    """
    Confidence-based selection (F) and agreement boost (D+F).
    Returns (final_label, final_confidence_0_1, needs_review_boolean).
    Heuristic:
      - If 2 sources agree (audio+text or any pair), boost confidence.
      - Otherwise pick label from the highest-confidence source.
      - Flag for review if final confidence < 0.55
    """
    # canonicalize
    def canon(x: str) -> str:
        if not x:
            return "Neutral"
        return EMO_LABELS_CANON.get(x.lower(), x.capitalize())

    a = canon(audio_label)
    t = canon(text_label)
    l = canon(llm_label) if llm_label else None

    # pairwise agreements
    agrees = []
    if l:
        if a == t:
            agrees = [a]
        elif a == l:
            agrees = [a]
        elif t == l:
            agrees = [t]

    if not agrees:
        if a == t:
            agrees = [a]

    if agrees:
        label = agrees[0]
        # combine confidences (average of agreeing sources)
        vals = []
        if label == a:
            vals.append(audio_conf)
        if label == t:
            vals.append(text_conf)
        if l and label == l:
            vals.append(llm_conf if llm_conf is not None else 0.5)
        base = np.mean(vals) if vals else max(audio_conf, text_conf, llm_conf or 0.0)
        final_conf = min(1.0, base + 0.15)  # agreement boost
    else:
        # pick by highest confidence among available
        triples = [(a, audio_conf), (t, text_conf)]
        if l is not None:
            triples.append((l, llm_conf if llm_conf is not None else 0.5))
        label, final_conf = max(triples, key=lambda x: x[1])

    needs_review = final_conf < 0.55
    return label, final_conf, needs_review


# =====================================
# Date/time normalization & fuzzy fallback
# =====================================
def normalize_date(date_str: str) -> Tuple[str, bool]:
    """
    Try parsing any format to YYYY-MM-DD.
    Returns (normalized_str, parsed_ok).
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return "", False
    try:
        dt = pd.to_datetime(date_str, errors="raise", dayfirst=True)
        return dt.strftime("%Y-%m-%d"), True
    except Exception:
        try:
            dt = pd.to_datetime(date_str, errors="raise", dayfirst=False)
            return dt.strftime("%Y-%m-%d"), True
        except Exception:
            return date_str, False


def normalize_time(time_str: str) -> Tuple[str, bool]:
    """
    Normalize to HH:MM (24h) if possible; otherwise return original.
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return "", False
    try:
        t = pd.to_datetime(time_str, errors="raise").time()
        return f"{t.hour:02d}:{t.minute:02d}", True
    except Exception:
        return time_str, False


def dates_equal(a: str, b: str) -> bool:
    """
    Try parse both; if both parse, compare ISO dates; else use fuzzy match.
    """
    na, oka = normalize_date(a)
    nb, okb = normalize_date(b)
    if oka and okb:
        return na == nb
    # fuzzy fallback
    return fuzz.ratio(str(a), str(b)) >= 90

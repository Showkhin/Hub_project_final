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
import json
from rapidfuzz import fuzz, process

# Optional spaCy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ===== Constants =====
OLLAMA_URL_GENERATE = os.getenv("OLLAMA_URL_GENERATE", "http://127.0.0.1:11434/api/generate")
OLLAMA_URL_CHAT = os.getenv("OLLAMA_URL_CHAT", "http://127.0.0.1:11434/v1/chat/completions")
OLLAMA_MODEL_TEXT = os.getenv("OLLAMA_MODEL_TEXT", "gemma3")  # for cleanup & emotion

AUDIO_TARGET_SR = 16000
VAD_FRAME_MS = 30

# ===== Audio I/O helpers (A) =====
def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ensure_wav_pcm16_mono_16k(input_path: str) -> str:
    """Convert any audio to mono, 16-bit PCM WAV @ 16kHz via ffmpeg (fallback to librosa)."""
    base, _ = os.path.splitext(input_path)
    out_path = base + "_16k_mono.wav"
    if _ffmpeg_available():
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(AUDIO_TARGET_SR),
            "-acodec", "pcm_s16le",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        y, sr = librosa.load(input_path, sr=AUDIO_TARGET_SR, mono=True)
        sf.write(out_path, y, AUDIO_TARGET_SR, subtype="PCM_16")
    return out_path

def _remove_dc_and_rms_normalize(y: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    if y.size == 0:
        return y
    y = y - np.mean(y)
    rms = np.sqrt(np.mean(np.square(y))) + 1e-8
    gain = target_rms / rms
    y = y * gain
    return np.clip(y, -0.999, 0.999)

def denoise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except Exception:
        return y

def apply_vad_pcm_bytes(pcm16_bytes: bytes, sr: int = 16000, aggressiveness: int = 2) -> bytes:
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * VAD_FRAME_MS / 1000) * 2
    frames = [pcm16_bytes[i:i + frame_size] for i in range(0, len(pcm16_bytes), frame_size)]
    voiced = bytearray()
    for f in frames:
        if len(f) < frame_size:
            continue
        if vad.is_speech(f, sr):
            voiced += f
    return bytes(voiced)

def trim_silence_with_vad(wav_path: str, aggressiveness: int = 2) -> str:
    with wave.open(wav_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    if num_channels != 1 or sampwidth != 2 or framerate != 16000:
        fixed = ensure_wav_pcm16_mono_16k(wav_path)
        return trim_silence_with_vad(fixed, aggressiveness)
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

def preprocess_audio(input_path: str, do_vad: bool = True, vad_aggr: int = 2, target_rms: float = 0.03) -> str:
    """Convert → DC/RMS normalize → denoise → optional VAD. Returns cleaned WAV path."""
    fixed = ensure_wav_pcm16_mono_16k(input_path)
    y, sr = sf.read(fixed)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = _remove_dc_and_rms_normalize(y, target_rms)
    y = denoise(y, sr)
    cleaned = fixed.replace(".wav", "_cleaned.wav")
    sf.write(cleaned, y, sr, subtype="PCM_16")
    if do_vad:
        return trim_silence_with_vad(cleaned, vad_aggr)
    return cleaned

# ===== Test-time augmentation (B) =====
def _variant_speed(y: np.ndarray, rate: float) -> np.ndarray:
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
    paths = [original_wav_16k]
    y, sr = sf.read(original_wav_16k)
    if y.ndim > 1:
        y = y.mean(axis=1)
    variants = [
        _variant_speed(y, 1.02),
        _variant_speed(y, 0.98),
        _variant_pitch(y, sr, +1.0),
        _variant_pitch(y, sr, -1.0),
        _variant_gain(y, +2.0),
        _variant_gain(y, -2.0),
    ]
    for v in variants:
        paths.append(save_temp_wav(v, sr))
    return paths

def weighted_majority(labels: List[str], scores: List[float]) -> Tuple[str, float]:
    bucket: Dict[str, float] = {}
    for lbl, sc in zip(labels, scores):
        bucket[lbl] = bucket.get(lbl, 0.0) + float(sc)
    best = max(bucket.items(), key=lambda x: x[1])
    total = sum(bucket.values()) + 1e-8
    return best[0], best[1] / total

# ===== LLM helpers (D) =====
def _try_ollama_generate(prompt: str, model: str) -> Optional[str]:
    try:
        r = requests.post(OLLAMA_URL_GENERATE, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
        if r.status_code == 200:
            return (r.json().get("response") or "").strip()
    except Exception:
        pass
    return None

def _try_ollama_chat(prompt: str, model: str) -> Optional[str]:
    try:
        r = requests.post(
            OLLAMA_URL_CHAT,
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            j = r.json()
            return j["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None

def _ollama_prompt(prompt: str, model: str) -> str:
    # Try /api/generate then /v1/chat/completions
    txt = _try_ollama_generate(prompt, model)
    if txt is not None and txt != "":
        return txt
    txt = _try_ollama_chat(prompt, model)
    return txt or ""

def clean_with_ollama(text: str) -> str:
    if not text or not text.strip():
        return text or ""
    prompt = (
        "Clean this transcription for grammar and remove filler words without changing meaning:\n\n"
        f"{text}"
    )
    try:
        out = _ollama_prompt(prompt, OLLAMA_MODEL_TEXT)
        return out or text
    except Exception:
        return text

EMO_LABELS_CANON = {
    "anger": "Anger", "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy", "happy": "Happy",
    "sad": "Sad", "sadness": "Sad",
    "calm": "Calm",
    "neutral": "Neutral",
    "surprise": "Surprised", "surprised": "Surprised",
}

def llm_text_emotion(text: str) -> Tuple[str, float]:
    if not text or not text.strip():
        return "Neutral", 0.4
    prompt = (
        "You are an emotion rater. From the text, infer ONE label among: "
        "Anger, Disgust, Fear, Happy, Neutral, Sad, Calm, Surprised.\n"
        "Return STRICT JSON {\"label\": <string>, \"confidence\": <0..1>}.\n\n"
        f"Text:\n{text}"
    )
    try:
        raw = _ollama_prompt(prompt, OLLAMA_MODEL_TEXT)
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return "Neutral", 0.4
        lbl = str(data.get("label", "")).strip().lower()
        conf = float(data.get("confidence", 0.4))
        conf = max(0.0, min(1.0, conf))
        return EMO_LABELS_CANON.get(lbl, "Neutral"), conf
    except Exception:
        return "Neutral", 0.4

# ===== Confidence-based final decision (F) =====
def decide_final_emotion(audio_label: str, text_label: str, audio_conf: float, text_conf: float,
                         llm_label: Optional[str] = None, llm_conf: Optional[float] = None) -> Tuple[str, float, bool]:
    def canon(x: str) -> str:
        if not x:
            return "Neutral"
        return EMO_LABELS_CANON.get(x.lower(), x.capitalize())
    a = canon(audio_label); t = canon(text_label); l = canon(llm_label) if llm_label else None
    # agreement boost
    agreed = None
    if l and (a == t or a == l or t == l):
        agreed = a if a == t or a == l else t
    elif a == t:
        agreed = a
    if agreed:
        vals = []
        if agreed == a: vals.append(audio_conf)
        if agreed == t: vals.append(text_conf)
        if l and agreed == l and llm_conf is not None: vals.append(llm_conf)
        base = np.mean(vals) if vals else max(audio_conf, text_conf, llm_conf or 0.0)
        final_conf = min(1.0, base + 0.15)
        needs_review = final_conf < 0.55
        return agreed, final_conf, needs_review
    # pick highest confidence
    triples = [(a, audio_conf), (t, text_conf)]
    if l is not None:
        triples.append((l, llm_conf if llm_conf is not None else 0.5))
    label, final_conf = max(triples, key=lambda x: x[1])
    return label, final_conf, final_conf < 0.55

# ===== Date/time normalization (strict formats) =====
def normalize_date(date_str: str) -> Tuple[str, bool]:
    """Return (YYYY-MM-DD, ok)."""
    if not isinstance(date_str, str) or not date_str.strip():
        return "", False
    for dayfirst in (True, False):
        try:
            dt = pd.to_datetime(date_str, errors="raise", dayfirst=dayfirst)
            return dt.strftime("%Y-%m-%d"), True
        except Exception:
            pass
    return date_str, False

def normalize_time(time_str: str) -> Tuple[str, bool]:
    """Return (HH:MM, ok). Always drop seconds."""
    if not isinstance(time_str, str) or not time_str.strip():
        return "", False
    try:
        t = pd.to_datetime(time_str, errors="raise").time()
        return f"{t.hour:02d}:{t.minute:02d}", True
    except Exception:
        return time_str, False

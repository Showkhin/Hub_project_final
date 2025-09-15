import numpy as np
import soundfile as sf
import noisereduce as nr
import os
import subprocess
import tempfile
import webrtcvad
import wave
import contextlib
import spacy
import collections
from rapidfuzz import fuzz, process

nlp = spacy.load("en_core_web_sm")

# =========================
# --- VAD with webrtcvad ---
# =========================
def apply_vad(input_wav: str, aggressiveness: int = 2) -> str:
    """
    Trim silence/non-speech using WebRTC VAD.
    If no voiced audio detected, return original input_wav.
    """
    import librosa, soundfile as sf

    # Force 16k mono PCM
    y, sr = librosa.load(input_wav, sr=16000, mono=True)
    tmp_pcm = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_pcm, y, 16000)

    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(tmp_pcm, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
    os.remove(tmp_pcm)

    frame_duration = 30  # ms
    frame_size = int(16000 * frame_duration / 1000) * 2
    frames = [pcm_data[i:i + frame_size] for i in range(0, len(pcm_data), frame_size)]

    voiced = b""
    for f in frames:
        if len(f) < frame_size:
            continue
        if vad.is_speech(f, 16000):
            voiced += f

    # 🔑 Fallback: if nothing voiced, return original file
    if not voiced:
        return input_wav

    out_path = input_wav.replace(".wav", "_vad.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(voiced)
    return out_path



# =========================
# --- Ollama cleanup ---
# =========================
def clean_with_ollama(text: str) -> str:
    """
    Call Ollama locally to grammar-fix transcription.
    """
    import requests
    payload = {"model": "gemma2", "prompt": f"Clean this transcription:\n{text}"}
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload)
        cleaned = resp.json().get("response", "").strip()
        return cleaned or text
    except Exception:
        return text


# =========================
# --- Fuzzy matching ---
# =========================
KNOWN_NAMES = ["Alice", "Bob", "Charlie", "John Smith"]  # replace with org dataset

def fuzzy_match_name(candidate: str, threshold: float = 0.75):
    if not candidate:
        return None, 0.0
    best = process.extractOne(candidate, KNOWN_NAMES, scorer=fuzz.ratio)
    if best and best[1] / 100 >= threshold:
        return best[0], best[1] / 100
    return None, 0.0


# =========================
# --- Noise reduction ---
# =========================
def preprocess_audio(file_path: str) -> str:
    """
    Apply noise reduction and return path to cleaned wav.
    """
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # convert to mono
    reduced = nr.reduce_noise(y=audio, sr=sr)
    cleaned_path = file_path.replace(".wav", "_cleaned.wav")
    sf.write(cleaned_path, reduced, sr)
    return cleaned_path


# =========================
# --- Ensemble emotion ---
# =========================
def decide_final_emotion(audio_em: str, text_em: str) -> str:
    audio_em = (audio_em or "").lower()
    text_em = (text_em or "").lower()
    mapping = {
        ("anger", "happy"): "Anger",
        ("anger", "neutral"): "Anger",
        ("neutral", "fear"): "Fear",
        ("sad", "neutral"): "Sad",
        ("neutral", "happy"): "Happy",
        ("calm", "neutral"): "Calm",
        ("disgust", "neutral"): "Disgust",
        ("fear", "neutral"): "Fear",
        ("happy", "sad"): "Happy",
        ("surprise", "neutral"): "Surprised",
    }
    if (audio_em, text_em) in mapping:
        return mapping[(audio_em, text_em)]
    if audio_em in {
        "anger", "sad", "calm", "disgust", "fear", "happy", "surprise", "neutral"
    }:
        return audio_em.capitalize()
    return text_em.capitalize() if text_em else "Neutral"

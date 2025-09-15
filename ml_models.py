import torch
import os

import whisper
from transformers import pipeline
from speechbrain.inference.interfaces import foreign_class
import streamlit as st

@st.cache_resource(show_spinner=True)
def get_audio_emotion_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": device},
    )
    return model

@st.cache_resource(show_spinner=True)
def get_whisper_model(model_size="medium"):
    try:
        return whisper.load_model(model_size)
    except Exception:
        return whisper.load_model("small")  # fallback

@st.cache_resource(show_spinner=True)
def get_text_emotion_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input


st.set_page_config(page_title="Image Captioning", layout="centered")

OWNER = "MuhammadKamal6"
REPO  = "NLP."      
TAG   = "v1"

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

TOKENIZER_PATH = ART_DIR / "tokenizer.json"
CONFIG_PATH    = ART_DIR / "config.json"

# These 3 MUST be attached in GitHub Release assets (v1)
MODEL_URLS = {
    "LSTM": f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/last_lstm_model.keras",
    "RNN":  f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/last_rnn_model.keras",
    "GRU":  f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/last_gru_model.keras",
}

MODEL_LOCAL = {
    "LSTM": ART_DIR / "last_lstm_model.keras",
    "RNN":  ART_DIR / "last_rnn_model.keras",
    "GRU":  ART_DIR / "last_gru_model.keras",
}

@st.cache_resource
def load_encoder():
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")

def download_if_missing(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with st.spinner(f"Downloading {out_path.name} ..."):
            urllib.request.urlretrieve(url, out_path)

@st.cache_resource
def load_tokenizer_and_config():
    # tokenizer.json + config.json should be inside repo under artifacts/
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_length = int(cfg["max_length"])
    index_word = {v: k for k, v in tok.word_index.items()}
    return tok, index_word, max_length

@st.cache_resource
def load_caption_model(model_key: str):
    download_if_missing(MODEL_URLS[model_key], MODEL_LOCAL[model_key])
    return load_model(MODEL_LOCAL[model_key], compile=False)

def extract_feature(encoder, pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = encoder.predict(x, verbose=0)  # (1,1920)
    return feat

def generate_caption_beam(model, tokenizer, index_word, photo, max_length, beam_size=5) -> str:
    start = tokenizer.word_index.get("startseq")
    end = tokenizer.word_index.get("endseq")

    sequences = [([start], 0.0)]  # (token_ids, log_prob)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if end is not None and seq[-1] == end:
                all_candidates.append((seq, score))
                continue

            seq_padded = pad_sequences([seq], maxlen=max_length, padding="post")
            yhat = model.predict([photo, seq_padded], verbose=0)[0]

            top_k = np.argsort(yhat)[-beam_size:]
            for tok in top_k:
                prob = float(yhat[tok])
                all_candidates.append((seq + [int(tok)], score + np.log(prob + 1e-12)))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best_seq = sequences[0][0]
    words = []
    for tok in best_seq:
        w = index_word.get(tok)
        if w is None or w == "startseq":
            continue
        if w == "endseq":
            break
        words.append(w)
    return " ".join(words)


st.title("🖼️ Image Captioning (DenseNet201 + LSTM/RNN/GRU)")

model_choice = st.selectbox("Choose model", ["LSTM", "RNN", "GRU"])
beam_size = st.slider("Beam size", 1, 10, 5)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_container_width=True)

    # Ensure tokenizer/config exist in repo (artifacts/)
    tok, index_word, max_length = load_tokenizer_and_config()
    encoder = load_encoder()
    caption_model = load_caption_model(model_choice)

    with st.spinner("Generating caption..."):
        photo = extract_feature(encoder, img)
        caption = generate_caption_beam(caption_model, tok, index_word, photo, max_length, beam_size=beam_size)

    st.subheader("Caption")
    st.write(caption)

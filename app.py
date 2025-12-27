import json
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input

st.set_page_config(page_title="Image Captioning", layout="centered")

ART_DIR = "artifacts"

MODEL_FILES = {
    "LSTM (last)": f"{ART_DIR}/last_lstm_model.keras",
    "RNN (last)":  f"{ART_DIR}/last_rnn_model.keras",
    "GRU (last)":  f"{ART_DIR}/last_gru_model.keras",
}

@st.cache_resource
def load_tokenizer_and_config():
    with open(f"{ART_DIR}/tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    with open(f"{ART_DIR}/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_length = int(cfg["max_length"])
    index_word = {v: k for k, v in tok.word_index.items()}
    return tok, index_word, max_length

@st.cache_resource
def load_encoder():
    # same as your notebook: DenseNet201(include_top=False, pooling="avg") -> (1920,)
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource
def load_caption_model(model_path: str):
    return load_model(model_path, compile=False)

def extract_feature(encoder, pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = encoder.predict(x, verbose=0)  # (1, 1920)
    return feat

def generate_caption_beam(model, tokenizer, index_word, photo, max_length, beam_size=5):
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

st.caption(f"TensorFlow version: {tf.__version__}")

model_choice = st.selectbox("Choose model", list(MODEL_FILES.keys()))
beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_container_width=True)

    tok, index_word, max_length = load_tokenizer_and_config()
    encoder = load_encoder()
    caption_model = load_caption_model(MODEL_FILES[model_choice])

    with st.spinner("Generating caption..."):
        photo = extract_feature(encoder, img)          # (1,1920)
        caption = generate_caption_beam(
            caption_model, tok, index_word, photo, max_length, beam_size=beam_size
        )

    st.subheader("Caption")
    st.write(caption)

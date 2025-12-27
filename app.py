import json
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from deep_translator import GoogleTranslator  # <-- Arabic translation

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input



OWNER = "MuhammadKamal6"
REPO  = "NLP."   # change to "NLP" if your repo has no dot
TAG   = "v1"

ART_DIR = Path("artifacts")
TOKENIZER_PATH = ART_DIR / "tokenizer.json"
CONFIG_PATH    = ART_DIR / "config.json"

MODEL_FILES = {
    "LSTM": "last_lstm_model.keras",
    "RNN":  "last_rnn_model.keras",
    "GRU":  "last_gru_model.keras",
}


# ============================================================
# PAGE + ONE-SCREEN STYLE
# ============================================================
st.set_page_config(page_title="CaptionLab", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
      header, footer, #MainMenu { visibility: hidden; }
      .block-container { max-width: 1250px; padding-top: .6rem; padding-bottom: .6rem; }
      div[data-testid="stVerticalBlock"] > div { gap: .55rem; }

      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(850px 550px at 12% 10%, rgba(14,165,233,.14), transparent 60%),
          radial-gradient(850px 550px at 88% 12%, rgba(168,85,247,.10), transparent 55%),
          linear-gradient(180deg, #F6F7FB 0%, #F3F5FA 100%);
      }

      .hero {
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.85);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 14px 32px rgba(15,23,42,.06);
      }

      .card {
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.80);
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 12px 28px rgba(15,23,42,.05);
      }

      .caption { font-size: 1.02rem; line-height: 1.45; color: #0F172A; }
      .muted { opacity: .75; color:#0F172A; font-size: .9rem; }
      .pill {
        display:inline-flex; align-items:center; gap:7px;
        padding: 5px 9px; border-radius: 999px;
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.85);
        font-size: .82rem;
        margin-right: 6px;
      }

      /* Keep image small so no scrolling */
      [data-testid="stImage"] img {
        max-height: 340px;
        width: 100%;
        object-fit: contain;
        border-radius: 14px;
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.85);
      }

      /* Arabic translation style */
      .ar {
        direction: rtl;
        text-align: right;
        font-size: 0.98rem;
        line-height: 1.5;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid rgba(15,23,42,.08);
        color: rgba(15,23,42,.85);
      }

      /* Buttons */
      div.stButton > button {
        border-radius: 12px !important;
        padding: .60rem .9rem !important;
        border: 1px solid rgba(14,165,233,.35) !important;
        background: linear-gradient(135deg, rgba(14,165,233,.95), rgba(168,85,247,.75)) !important;
        color: white !important;
        font-weight: 800 !important;
      }
      div.stButton > button:hover { filter: brightness(1.05); }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DOWNLOAD + LOAD HELPERS
# ============================================================
def release_url(filename: str) -> str:
    return f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{filename}"

MODEL_URL = {k: release_url(v) for k, v in MODEL_FILES.items()}
MODEL_LOCAL = {k: ART_DIR / v for k, v in MODEL_FILES.items()}

def safe_exists(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False

def download_if_missing(model_key: str):
    out_path = MODEL_LOCAL[model_key]
    url = MODEL_URL[model_key]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if safe_exists(out_path):
        return

    with st.spinner(f"Preparing {model_key} model… (first time only)"):
        try:
            urllib.request.urlretrieve(url, out_path)
        except HTTPError as e:
            raise RuntimeError(f"Model download failed. Check GitHub Release assets.\n{url}\n{e}")
        except URLError as e:
            raise RuntimeError(f"Network error while downloading model.\n{url}\n{e}")

@st.cache_resource
def load_encoder():
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource
def load_tokenizer_config():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Missing artifacts/tokenizer.json in repo.")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing artifacts/config.json in repo.")

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_length = int(cfg["max_length"])
    index_word = {v: k for k, v in tok.word_index.items()}
    return tok, index_word, max_length

@st.cache_resource
def load_caption_model(model_key: str):
    download_if_missing(model_key)
    return load_model(MODEL_LOCAL[model_key], compile=False)


# ============================================================
# TRANSLATION (English -> Arabic)
# ============================================================
@st.cache_data(show_spinner=False)
def translate_to_ar(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return GoogleTranslator(source="en", target="ar").translate(text)
    except Exception:
        return ""


# ============================================================
# INFERENCE
# ============================================================
def extract_feature(encoder, pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return encoder.predict(x, verbose=0)  # (1,1920)

def greedy_decode(model, tokenizer, index_word, photo, max_length) -> str:
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        yhat = model.predict([photo, seq], verbose=0)[0]
        yhat = int(np.argmax(yhat))
        word = index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    words = [w for w in in_text.split() if w not in ("startseq", "endseq")]
    return " ".join(words).strip()

def beam_decode(model, tokenizer, index_word, photo, max_length, beam_size=5) -> str:
    start = tokenizer.word_index.get("startseq")
    end = tokenizer.word_index.get("endseq")
    sequences = [([start], 0.0)]

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
    return " ".join(words).strip()

def post_process(text: str, tone_mode: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if tone_mode == "Short":
        parts = text.split()
        text = " ".join(parts[:8]) if len(parts) > 8 else text
    if tone_mode == "Friendly":
        text = text[:1].upper() + text[1:]
    return text


# ============================================================
# UI (ALL ON ONE SCREEN)
# ============================================================
tok, index_word, max_length = load_tokenizer_config()
encoder = load_encoder()

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
        <div>
          <div style="font-size:1.35rem; font-weight:900; color:#0F172A;">✨ CaptionLab</div>
          <div class="muted">Upload a photo and get 3 captions + Arabic translation instantly.</div>
        </div>
        <div style="text-align:right;">
          <span class="pill">🧠 LSTM • RNN • GRU</span>
          <span class="pill">🧾 Max length: <b>{max_length}</b></span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

topL, topR = st.columns([1.15, 0.85], gap="large")

with topL:
    st.markdown("#### 📤 Upload")
    uploaded = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
    else:
        img = None

with topR:
    st.markdown("#### ⚙️ Controls")
    mode = st.selectbox("Generation", ["Beam Search (Best Quality)", "Greedy (Fast)"], index=0)
    beam_size = st.slider("Beam size", 1, 10, 5, disabled=mode.startswith("Greedy"))
    tone = st.selectbox("Caption style", ["Friendly", "Simple", "Short"], index=0)
    show_ar = st.toggle("Show Arabic translation", value=True)
    run = st.button("✨ Generate Captions", use_container_width=True, disabled=(img is None))

c1, c2, c3 = st.columns(3, gap="large")

def caption_card(col, title, caption, ms):
    col.markdown(f"#### {title}")
    col.markdown('<div class="card">', unsafe_allow_html=True)
    if caption:
        col.markdown(f'<div class="caption">“{caption}”</div>', unsafe_allow_html=True)

        if show_ar:
            ar = translate_to_ar(caption)
            if ar:
                col.markdown(f'<div class="ar">{ar}</div>', unsafe_allow_html=True)
            else:
                col.markdown('<div class="muted">Arabic translation unavailable.</div>', unsafe_allow_html=True)

        col.markdown(f'<div class="muted">⏱️ {ms:.0f} ms</div>', unsafe_allow_html=True)
    else:
        col.markdown('<div class="muted">Upload an image, then click “Generate Captions”.</div>', unsafe_allow_html=True)
    col.markdown("</div>", unsafe_allow_html=True)

results = {}

if run and img is not None:
    try:
        photo = extract_feature(encoder, img)

        for mk in ["LSTM", "RNN", "GRU"]:
            model = load_caption_model(mk)
            t0 = time.time()

            if mode.startswith("Greedy"):
                cap = greedy_decode(model, tok, index_word, photo, max_length)
            else:
                cap = beam_decode(model, tok, index_word, photo, max_length, beam_size=beam_size)

            ms = (time.time() - t0) * 1000
            cap = post_process(cap, tone)
            results[mk] = (cap if cap else "No caption generated. Try Greedy mode.", ms)

        st.session_state.setdefault("history", [])
        st.session_state["history"].insert(
            0,
            {"image": getattr(uploaded, "name", "uploaded"),
             "LSTM": results["LSTM"][0],
             "RNN":  results["RNN"][0],
             "GRU":  results["GRU"][0]}
        )

    except Exception as e:
        st.error(str(e))

caption_card(c1, "🟣 LSTM Caption", results.get("LSTM", ("", 0))[0], results.get("LSTM", ("", 0))[1])
caption_card(c2, "🟠 RNN Caption",  results.get("RNN",  ("", 0))[0], results.get("RNN",  ("", 0))[1])
caption_card(c3, "🟢 GRU Caption",  results.get("GRU",  ("", 0))[0], results.get("GRU",  ("", 0))[1])

with st.expander("🧾 View history (optional)", expanded=False):
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet.")
    else:
        st.dataframe(pd.DataFrame(hist[:20]), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

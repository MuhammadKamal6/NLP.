import json
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input


# ---------------------------
# EDIT THESE (your GitHub Release)
# ---------------------------
OWNER = "MuhammadKamal6"
REPO  = "NLP."   # change to "NLP" if your repo name has no dot
TAG   = "v1"

ART_DIR = Path("artifacts")
TOKENIZER_PATH = ART_DIR / "tokenizer.json"
CONFIG_PATH    = ART_DIR / "config.json"

MODEL_FILES = {
    "LSTM": "last_lstm_model.keras",
    "RNN":  "last_rnn_model.keras",
    "GRU":  "last_gru_model.keras",
}


# ---------------------------
# PAGE + COMPACT STYLE (no scroll)
# ---------------------------
st.set_page_config(page_title="CaptionLab", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
      header, footer, #MainMenu { visibility: hidden; }
      .block-container {
        max-width: 1250px;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
      }
      /* Compact spacing between elements */
      div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }

      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(850px 550px at 12% 10%, rgba(14,165,233,.14), transparent 60%),
          radial-gradient(850px 550px at 88% 12%, rgba(168,85,247,.10), transparent 55%),
          linear-gradient(180deg, #F6F7FB 0%, #F3F5FA 100%);
      }

      .hero {
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.80);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 14px 32px rgba(15,23,42,.06);
      }
      .card {
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.78);
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 12px 28px rgba(15,23,42,.05);
      }
      .caption {
        font-size: 1.02rem;
        line-height: 1.45;
        color: #0F172A;
      }
      .muted { opacity: .75; color:#0F172A; font-size: .9rem; }
      .pill {
        display:inline-flex; align-items:center; gap:7px;
        padding: 5px 9px; border-radius: 999px;
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.85);
        font-size: .82rem;
        margin-right: 6px;
      }
      /* Nice button */
      div.stButton > button {
        border-radius: 12px !important;
        padding: .60rem .9rem !important;
        border: 1px solid rgba(14,165,233,.35) !important;
        background: linear-gradient(135deg, rgba(14,165,233,.95), rgba(168,85,247,.75)) !important;
        color: white !important;
        font-weight: 750 !important;
      }
      /* Make file uploader compact */
      [data-testid="stFileUploaderDropzone"]{
        padding: 0.75rem !important;
        border-radius: 14px !important;
        border: 1px dashed rgba(15,23,42,.25) !important;
        background: rgba(255,255,255,.72) !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# DOWNLOAD + LOAD
# ---------------------------
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


# ---------------------------
# INFERENCE
# ---------------------------
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


# ---------------------------
# UI (ONE SCREEN)
# ---------------------------
tok, index_word, max_length = load_tokenizer_config()
encoder = load_encoder()

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
        <div>
          <div style="font-size:1.35rem; font-weight:850; color:#0F172A;">✨ CaptionLab</div>
          <div class="muted">Turn any photo into a clean, human-friendly caption in seconds.</div>
        </div>
        <div style="text-align:right;">
          <span class="pill">🧠 3 Models</span>
          <span class="pill">⚡ Fast Inference</span>
          <span class="pill">🧾 Max len: <b>{max_length}</b></span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

topL, topR = st.columns([1.05, 0.95], gap="large")

with topL:
    st.markdown("#### 📤 Upload your image")
    uploaded = st.file_uploader("Drag & drop or browse", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

with topR:
    st.markdown("#### ⚙️ Caption controls")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
    with c1:
        decoding = st.selectbox("Mode", ["Beam Search (Best)", "Greedy (Fast)"], index=0)
    with c2:
        beam_size = st.slider("Beam", 1, 10, 5, disabled=(decoding.startswith("Greedy")))
    with c3:
        tone = st.selectbox("Style", ["Friendly", "Simple", "Short"], index=0)

    run = st.button("✨ Generate captions", use_container_width=True, disabled=(uploaded is None))

def post_process(text: str, tone_mode: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if tone_mode == "Short":
        # keep first ~8 words if long
        parts = text.split()
        text = " ".join(parts[:8]) if len(parts) > 8 else text
    if tone_mode == "Friendly":
        # sentence-case
        text = text[:1].upper() + text[1:]
    return text

# Result row: 3 cards side-by-side
col1, col2, col3 = st.columns(3, gap="large")

def caption_card(col, title, caption, ms):
    col.markdown(f"#### {title}")
    col.markdown('<div class="card">', unsafe_allow_html=True)
    if caption:
        col.markdown(f'<div class="caption">“{caption}”</div>', unsafe_allow_html=True)
        col.markdown(
            f'<span class="pill">⏱️ {ms:.0f} ms</span>',
            unsafe_allow_html=True
        )
        col.button("Copy", key=f"copy_{title}", help="Select the caption and copy it.")
    else:
        col.markdown('<div class="muted">Upload an image and click “Generate captions”.</div>', unsafe_allow_html=True)
    col.markdown("</div>", unsafe_allow_html=True)

if run and uploaded:
    try:
        photo = extract_feature(encoder, img)

        results = {}
        for mk in ["LSTM", "RNN", "GRU"]:
            model = load_caption_model(mk)
            t0 = time.time()
            if decoding.startswith("Greedy"):
                cap = greedy_decode(model, tok, index_word, photo, max_length)
            else:
                cap = beam_decode(model, tok, index_word, photo, max_length, beam_size=beam_size)
            ms = (time.time() - t0) * 1000
            cap = post_process(cap, tone)
            results[mk] = (cap if cap else "No caption generated. Try Greedy mode.", ms)

        st.session_state.setdefault("history", [])
        st.session_state["history"].insert(0, {"image": getattr(uploaded, "name", "uploaded"), **{k: v[0] for k, v in results.items()}})

    except Exception as e:
        st.error(str(e))
        results = {}

else:
    results = {}

caption_card(col1, "🟣 LSTM Caption", results.get("LSTM", ("", 0))[0], results.get("LSTM", ("", 0))[1])
caption_card(col2, "🟠 RNN Caption",  results.get("RNN",  ("", 0))[0], results.get("RNN",  ("", 0))[1])
caption_card(col3, "🟢 GRU Caption",  results.get("GRU",  ("", 0))[0], results.get("GRU",  ("", 0))[1])

# Keep extras collapsed to avoid scrolling
with st.expander("📦 Batch mode (optional) — caption multiple images", expanded=False):
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    batch_model = st.selectbox("Use model", ["LSTM", "RNN", "GRU"], index=0, key="batch_model")
    run_b = st.button("Run batch", disabled=not files)

    if run_b and files:
        model = load_caption_model(batch_model)
        rows = []
        for f in files:
            im = Image.open(f)
            p = extract_feature(encoder, im)
            if decoding.startswith("Greedy"):
                cap = greedy_decode(model, tok, index_word, p, max_length)
            else:
                cap = beam_decode(model, tok, index_word, p, max_length, beam_size=beam_size)
            cap = post_process(cap, tone)
            rows.append({"file": f.name, "model": batch_model, "caption": cap})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "captions.csv", "text/csv")

with st.expander("🧾 History (optional) — recent results", expanded=False):
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet.")
    else:
        st.dataframe(pd.DataFrame(hist[:20]), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

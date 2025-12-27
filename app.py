import json
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input


# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="Image Captioning Studio",
    page_icon="✨",
    layout="wide",
)

# =========================
# GLOBAL CSS (Full restyle)
# =========================
st.markdown(
    """
    <style>
      /* App background */
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1200px 600px at 15% 10%, rgba(124,58,237,0.30), transparent 50%),
          radial-gradient(900px 500px at 85% 20%, rgba(59,130,246,0.20), transparent 55%),
          radial-gradient(1000px 800px at 60% 80%, rgba(16,185,129,0.12), transparent 60%),
          linear-gradient(180deg, rgba(7,10,18,1) 0%, rgba(6,9,16,1) 100%);
      }

      /* Reduce top padding */
      .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1200px; }

      /* Sidebar styling */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(14,21,38,0.95) 0%, rgba(10,15,28,0.95) 100%);
        border-right: 1px solid rgba(148,163,184,0.15);
      }

      /* Typography */
      h1, h2, h3 { letter-spacing: -0.02em; }
      p, li { color: rgba(233,238,246,0.92); }

      /* Buttons */
      div.stButton > button {
        border-radius: 12px !important;
        padding: 0.70rem 1rem !important;
        border: 1px solid rgba(124,58,237,0.35) !important;
        background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(59,130,246,0.70)) !important;
        color: #ffffff !important;
        font-weight: 650 !important;
        transition: transform 0.05s ease-in-out, filter 0.15s ease-in-out;
      }
      div.stButton > button:hover { filter: brightness(1.06); }
      div.stButton > button:active { transform: scale(0.99); }

      /* File uploader */
      [data-testid="stFileUploaderDropzone"]{
        border: 1px dashed rgba(148,163,184,0.35) !important;
        border-radius: 14px !important;
        background: rgba(148,163,184,0.05) !important;
      }

      /* Inputs */
      [data-baseweb="select"] > div,
      [data-baseweb="input"] > div{
        border-radius: 12px !important;
      }

      /* Custom cards */
      .card {
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(148,163,184,0.06);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 14px 40px rgba(0,0,0,0.22);
      }

      .hero {
        border: 1px solid rgba(148,163,184,0.16);
        background: linear-gradient(135deg, rgba(124,58,237,0.20), rgba(59,130,246,0.10));
        border-radius: 22px;
        padding: 20px 20px;
        box-shadow: 0 14px 40px rgba(0,0,0,0.20);
      }

      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(148,163,184,0.05);
        font-size: 0.85rem;
        color: rgba(233,238,246,0.92);
        margin-right: 8px;
      }

      .muted { opacity: 0.78; font-size: 0.92rem; }
      .captionText { font-size: 1.10rem; line-height: 1.55; }

      /* Hide Streamlit default footer/menu */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }
      header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# CONFIG / PATHS
# =========================
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

TOKENIZER_PATH = ART_DIR / "tokenizer.json"
CONFIG_PATH = ART_DIR / "config.json"

# ✅ EDIT THESE to match your GitHub repo/release
OWNER = "MuhammadKamal6"
REPO  = "NLP."   # change to "NLP" if your repo name is without dot
TAG   = "v1"

MODEL_ASSET = {
    "LSTM": "last_lstm_model.keras",
    "RNN":  "last_rnn_model.keras",
    "GRU":  "last_gru_model.keras",
}

def release_url(filename: str) -> str:
    return f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{filename}"

MODEL_URLS = {k: release_url(v) for k, v in MODEL_ASSET.items()}
MODEL_LOCAL = {k: (ART_DIR / v) for k, v in MODEL_ASSET.items()}


# =========================
# HELPERS
# =========================
def safe_exists(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False

def mb(n: int) -> str:
    return f"{n/1024/1024:.1f} MB"

def download_model(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if safe_exists(out_path):
        return

    st.toast(f"Downloading {out_path.name}…", icon="⬇️")

    prog = st.progress(0)
    status = st.empty()

    try:
        with urllib.request.urlopen(url) as resp:
            total = resp.length or 0
            chunk = 1024 * 1024  # 1MB
            downloaded = 0

            with open(out_path, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)

                    if total > 0:
                        pct = min(int(downloaded * 100 / total), 100)
                        prog.progress(pct)
                        status.write(f"Downloaded {mb(downloaded)} / {mb(total)}")
                    else:
                        prog.progress(min(int((downloaded / (70*1024*1024)) * 100), 100))
                        status.write(f"Downloaded {mb(downloaded)}")

        prog.progress(100)
        status.empty()

    except HTTPError as e:
        prog.empty()
        status.empty()
        raise RuntimeError(
            f"Download failed (HTTP). Make sure the Release is published and asset exists.\nURL: {url}\n{e}"
        )
    except URLError as e:
        prog.empty()
        status.empty()
        raise RuntimeError(f"Network error while downloading model.\nURL: {url}\n{e}")


@st.cache_resource
def load_encoder():
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource
def load_tokenizer_config():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Missing artifacts/tokenizer.json (upload it in the repo).")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing artifacts/config.json (upload it in the repo).")

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_length = int(cfg["max_length"])
    index_word = {v: k for k, v in tok.word_index.items()}
    return tok, index_word, max_length

@st.cache_resource
def load_caption_model(model_key: str):
    download_model(MODEL_URLS[model_key], MODEL_LOCAL[model_key])
    return load_model(MODEL_LOCAL[model_key], compile=False)

def extract_feature(encoder, pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = encoder.predict(x, verbose=0)  # (1,1920)
    return feat

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
    return " ".join(words).strip()


# =========================
# SIDEBAR (Clean)
# =========================
with st.sidebar:
    st.markdown("## ✨ Captioning Studio")
    st.markdown('<div class="muted">A clean demo for your Flickr8k captioning models.</div>', unsafe_allow_html=True)
    st.write("")

    model_choice = st.selectbox("Model", ["LSTM", "RNN", "GRU"], index=0)

    decoding = st.radio("Decoding", ["Beam Search", "Greedy"], index=0, horizontal=True)
    beam_size = st.slider("Beam size", 1, 10, 5, disabled=(decoding == "Greedy"))

    st.write("")
    st.markdown("### 📦 Model source")
    st.markdown('<div class="muted">Downloaded from GitHub Releases and cached.</div>', unsafe_allow_html=True)
    st.code(MODEL_URLS[model_choice], language="text")

    st.write("")
    st.markdown("### ✅ Quick checks")
    st.write(f"- tokenizer.json: {'OK' if TOKENIZER_PATH.exists() else 'Missing'}")
    st.write(f"- config.json: {'OK' if CONFIG_PATH.exists() else 'Missing'}")


# =========================
# HERO HEADER
# =========================
st.markdown(
    """
    <div class="hero">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
        <div>
          <div style="font-size:0.95rem; opacity:0.85;">NLP Project • Image Captioning</div>
          <div style="font-size:2.1rem; font-weight:800; margin-top:6px;">Generate a caption for any image</div>
          <div class="muted" style="margin-top:6px;">
            Encoder: DenseNet201 • Decoder: LSTM / RNN / GRU • Decode: Greedy or Beam Search
          </div>
        </div>
        <div style="text-align:right;">
          <span class="pill">🧠 Model: <b style="margin-left:6px;">""" + model_choice + """</b></span><br/>
          <span class="pill">🧾 Decode: <b style="margin-left:6px;">""" + decoding + """</b></span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# =========================
# MAIN UI
# =========================
tab1, tab2 = st.tabs(["🖼️ Caption", "ℹ️ About"])

with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("### Upload an image")
        uploaded = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"])

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
        else:
            st.markdown('<div class="card muted">Tip: use clear images (people, animals, outdoor scenes) for best results.</div>', unsafe_allow_html=True)

        st.write("")
        generate_btn = st.button("✨ Generate Caption", use_container_width=True, disabled=(uploaded is None))

    with right:
        st.markdown("### Result")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if uploaded is None:
            st.markdown("**Waiting for an image…**")
            st.markdown('<div class="muted">Upload an image on the left to generate a caption.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if generate_btn:
                try:
                    tok, index_word, max_length = load_tokenizer_config()
                    encoder = load_encoder()
                    caption_model = load_caption_model(model_choice)

                    with st.spinner("Running model…"):
                        photo = extract_feature(encoder, img)

                        if decoding == "Greedy":
                            caption = greedy_decode(caption_model, tok, index_word, photo, max_length)
                        else:
                            caption = beam_decode(caption_model, tok, index_word, photo, max_length, beam_size=beam_size)

                    if not caption:
                        caption = "⚠️ Empty caption. Try Greedy or a smaller beam size."

                    st.markdown("**Generated Caption**")
                    st.markdown(f'<div class="captionText">“{caption}”</div>', unsafe_allow_html=True)

                    st.write("")
                    st.markdown(
                        f'<span class="pill">🧠 {model_choice}</span>'
                        f'<span class="pill">🧾 {decoding}</span>'
                        + (f'<span class="pill">🔎 Beam {beam_size}</span>' if decoding == "Beam Search" else ""),
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(str(e))

            else:
                st.markdown("**Ready.**")
                st.markdown('<div class="muted">Click “Generate Caption” to run inference.</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown(
        """
        <div class="card">
          <h3 style="margin-top:0;">How it works</h3>
          <ul>
            <li><b>DenseNet201</b> extracts a 1920-dim feature vector from the image.</li>
            <li>A decoder model (LSTM / RNN / GRU) predicts the caption word-by-word.</li>
            <li><b>Greedy</b> picks the best word each step. <b>Beam Search</b> keeps multiple best candidates.</li>
          </ul>
          <div class="muted">Note: Models are downloaded from GitHub Releases once and cached on the server.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")
st.markdown('<div class="muted" style="text-align:center;">Made for your NLP Image Captioning Project • Streamlit UI</div>', unsafe_allow_html=True)

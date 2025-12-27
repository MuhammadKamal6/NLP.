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


# ---------------------------
# Page + Styling
# ---------------------------
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .caption-card {
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.03);
      }
      .small-muted { opacity: 0.75; font-size: 0.9rem; }
      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(49,51,63,0.2);
        margin-right: 6px;
        font-size: 0.85rem;
        opacity: 0.9;
      }
      .title-row {
        display:flex; align-items:center; gap:12px;
      }
      .title-emoji { font-size: 2rem; }
      .title-text { margin: 0; line-height: 1.2; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Paths / Repo settings
# ---------------------------
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

TOKENIZER_PATH = ART_DIR / "tokenizer.json"
CONFIG_PATH = ART_DIR / "config.json"

# EDIT THESE to match your GitHub:
OWNER = "MuhammadKamal6"
REPO  = "NLP."   # change to "NLP" if your repo name has no dot
TAG   = "v1"

MODEL_ASSET = {
    "LSTM": "last_lstm_model.keras",
    "RNN":  "last_rnn_model.keras",
    "GRU":  "last_gru_model.keras",
}

def release_url(filename: str) -> str:
    # Public download URL for a GitHub release asset
    return f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{filename}"

MODEL_URLS = {k: release_url(v) for k, v in MODEL_ASSET.items()}
MODEL_LOCAL = {k: (ART_DIR / v) for k, v in MODEL_ASSET.items()}


# ---------------------------
# Utilities
# ---------------------------
def human_mb(num_bytes: int) -> str:
    return f"{num_bytes/1024/1024:.1f} MB"

def safe_exists(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False

def download_with_progress(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if safe_exists(out_path):
        return

    info = st.info(f"Downloading **{out_path.name}** from GitHub Releases…", icon="⬇️")
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
                        status.write(f"Downloaded {human_mb(downloaded)} / {human_mb(total)}")
                    else:
                        # no content-length
                        prog.progress(min(int((downloaded / (70*1024*1024)) * 100), 100))
                        status.write(f"Downloaded {human_mb(downloaded)}")

        prog.progress(100)
        info.empty()
        status.empty()
        st.success(f"Model ready: **{out_path.name}**", icon="✅")

    except HTTPError as e:
        info.empty()
        prog.empty()
        status.empty()
        raise RuntimeError(
            f"HTTP error while downloading model: {e}. "
            f"Check that the release is published and the asset exists:\n{url}"
        )
    except URLError as e:
        info.empty()
        prog.empty()
        status.empty()
        raise RuntimeError(f"Network error while downloading model: {e}")
    except Exception as e:
        info.empty()
        prog.empty()
        status.empty()
        raise RuntimeError(f"Unexpected download error: {e}")


@st.cache_resource
def load_encoder():
    # DenseNet201(include_top=False, pooling="avg") -> (1920,)
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")


@st.cache_resource
def load_tokenizer_config():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Missing artifacts/tokenizer.json in the repo.")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing artifacts/config.json in the repo.")

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_length = int(cfg["max_length"])
    index_word = {v: k for k, v in tok.word_index.items()}
    return tok, index_word, max_length


@st.cache_resource
def load_caption_model(model_key: str):
    download_with_progress(MODEL_URLS[model_key], MODEL_LOCAL[model_key])
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


def beam_search_decode(model, tokenizer, index_word, photo, max_length, beam_size=5) -> str:
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


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", ["LSTM", "RNN", "GRU"], index=0)
    decoding = st.radio("Decoding", ["Beam Search", "Greedy"], index=0)
    beam_size = st.slider("Beam size", 1, 10, 5, disabled=(decoding == "Greedy"))
    st.divider()

    st.subheader("📦 Model source")
    st.write("Models are downloaded from GitHub Releases:")
    st.code(MODEL_URLS[model_choice], language="text")
    st.caption("If download fails: verify the Release is **published** and assets exist under tag v1.")


# ---------------------------
# Main Layout
# ---------------------------
st.markdown(
    """
    <div class="title-row">
      <div class="title-emoji">🖼️</div>
      <div>
        <h1 class="title-text">Image Captioning Demo</h1>
        <div class="small-muted">DenseNet201 encoder + (LSTM / RNN / GRU) caption decoder</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="pill">Upload an image</span>
    <span class="pill">Choose a model</span>
    <span class="pill">Generate caption</span>
    """,
    unsafe_allow_html=True,
)

colA, colB = st.columns([1.2, 1], gap="large")

with colA:
    st.subheader("Upload")
    uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

with colB:
    st.subheader("Result")
    if not uploaded:
        st.info("Upload an image to generate a caption.", icon="👈")
    else:
        try:
            tok, index_word, max_length = load_tokenizer_config()
            encoder = load_encoder()
            caption_model = load_caption_model(model_choice)

            with st.spinner("Extracting features + generating caption…"):
                photo = extract_feature(encoder, img)

                if decoding == "Greedy":
                    caption = greedy_decode(caption_model, tok, index_word, photo, max_length)
                else:
                    caption = beam_search_decode(caption_model, tok, index_word, photo, max_length, beam_size=beam_size)

            st.markdown('<div class="caption-card">', unsafe_allow_html=True)
            st.markdown("**Generated caption**")
            st.write(caption if caption else "⚠️ Empty caption (try another image or smaller beam size).")
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"Model: {model_choice} • Decoding: {decoding}" + (f" • Beam: {beam_size}" if decoding == "Beam Search" else ""))

        except Exception as e:
            st.error(str(e))
            st.stop()

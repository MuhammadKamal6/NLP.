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
# CONFIG (EDIT THESE)
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
# PAGE + STYLE (Different look)
# ---------------------------
st.set_page_config(page_title="CaptionLab", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1.2rem; padding-bottom: 2.2rem; }
      header, footer, #MainMenu { visibility: hidden; }

      /* Light, clean studio look */
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(900px 600px at 10% 10%, rgba(14,165,233,.15), transparent 60%),
          radial-gradient(900px 600px at 85% 15%, rgba(34,197,94,.10), transparent 55%),
          radial-gradient(1000px 700px at 50% 90%, rgba(168,85,247,.10), transparent 60%),
          linear-gradient(180deg, #F6F7FB 0%, #F3F5FA 100%);
      }

      .hero {
        border: 1px solid rgba(15,23,42,.12);
        background: linear-gradient(135deg, rgba(255,255,255,.85), rgba(255,255,255,.55));
        border-radius: 20px;
        padding: 16px 18px;
        box-shadow: 0 18px 45px rgba(15,23,42,.08);
      }

      .card {
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.75);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 16px 40px rgba(15,23,42,.06);
      }

      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,.10);
        background: rgba(255,255,255,.75);
        font-size: .85rem;
        margin-right: 8px;
      }

      .caption {
        font-size: 1.12rem;
        line-height: 1.55;
        color: #0F172A;
      }

      /* nicer buttons */
      div.stButton > button {
        border-radius: 12px !important;
        padding: .7rem 1rem !important;
        border: 1px solid rgba(14,165,233,.35) !important;
        background: linear-gradient(135deg, rgba(14,165,233,.95), rgba(168,85,247,.75)) !important;
        color: white !important;
        font-weight: 700 !important;
      }
      div.stButton > button:hover { filter: brightness(1.05); }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# DOWNLOAD + LOAD HELPERS
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

    with st.spinner(f"Downloading {out_path.name} (one-time)…"):
        try:
            urllib.request.urlretrieve(url, out_path)
        except HTTPError as e:
            raise RuntimeError(
                f"Download failed (HTTP). Check Release is published + asset exists.\nURL: {url}\n{e}"
            )
        except URLError as e:
            raise RuntimeError(f"Network error while downloading.\nURL: {url}\n{e}")

@st.cache_resource
def load_encoder():
    return DenseNet201(weights="imagenet", include_top=False, pooling="avg")  # 1920-D

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
# CAPTIONING
# ---------------------------
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


# ---------------------------
# HEADER (Top bar look)
# ---------------------------
tok, index_word, max_length = load_tokenizer_config()
encoder = load_encoder()

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
        <div>
          <div style="opacity:.75; font-size:.95rem;">NLP Project • Image Captioning</div>
          <div style="font-size:2.1rem; font-weight:850; margin-top:4px; color:#0F172A;">CaptionLab</div>
          <div style="opacity:.75; margin-top:6px; color:#0F172A;">
            Single • Compare models • Batch captions • Download results
          </div>
        </div>
        <div style="text-align:right;">
          <span class="pill">📦 Encoder: <b>DenseNet201</b></span>
          <span class="pill">🧾 Max length: <b>{max_length}</b></span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------
# GLOBAL CONTROLS (no sidebar)
# ---------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.1])
with c1:
    decoding = st.selectbox("Decoding", ["Beam Search", "Greedy"], index=0)
with c2:
    beam_size = st.slider("Beam size", 1, 10, 5, disabled=(decoding == "Greedy"))
with c3:
    default_model = st.selectbox("Default model", ["LSTM", "RNN", "GRU"], index=0)
with c4:
    st.caption("Tip: Compare tab makes your project look unique ✅")

tabs = st.tabs(["🖼️ Single", "🆚 Compare", "📦 Batch", "🧾 History"])


# ---------------------------
# TAB: Single
# ---------------------------
with tabs[0]:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Upload")
        up = st.file_uploader("JPG / PNG", type=["jpg", "jpeg", "png"], key="single_up")
        chosen_model = st.selectbox("Model", ["LSTM", "RNN", "GRU"], index=["LSTM","RNN","GRU"].index(default_model))
        run = st.button("✨ Generate Caption", use_container_width=True, disabled=(up is None))

        if up:
            img = Image.open(up)
            st.image(img, use_container_width=True)

    with right:
        st.markdown("### Result")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if not up:
            st.write("Upload an image to generate a caption.")
        elif run:
            try:
                model = load_caption_model(chosen_model)
                photo = extract_feature(encoder, img)

                t0 = time.time()
                if decoding == "Greedy":
                    cap = greedy_decode(model, tok, index_word, photo, max_length)
                else:
                    cap = beam_decode(model, tok, index_word, photo, max_length, beam_size=beam_size)
                ms = (time.time() - t0) * 1000

                if not cap:
                    cap = "Empty caption. Try Greedy or smaller beam size."

                st.markdown("**Generated caption**")
                st.markdown(f'<div class="caption">“{cap}”</div>', unsafe_allow_html=True)

                st.write("")
                st.markdown(
                    f'<span class="pill">🧠 {chosen_model}</span>'
                    f'<span class="pill">🧾 {decoding}</span>'
                    + (f'<span class="pill">🔎 Beam {beam_size}</span>' if decoding == "Beam Search" else "")
                    + f'<span class="pill">⏱️ {ms:.0f} ms</span>',
                    unsafe_allow_html=True
                )

                st.session_state.setdefault("history", [])
                st.session_state["history"].insert(0, {"mode": "Single", "model": chosen_model, "caption": cap})

            except Exception as e:
                st.error(str(e))

        else:
            st.write("Ready. Click **Generate Caption**.")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# TAB: Compare (key difference vs your screenshot)
# ---------------------------
with tabs[1]:
    st.markdown("### Compare LSTM vs RNN vs GRU (same image)")
    upc = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="cmp_up")
    run_cmp = st.button("🆚 Generate for all models", disabled=(upc is None), use_container_width=True)

    if upc:
        imgc = Image.open(upc)
        st.image(imgc, use_container_width=True)

    if upc and run_cmp:
        photo = extract_feature(encoder, imgc)

        cols = st.columns(3, gap="large")
        results = []
        for col, mk in zip(cols, ["LSTM", "RNN", "GRU"]):
            with col:
                st.markdown(f"#### {mk}")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                try:
                    model = load_caption_model(mk)
                    if decoding == "Greedy":
                        cap = greedy_decode(model, tok, index_word, photo, max_length)
                    else:
                        cap = beam_decode(model, tok, index_word, photo, max_length, beam_size=beam_size)

                    st.markdown(f'<div class="caption">“{cap}”</div>', unsafe_allow_html=True)
                    results.append({"mode": "Compare", "model": mk, "caption": cap})
                except Exception as e:
                    st.error(str(e))
                st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.setdefault("history", [])
        st.session_state["history"] = results + st.session_state["history"]


# ---------------------------
# TAB: Batch (another big difference)
# ---------------------------
with tabs[2]:
    st.markdown("### Batch captioning (multiple images → table → download CSV)")
    model_for_batch = st.selectbox("Model for batch", ["LSTM", "RNN", "GRU"], index=["LSTM","RNN","GRU"].index(default_model))
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        run_b = st.button("📦 Run batch", use_container_width=True)
        if run_b:
            model = load_caption_model(model_for_batch)
            rows = []

            for f in files:
                img = Image.open(f)
                photo = extract_feature(encoder, img)
                if decoding == "Greedy":
                    cap = greedy_decode(model, tok, index_word, photo, max_length)
                else:
                    cap = beam_decode(model, tok, index_word, photo, max_length, beam_size=beam_size)

                rows.append({"file": f.name, "model": model_for_batch, "caption": cap})

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="captions.csv", mime="text/csv")


# ---------------------------
# TAB: History
# ---------------------------
with tabs[3]:
    st.markdown("### Session history")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet. Generate some captions first.")
    else:
        st.dataframe(pd.DataFrame(hist[:50]), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

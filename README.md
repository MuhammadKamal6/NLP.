# 🖼️ CaptionLab — Image Captioning Web App (EN + AR)

CaptionLab is an online **image captioning** app that generates a natural-language caption for any uploaded image using **DenseNet201 + (LSTM / RNN / GRU)**, and shows an **Arabic translation** under the English caption.

## ✨ Features
- Upload an image and get captions instantly
- **3 models shown together**: LSTM / RNN / GRU (side-by-side)
- Decoding options: **Beam Search** (better quality) or **Greedy** (faster)
- **Arabic translation** below each caption (RTL formatted)
- Clean UI designed to keep image + results on one screen
- Deployed online using **Streamlit**

---

## 🧠 Model Overview
- **Encoder**: DenseNet201 (ImageNet) to extract a 1920-dim feature vector
- **Decoder**: Sequence model (LSTM / RNN / GRU) to generate the caption word-by-word
- Dataset: **Flickr8k**

---

## 📁 Project Structure
```

app.py
requirements.txt
runtime.txt                  # (recommended) python-3.12
artifacts/
tokenizer.json
config.json
last_lstm_model.keras      # downloaded from GitHub Releases
last_rnn_model.keras       # downloaded from GitHub Releases
last_gru_model.keras       # downloaded from GitHub Releases

````

> Note: the `.keras` model files are large, so they should be uploaded to **GitHub Releases** (not normal repo upload).

---

## 🚀 Run Locally
### 1) Install dependencies
```bash
pip install -r requirements.txt
````

### 2) Run the app

```bash
streamlit run app.py
```

Open:

* [http://localhost:8501](http://localhost:8501)

---

## 🌐 Deploy on Streamlit Cloud (Recommended)

### 1) Push code to GitHub

Make sure these files exist in your repo:

* `app.py`
* `requirements.txt`
* `artifacts/tokenizer.json`
* `artifacts/config.json`

### 2) Upload models to GitHub Releases

Go to:

* **Releases → New release → Tag: `v1`**
  Attach these assets:
* `last_lstm_model.keras`
* `last_rnn_model.keras`
* `last_gru_model.keras`

### 3) Add `runtime.txt` (important)

Create `runtime.txt` to avoid TensorFlow/Python mismatch:

```txt
python-3.12
```

### 4) Deploy

* Go to Streamlit Cloud → **New app**
* Select your repo + `app.py`
* Deploy ✅

> The app will automatically download the models from your GitHub Release assets the first time it runs.

---

## ⚙️ Configuration

`artifacts/config.json` should include at least:

```json
{
  "max_length": 34,
  "vocab_size": 8000
}
```

---

## 🔧 Requirements

Example `requirements.txt`:

```txt
streamlit
tensorflow==2.20.0
numpy
pillow
pandas
deep-translator
```

---

## ✅ Notes / Troubleshooting

### TensorFlow install error on Streamlit

If you see:

> No matching distribution found for tensorflow==X

Fix:

* Use `runtime.txt` with **python-3.12**
* Pin TensorFlow to an available version (e.g. `tensorflow==2.20.0`)

### Model download fails

* Ensure your release is **Published**
* Ensure file names match exactly:

  * `last_lstm_model.keras`
  * `last_rnn_model.keras`
  * `last_gru_model.keras`
* Ensure your `OWNER / REPO / TAG` values in `app.py` are correct

---

## 📸 Demo

* Live app: **https://imagecaption1w.streamlit.app/**
* GitHub repo: **https://github.com/MuhammadKamal6/NLP./tree/main**

---

## 🙏 Acknowledgements

* Flickr8k dataset
* TensorFlow/Keras
* Streamlit

---


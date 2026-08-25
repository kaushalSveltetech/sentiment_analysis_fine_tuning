# 🎬 IMDB Sentiment Analysis — DistilBERT Fine-Tuning + Streamlit App

A complete, from-scratch fine-tuning project for binary sentiment classification on the IMDB movie review dataset, using `distilbert-base-uncased`. Includes a production-style training script with train/evaluate/predict modes and an interactive Streamlit app for serving predictions.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Transformers](https://img.shields.io/badge/🤗%20transformers-fine--tuning-yellow)
![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b)
<!-- ![License](https://img.shields.io/badge/license-MIT-green) -->

---

## 📊 Results

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 93.7% | **94.2%** |
| Precision | 0.931 | 0.937 |
| Recall | 0.945 | 0.947 |
| F1 | 0.938 | **0.942** |

Trained for 2 epochs on a full stratified 80/10/10 split of the cleaned, deduplicated IMDB dataset (49,582 reviews after removing 418 duplicates).

<details>
<summary>Full classification report</summary>

```
              precision    recall  f1-score   support
    NEGATIVE       0.95      0.94      0.94      2469
    POSITIVE       0.94      0.95      0.94      2489
    accuracy                           0.94      4958
   macro avg       0.94      0.94      0.94      4958
weighted avg       0.94      0.94      0.94      4958
```

Confusion matrix:

```
[[2310  159]
 [ 131 2358]]
```
</details>

---

## 📁 Project Structure

```
kaushal/
├── train_script.py     # Training / evaluation / prediction pipeline (CLI)
├── app.py               # Streamlit app for interactive predictions
├── requirements.txt     # Python dependencies
├── sentiment_model/     # Saved model + checkpoints (gitignored — generated locally)
│   └── final/            # Final fine-tuned model + tokenizer, loaded by app.py
└── review.txt            # Example input file for --text_file predictions (gitignored)
```

---

## 🚀 Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/kaushalSveltetech/sentiment_analysis_fine_tuning.git
cd sentiment_analysis_fine_tuning
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_script.py --mode train
```

This downloads and cleans the IMDB dataset, fine-tunes DistilBERT for 2 epochs, saves the model to `sentiment_model/final/`, and prints test-set metrics automatically. On a modern RTX GPU this takes roughly 3 minutes end-to-end.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a review and get an instant sentiment prediction with confidence scores.

---

## 🧠 Model Details

| | |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Binary sequence classification (POSITIVE / NEGATIVE) |
| Dataset | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) — full 50k reviews, deduplicated and cleaned |
| Max sequence length | 512 tokens (covers 87.1% of reviews without truncation) |
| Optimizer | AdamW, cosine LR schedule, 2e-5 learning rate |
| Precision | bf16 (Ampere+ GPUs) / fp16 fallback, TF32 matmuls enabled |
| Best model selection | Validation F1 |

See [`train_script.py`](./train_script.py) for the full preprocessing, tokenization, and training pipeline.

---

## 🛠️ Usage — `train_script.py`

The script has three modes, all controlled by `--mode`.

### Train (fresh run)

```bash
python train_script.py --mode train
```
Trains from scratch and overwrites any existing saved model at `sentiment_model/final/`.

### Train (resume from a checkpoint)

```bash
python train_script.py --mode train \
  --resume_from_checkpoint sentiment_model/checkpoint-4960 \
  --total_epochs 5
```
`--total_epochs` is the **total** epoch count to reach, not additional epochs — resuming a checkpoint saved at epoch 2 with `--total_epochs 5` trains epochs 3–5 only.

### Evaluate (no retraining)

```bash
python train_script.py --mode evaluate
```
Loads the saved model from `sentiment_model/final/` and scores it on the held-out test set.

### Predict (single example)

```bash
# Short text, no embedded quotes
python train_script.py --mode predict --text "Great movie, loved it"

# Longer text or text with quotes/apostrophes — use a file instead
python train_script.py --mode predict --text_file review.txt
```

---

## 💻 Usage — `app.py` (Streamlit)

```bash
streamlit run app.py
```

**Features:**
- **Single review tab** — paste a review, get label + confidence + probability breakdown
- **Batch tab** — paste multiple reviews (one per line) or upload a `.txt` file, get a results table
- Runs on GPU automatically if available, otherwise CPU

> The app expects a trained model at `sentiment_model/final/`. Run `python train_script.py --mode train` first if that directory doesn't exist yet.

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt). Core dependencies: `torch`, `transformers`, `datasets`, `scikit-learn`, `beautifulsoup4`, `streamlit`.

---


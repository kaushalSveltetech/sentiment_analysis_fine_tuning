import re

import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "sentiment_model/final"
MAX_LENGTH = 512

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_id = int(probs.argmax())
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return {
        "label": label,
        "confidence": confidence,
        "neg_prob": float(probs[0]),
        "pos_prob": float(probs[1]),
        "token_count": len(inputs["input_ids"][0]),
    }


st.title("🎬 Movie Review Sentiment Analyzer")
st.caption(
    "DistilBERT fine-tuned on the IMDB dataset — 94.2% test accuracy, F1 0.942"
)

try:
    tokenizer, model, device = load_model()
except OSError:
    st.error(
        f"No trained model found at `{MODEL_DIR}`. "
        f"Run `python sentiment_train.py --mode train` first to produce it."
    )
    st.stop()

st.info(f"Model loaded on **{device.upper()}**", icon="✅")

tab_single, tab_batch = st.tabs(["Single review", "Batch (multiple reviews)"])

with tab_single:
    example_texts = {
        "-- Select an example --": "",
        "Positive example": (
            "This is just a precious little diamond. The play, the script are "
            "excellent. I can't compare this movie with anything else. Buy this one!"
        ),
        "Negative example": (
            "What a waste of time. The plot made no sense, the acting was wooden, "
            "and I nearly fell asleep halfway through. Would not recommend."
        ),
    }
    # choice = st.selectbox("Try an example or write your own below:", list(example_texts.keys()))

    text_input = st.text_area(
        "Review text",
        # value=example_texts[choice],
        height=180,
        placeholder="Paste a movie review here...",
    )

    if st.button("Analyze sentiment", type="primary", disabled=not text_input.strip()):
        result = predict(text_input, tokenizer, model, device)

        if result["label"] == "POSITIVE":
            st.success(f"**POSITIVE** — {result['confidence']:.1%} confidence")
        else:
            st.error(f"**NEGATIVE** — {result['confidence']:.1%} confidence")

        col1, col2 = st.columns(2)
        col1.metric("Negative probability", f"{result['neg_prob']:.1%}")
        col2.metric("Positive probability", f"{result['pos_prob']:.1%}")

        st.progress(result["pos_prob"], text="Positive ↑ / Negative ↓")

        with st.expander("Details"):
            st.write(f"Tokens used: {result['token_count']} / {MAX_LENGTH} max")
            if result["token_count"] >= MAX_LENGTH:
                st.warning("Input was truncated — only the first 512 tokens were used.")

with tab_batch:
    st.write("Paste multiple reviews, one per line, or upload a `.txt` file (one review per line).")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    batch_text = st.text_area("Or paste reviews here (one per line)", height=180)

    lines = []
    if uploaded_file is not None:
        lines = uploaded_file.read().decode("utf-8").splitlines()
    elif batch_text.strip():
        lines = batch_text.splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    if st.button("Analyze batch", type="primary", disabled=not lines):
        rows = []
        progress_bar = st.progress(0.0)
        for i, line in enumerate(lines):
            result = predict(line, tokenizer, model, device)
            rows.append(
                {
                    "review": line if len(line) < 80 else line[:77] + "...",
                    "label": result["label"],
                    "confidence": f"{result['confidence']:.1%}",
                }
            )
            progress_bar.progress((i + 1) / len(lines))
        st.dataframe(rows, use_container_width=True)

st.divider()
st.caption(
    "Model: distilbert-base-uncased fine-tuned on IMDB (49,582 cleaned reviews, "
    "80/10/10 split, max_length=512)."
)
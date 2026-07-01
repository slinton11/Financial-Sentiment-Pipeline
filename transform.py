"""
transform.py — Layer 2: NLP Transformation

STAGE A: Text preprocessing (clean title + description)
STAGE B: FinBERT sentiment scoring
STAGE C: MiniLM embeddings

Usage:
    python transform.py NVDA
"""

import os
import re
import sys
import json
import glob
import html
import unicodedata
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"

# --- Load FinBERT once at module level (not per-article) ---
# Loading the model is slow, so we do it a single time and reuse it.
print("Loading FinBERT model (first run downloads ~440MB)...")
FINBERT_NAME = "ProsusAI/finbert"
_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_NAME)
_model.eval()  # put model in inference mode (no training)

# FinBERT's output order is: positive, negative, neutral
FINBERT_LABELS = ["positive", "negative", "neutral"]
print("FinBERT loaded.\n")

# --- Load the embedding model once (MiniLM) ---
# This is a SEPARATE model from FinBERT. FinBERT judges sentiment;
# MiniLM converts text into embedding vectors for semantic search.
print("Loading embedding model (MiniLM)...")
EMBED_NAME = "all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBED_NAME)
print("Embedding model loaded.\n")


def find_latest_raw_file(ticker):
    """Find the most recent raw ingest file for this ticker."""
    pattern = os.path.join(DATA_DIR, f"{ticker}_*.json")
    files = glob.glob(pattern)

    # Exclude already-processed files (we name those *_clean.json)
    files = [f for f in files if "_clean" not in f]

    if not files:
        print(f"No raw data file found for {ticker}. Run ingest.py first.")
        return None

    # Most recent by filename (timestamps sort correctly as strings)
    latest = max(files)
    print(f"Using raw file: {latest}")
    return latest


def clean_text(text):
    """
    Clean a single piece of text:
    - return None if it's missing or junk
    - unescape HTML entities (&amp; -> &)
    - normalize unicode (smart quotes -> regular quotes)
    - strip leftover HTML tags
    - collapse whitespace
    """
    if not text:
        return None

    # NewsAPI uses "[Removed]" as a placeholder for deleted articles
    if text.strip() == "[Removed]":
        return None

    # Decode HTML entities like &amp; &quot; &#39;
    text = html.unescape(text)

    # Normalize unicode: convert fancy quotes/dashes to plain ASCII equivalents
    text = unicodedata.normalize("NFKD", text)

    # Remove any leftover HTML tags like <b> or <a href=...>
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    # If after cleaning there's basically nothing left, treat as junk
    if len(text) < 3:
        return None

    return text


def preprocess_articles(articles):
    """
    Clean each article's title + description, combine them,
    and drop anything that's junk after cleaning.
    Returns a list of cleaned article dicts.
    """
    cleaned = []
    dropped = 0

    for a in articles:
        title = clean_text(a.get("title"))
        description = clean_text(a.get("description"))

        # If both are junk, drop the whole article
        if not title and not description:
            dropped += 1
            continue

        # Combine into one text field for sentiment + embeddings.
        # Use whichever parts survived cleaning.
        parts = [p for p in [title, description] if p]
        combined_text = ". ".join(parts)

        cleaned.append({
            "title": title,
            "description": description,
            "combined_text": combined_text,
            "source": a.get("source"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
        })

    print(f"  Cleaned {len(cleaned)} articles, dropped {dropped} junk entries.")
    return cleaned


def score_sentiment(text):
    """
    Run one piece of text through FinBERT.
    Returns a dict with the winning label and all three probabilities.
    """
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    top_idx = int(probs.argmax())

    return {
        "label": FINBERT_LABELS[top_idx],
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2]),
    }


def embed_text(text):
    """
    Convert one piece of text into its embedding vector using MiniLM.
    Returns a list of 384 floats (the 'location on the meaning map').
    """
    vector = _embedder.encode(text)
    # .encode returns a numpy array; convert to a plain list so it's
    # JSON-serializable and easy to store later.
    return vector.tolist()

def run_transform(ticker):
    """Callable version of transform (for the dashboard). Returns output filepath."""
    ticker = ticker.upper()
    print(f"\n=== Transforming data for {ticker} ===\n")

    raw_file = find_latest_raw_file(ticker)
    if not raw_file:
        raise RuntimeError(f"No raw file for {ticker}. Ingest first.")

    with open(raw_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_articles = payload.get("news_data") or []
    print(f"Loaded {len(raw_articles)} raw articles.")

    cleaned_articles = preprocess_articles(raw_articles)

    print(f"\nScoring sentiment for {len(cleaned_articles)} articles...")
    for i, article in enumerate(cleaned_articles, start=1):
        article["sentiment"] = score_sentiment(article["combined_text"])
        if i % 20 == 0:
            print(f"  Scored {i}/{len(cleaned_articles)}...")

    print(f"\nGenerating embeddings for {len(cleaned_articles)} articles...")
    for i, article in enumerate(cleaned_articles, start=1):
        article["embedding"] = embed_text(article["combined_text"])
        if i % 20 == 0:
            print(f"  Embedded {i}/{len(cleaned_articles)}...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_DIR, f"{ticker}_{timestamp}_clean.json")
    out_payload = {
        "ticker": ticker,
        "processed_at": timestamp,
        "price_data": payload.get("price_data"),
        "cleaned_articles": cleaned_articles,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved cleaned data to: {out_file}")
    return out_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python transform.py <TICKER>")
        sys.exit(1)
    run_transform(sys.argv[1])
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
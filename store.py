"""
store.py — Layer 3: Hybrid Storage

Loads the latest clean file for a ticker and populates TWO stores:
  - SQLite:   prices + article metadata + sentiment (tabular, for charts/queries)
  - ChromaDB: article embeddings (vectors, for semantic search)

A shared article_id links a ChromaDB vector back to its full SQLite record.

Usage:
    python store.py NVDA
"""

import os
import sys
import json
import glob
import sqlite3
import hashlib
from datetime import datetime

import chromadb

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "pipeline.db")      # SQLite file
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")        # ChromaDB folder


def find_latest_clean_file(ticker):
    """Find the most recent *_clean.json file for this ticker."""
    pattern = os.path.join(DATA_DIR, f"{ticker}_*_clean.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No clean file found for {ticker}. Run transform.py first.")
        return None

    latest = max(files)
    print(f"Using clean file: {latest}")
    return latest


def make_article_id(ticker, article):
    """
    Build a stable, unique ID for an article so the same article
    doesn't get stored twice. We hash ticker + url + title.
    """
    raw = f"{ticker}|{article.get('url')}|{article.get('title')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def init_sqlite():
    """Create the SQLite tables if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Price data: one row per ticker per day
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)

    # Articles: one row per article, with sentiment scores
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            article_id TEXT PRIMARY KEY,
            ticker TEXT,
            title TEXT,
            description TEXT,
            combined_text TEXT,
            source TEXT,
            published_at TEXT,
            url TEXT,
            sentiment_label TEXT,
            sentiment_positive REAL,
            sentiment_negative REAL,
            sentiment_neutral REAL
        )
    """)

    conn.commit()
    return conn


def store_prices(conn, ticker, price_data):
    """Insert price rows. INSERT OR REPLACE avoids duplicate-key errors."""
    if not price_data:
        print("  No price data to store.")
        return

    cur = conn.cursor()
    for row in price_data:
        cur.execute("""
            INSERT OR REPLACE INTO prices
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            row.get("Date"),
            row.get("Open"),
            row.get("High"),
            row.get("Low"),
            row.get("Close"),
            row.get("Volume"),
        ))
    conn.commit()
    print(f"  Stored {len(price_data)} price rows in SQLite.")


def store_articles_sql(conn, ticker, articles):
    """Insert article metadata + sentiment into SQLite."""
    cur = conn.cursor()
    for article in articles:
        article_id = article["article_id"]   # set earlier in main()
        s = article.get("sentiment", {})
        cur.execute("""
            INSERT OR REPLACE INTO articles
            (article_id, ticker, title, description, combined_text,
             source, published_at, url,
             sentiment_label, sentiment_positive,
             sentiment_negative, sentiment_neutral)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id,
            ticker,
            article.get("title"),
            article.get("description"),
            article.get("combined_text"),
            article.get("source"),
            article.get("publishedAt"),
            article.get("url"),
            s.get("label"),
            s.get("positive"),
            s.get("negative"),
            s.get("neutral"),
        ))
    conn.commit()
    print(f"  Stored {len(articles)} articles in SQLite.")


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def init_chroma():
    """Open a persistent ChromaDB and return a collection for our articles."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create means re-running won't crash if it already exists
    collection = client.get_or_create_collection(name="articles")
    return collection


def store_articles_chroma(collection, ticker, articles):
    """
    Upsert each article's embedding into ChromaDB.
    We attach metadata so a vector hit can link back to SQLite and
    so we can filter searches (e.g. by ticker).
    """
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for article in articles:
        ids.append(article["article_id"])
        embeddings.append(article["embedding"])
        documents.append(article["combined_text"])
        s = article.get("sentiment", {})
        metadatas.append({
            "ticker": ticker,
            "title": article.get("title") or "",
            "published_at": article.get("publishedAt") or "",
            "sentiment_label": s.get("label") or "",
            "url": article.get("url") or "",
        })

    # upsert = insert new, update existing. Safe to re-run.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    print(f"  Upserted {len(ids)} embeddings into ChromaDB.")

def run_store(ticker):
    """Callable version of store (for the dashboard)."""
    ticker = ticker.upper()
    print(f"\n=== Storing data for {ticker} ===\n")

    clean_file = find_latest_clean_file(ticker)
    if not clean_file:
        raise RuntimeError(f"No clean file for {ticker}. Transform first.")

    with open(clean_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    price_data = payload.get("price_data") or []
    articles = payload.get("cleaned_articles") or []

    for article in articles:
        article["article_id"] = make_article_id(ticker, article)

    conn = init_sqlite()
    store_prices(conn, ticker, price_data)
    store_articles_sql(conn, ticker, articles)
    conn.close()

    collection = init_chroma()
    store_articles_chroma(collection, ticker, articles)
    return ticker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python store.py <TICKER>")
        sys.exit(1)
    run_store(sys.argv[1])
    print("\n=== Done ===\n")

if __name__ == "__main__":
    main()
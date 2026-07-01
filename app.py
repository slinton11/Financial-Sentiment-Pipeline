"""
app.py — Streamlit dashboard for the Financial Sentiment Pipeline.

Run with:
    streamlit run app.py
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import sqlite3
import pandas as pd
from pipeline import run_full_pipeline

st.set_page_config(
    page_title="Financial Sentiment Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Sentiment & Semantic Intelligence")
st.caption(
    "Ingest financial news, score sentiment with FinBERT, "
    "and search it by meaning using embeddings + a vector database."
)

# ---------------------------------------------------------------------------
# Shared resources (cached so they load once, not on every rerun)
# ---------------------------------------------------------------------------
CHROMA_PATH = os.path.join("data", "chroma")


@st.cache_resource
def get_search_embedder():
    """Load the MiniLM embedding model once and reuse it."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_collection():
    """Connect to the persistent ChromaDB collection once and reuse it."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name="articles")


@st.cache_data
def get_available_tickers():
    """Return the sorted list of tickers currently stored in ChromaDB."""
    collection = get_chroma_collection()
    result = collection.get(include=["metadatas"])
    tickers = sorted(
        {m.get("ticker") for m in result["metadatas"] if m.get("ticker")}
    )
    return tickers


# ---------------------------------------------------------------------------
# SECTION 1: Select an existing company OR add a new one.
# The chosen company is stored in session_state and used everywhere below.
# ---------------------------------------------------------------------------
st.header("Company")

available = get_available_tickers()

# Initialise the active ticker once (default to the first one in the DB)
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = available[0] if available else None

select_col, add_col = st.columns(2)

# --- Left: pick from companies already in the database ---
with select_col:
    st.write("**Select a company already in the database:**")
    if available:
        default_index = (
            available.index(st.session_state.active_ticker)
            if st.session_state.active_ticker in available
            else 0
        )
        chosen = st.selectbox(
            "Company",
            options=available,
            index=default_index,
            label_visibility="collapsed",
        )
        st.session_state.active_ticker = chosen
    else:
        st.info("No companies yet — add one on the right to get started.")

# --- Right: add a brand-new company (runs the full pipeline) ---
with add_col:
    st.write("**Or add a new company:**")
    add_c1, add_c2 = st.columns([3, 1])
    with add_c1:
        new_ticker = st.text_input(
            "New ticker",
            placeholder="e.g. AAPL, MSFT, GOOGL",
            label_visibility="collapsed",
        )
    with add_c2:
        run_clicked = st.button(
            "Add", type="primary", use_container_width=True
        )

    if run_clicked:
        ticker = new_ticker.strip().upper()
        if not ticker:
            st.warning("Please enter a ticker symbol first.")
        else:
            with st.status(f"Processing {ticker}...", expanded=True) as status:
                try:
                    def progress(msg):
                        st.write(msg)

                    run_full_pipeline(ticker, progress=progress)

                    status.update(
                        label=f"Done — {ticker} is ready to explore.",
                        state="complete",
                    )
                    st.success(f"{ticker} processed successfully.")

                    # Refresh the ticker list and make the new one active
                    get_available_tickers.clear()
                    st.session_state.active_ticker = ticker
                    st.rerun()
                except Exception as e:
                    status.update(
                        label=f"Failed to process {ticker}.", state="error"
                    )
                    st.error(f"Something went wrong: {e}")

# The single source of truth for the rest of the app
active_ticker = st.session_state.active_ticker

if active_ticker:
    st.success(f"Active company: **{active_ticker}**")


# ---------------------------------------------------------------------------
# SECTION 2: Semantic search (scoped to the active company, no extra picker)
# ---------------------------------------------------------------------------
st.divider()
st.header("Semantic search")

if not active_ticker:
    st.info("Add a company above to start searching.")
else:
    st.write(
        f"Searching **{active_ticker}** news by **meaning**, not keywords. "
        "Try a concept like *'supply chain problems'* or *'raising capital'* — "
        "it finds related articles even when they don't contain those exact words."
    )

    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        query = st.text_input(
            "Search concept",
            placeholder="e.g. artificial intelligence chips",
            label_visibility="collapsed",
        )
    with search_col2:
        search_clicked = st.button(
            "Search", type="primary", use_container_width=True
        )

    if search_clicked:
        if not query.strip():
            st.warning("Type something to search for.")
        else:
            embedder = get_search_embedder()
            collection = get_chroma_collection()

            query_embedding = embedder.encode(query).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"ticker": active_ticker},
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            if not docs:
                st.info(f"No matching articles found for {active_ticker}.")
            else:
                st.write(
                    f"Top {len(docs)} matches in **{active_ticker}** "
                    f"for: *{query}*"
                )
                for meta, dist in zip(metas, dists):
                    similarity = 1 - dist
                    with st.container(border=True):
                        st.markdown(f"**{meta.get('title')}**")
                        st.caption(
                            f"{meta.get('sentiment_label')}  •  "
                            f"{meta.get('published_at')}  •  "
                            f"similarity {similarity:.3f}"
                        )
st.divider()
st.header("Price & sentiment over time")

DB_PATH = os.path.join("data", "pipeline.db")


@st.cache_data
def load_price_and_sentiment(ticker):
    """Pull this company's prices and per-day average sentiment from SQLite."""
    conn = sqlite3.connect(DB_PATH)

    # Prices: one row per day
    prices = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,),
    )

    # Articles: we'll turn sentiment labels into numbers and average per day
    articles = pd.read_sql_query(
        "SELECT published_at, sentiment_label FROM articles WHERE ticker = ?",
        conn, params=(ticker,),
    )
    conn.close()

    return prices, articles


def build_daily_sentiment(articles):
    """Convert sentiment labels to numbers and average by calendar day."""
    if articles.empty:
        return pd.DataFrame(columns=["date", "avg_sentiment"])

    # Map labels to a numeric score
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    articles = articles.copy()
    articles["score"] = articles["sentiment_label"].map(score_map)

    # published_at looks like "2026-06-15T17:32:15Z" — take just the date part
    articles["date"] = articles["published_at"].str[:10]

    daily = (
        articles.dropna(subset=["score"])
        .groupby("date")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_sentiment"})
    )
    return daily


if not active_ticker:
    st.info("Add a company above to see its price and sentiment.")
else:
    prices, articles = load_price_and_sentiment(active_ticker)

    if prices.empty:
        st.info(f"No price data stored for {active_ticker}.")
    else:
        daily_sentiment = build_daily_sentiment(articles)

        # Merge price + sentiment on date so they share an x-axis
        prices["date"] = prices["date"].str[:10]
        merged = prices.merge(daily_sentiment, on="date", how="left")

        # Build a Plotly chart with two y-axes: price (left), sentiment (right)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=merged["date"], y=merged["close"],
                name="Close price", line=dict(width=2),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=merged["date"], y=merged["avg_sentiment"],
                name="Avg sentiment", line=dict(width=2, dash="dot"),
                connectgaps=True,
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=f"{active_ticker}: price vs. news sentiment",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_yaxes(title_text="Close price ($)", secondary_y=False)
        fig.update_yaxes(
            title_text="Avg sentiment (-1 to +1)",
            secondary_y=True, range=[-1, 1],
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Sentiment is the daily average of article scores "
            "(positive = +1, neutral = 0, negative = −1). "
            "The dotted line connects across days with no news."
        )
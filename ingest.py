"""
ingest.py — Layer 1: Data Ingestion

Pulls recent news headlines + historical price data for a given ticker
and saves both to timestamped files in the data/ folder.

Usage:
    python ingest.py AAPL
"""

import os
import sys
import json
from datetime import datetime, timedelta

import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from .env (gets our NEWSAPI_KEY)
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Where we save everything
DATA_DIR = "data"


def get_price_data(ticker, days=30):
    """Download daily OHLCV price data for the ticker via yfinance."""
    print(f"Fetching {days} days of price data for {ticker}...")

    stock = yf.Ticker(ticker)
    # period uses yfinance shorthand: "1mo", "3mo", etc.
    # We'll request a bit more than 'days' to be safe, then trim.
    hist = stock.history(period=f"{days}d")

    if hist.empty:
        print(f"  WARNING: No price data returned for {ticker}. "
              f"Is the ticker symbol correct?")
        return None

    # Reset index so the Date becomes a normal column, then convert to records
    hist = hist.reset_index()
    hist["Date"] = hist["Date"].astype(str)  # make dates JSON-friendly

    # Keep only the columns we care about
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    records = hist[cols].to_dict(orient="records")

    print(f"  Got {len(records)} days of price data.")
    return records


def get_news_data(ticker, days=30):
    """Fetch recent news headlines mentioning the ticker via NewsAPI."""
    print(f"Fetching news headlines for {ticker}...")

    if not NEWSAPI_KEY:
        print("  ERROR: No NEWSAPI_KEY found. Check your .env file.")
        return None

    # NewsAPI 'everything' endpoint, searching for the ticker symbol
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,  # max per request on free tier
        "apiKey": NEWSAPI_KEY,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  ERROR: NewsAPI returned status {response.status_code}")
        print(f"  {response.text}")
        return None

    data = response.json()
    articles = data.get("articles", [])

    # Trim to just the fields we need
    headlines = []
    for a in articles:
        headlines.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
        })

    print(f"  Got {len(headlines)} headlines.")
    return headlines


def save_data(ticker, price_data, news_data):
    """Save both datasets to timestamped JSON files in data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "ticker": ticker,
        "fetched_at": timestamp,
        "price_data": price_data,
        "news_data": news_data,
    }

    filename = os.path.join(DATA_DIR, f"{ticker}_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved everything to: {filename}")
    return filename

def run_ingest(ticker):
    """
    Callable version of ingestion (for the dashboard).
    Returns the saved filepath, or raises an exception on failure.
    """
    ticker = ticker.upper()
    print(f"\n=== Ingesting data for {ticker} ===\n")

    price_data = get_price_data(ticker)
    news_data = get_news_data(ticker)

    if not price_data and not news_data:
        raise RuntimeError(f"Nothing fetched for {ticker}.")

    return save_data(ticker, price_data, news_data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <TICKER>")
        sys.exit(1)
    run_ingest(sys.argv[1])
    print("\n=== Done ===\n")

if __name__ == "__main__":
    main()
"""
pipeline.py — Runs the full pipeline for a ticker end to end.

Can be called from the dashboard (run_full_pipeline) or the command line.

Usage:
    python pipeline.py NVDA
"""

import sys

from ingest import run_ingest
from transform import run_transform
from store import run_store


def run_full_pipeline(ticker, progress=None):
    """
    Run ingest -> transform -> store for one ticker.
    `progress` is an optional callback(str) so the dashboard can show
    status messages. If None, we just print.
    """
    ticker = ticker.upper()

    def report(msg):
        if progress:
            progress(msg)
        else:
            print(msg)

    report(f"Ingesting {ticker}...")
    run_ingest(ticker)

    report(f"Running NLP + embeddings for {ticker}...")
    run_transform(ticker)

    report(f"Storing {ticker} in databases...")
    run_store(ticker)

    report(f"Done with {ticker}.")
    return ticker


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <TICKER>")
        sys.exit(1)
    run_full_pipeline(sys.argv[1])


if __name__ == "__main__":
    main()
"""
visualize.py — Visualize the embedding space with UMAP.

Pulls all stored embeddings from ChromaDB, reduces them from 384
dimensions down to 2, and plots them colored by ticker. Articles
about similar things should cluster together.

Usage:
    python visualize.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import umap
import chromadb

DATA_DIR = "data"
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")


def load_all_embeddings():
    """Pull every stored article's embedding + metadata out of ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="articles")

    # include=["embeddings", "metadatas"] tells Chroma to hand back
    # the actual vectors and metadata, not just the IDs.
    result = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(result["embeddings"])
    metadatas = result["metadatas"]

    print(f"Loaded {len(embeddings)} embeddings from ChromaDB.")
    return embeddings, metadatas


def reduce_dimensions(embeddings):
    """
    Use UMAP to collapse 384-dim vectors into 2-dim points we can plot.
    UMAP tries to keep points that were close in 384-dim space close
    in 2-dim space too, so the visual clustering is meaningful.
    """
    print("Running UMAP dimensionality reduction (this takes a moment)...")
    reducer = umap.UMAP(
        n_neighbors=15,    # how much local vs global structure to weigh
        min_dist=0.1,      # how tightly points can pack together
        metric="cosine",   # cosine is the right distance for text embeddings
        random_state=42,   # makes the layout reproducible
    )
    coords_2d = reducer.fit_transform(embeddings)
    print("Reduction complete.")
    return coords_2d


def plot(coords_2d, metadatas):
    """Scatter-plot the 2D points, colored by ticker."""
    tickers = [m.get("ticker", "UNKNOWN") for m in metadatas]
    unique_tickers = sorted(set(tickers))

    plt.figure(figsize=(11, 8))

    # Plot each ticker as its own colored group so we get a legend
    for ticker in unique_tickers:
        idxs = [i for i, t in enumerate(tickers) if t == ticker]
        xs = coords_2d[idxs, 0]
        ys = coords_2d[idxs, 1]
        plt.scatter(xs, ys, label=ticker, alpha=0.7, s=60)

    plt.title("Article Embedding Space (UMAP projection)", fontsize=14)
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.legend(title="Ticker")
    plt.tight_layout()

    out_path = os.path.join(DATA_DIR, "embedding_space.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")

    # Also pop it open in a window
    plt.show()


def main():
    embeddings, metadatas = load_all_embeddings()

    if len(embeddings) < 5:
        print("Not enough embeddings to visualize. Store more data first.")
        return

    coords_2d = reduce_dimensions(embeddings)
    plot(coords_2d, metadatas)


if __name__ == "__main__":
    main()
"""
search.py — Quick semantic search test against ChromaDB.

Type a concept (not keywords) and get back the most semantically
similar articles, even if they don't share any words with your query.

Usage:
    python search.py "supply chain problems"
    python search.py "new product launch" --ticker NVDA
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")

# Must be the SAME embedding model used in transform.py, or the
# search vector won't live in the same "meaning space" as the stored ones.
print("Loading embedding model...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Loaded.\n")


def search(query, ticker=None, n_results=5):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="articles")

    # Turn the user's query into an embedding using the same model
    query_embedding = _embedder.encode(query).tolist()

    # Optional filter: only search within one ticker's articles
    where = {"ticker": ticker.upper()} if ticker else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )

    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python search.py "your concept" [--ticker NVDA]')
        sys.exit(1)

    query = sys.argv[1]

    # Optional --ticker flag
    ticker = None
    if "--ticker" in sys.argv:
        idx = sys.argv.index("--ticker")
        if idx + 1 < len(sys.argv):
            ticker = sys.argv[idx + 1]

    scope = f" (ticker={ticker.upper()})" if ticker else ""
    print(f'Searching for: "{query}"{scope}\n')

    results = search(query, ticker=ticker)

    # ChromaDB returns parallel lists; index [0] because we sent one query
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print("No results found.")
        return

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # Lower distance = closer in meaning. We show it so you can see
        # how confident the match is.
        similarity = 1 - dist  # rough, readable similarity score
        print(f"--- Result {rank}  (similarity: {similarity:.3f}) ---")
        print(f"  {meta.get('title')}")
        print(f"  Sentiment: {meta.get('sentiment_label')}  |  "
              f"{meta.get('published_at')}")
        print()


if __name__ == "__main__":
    main()
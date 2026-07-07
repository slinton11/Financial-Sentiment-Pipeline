Financial Sentiment & Semantic Intelligence Pipeline
An end-to-end pipeline that ingests financial news, scores its sentiment with a finance-tuned transformer, embeds it into a vector space for semantic search, and correlates news sentiment against stock price movement — all explorable through an interactive dashboard.

The project bridges traditional tabular data work (prices, time series) with modern NLP engineering (transformers, embeddings, vector databases).


What it does
Ingests financial news headlines and daily price data for any stock ticker.
Cleans the raw, messy news text (HTML entities, unicode, placeholders).
Scores sentiment on each article using FinBERT, a transformer trained on financial language.
Embeds each article into a 384-dimensional vector so it can be searched by meaning rather than keywords.
Stores everything in a hybrid layer: a relational database for tabular data and a vector database for embeddings, linked by a shared article ID.
Surfaces it all in a dashboard with live ingestion, semantic search, and a price-vs-sentiment chart.


Architecture
The pipeline is built in four layers, each as a self-contained, callable module.

Layer 1: Ingestion        ->  ingest.py    (NewsAPI + yfinance)

Layer 2: NLP Transform    ->  transform.py (FinBERT + MiniLM embeddings)

Layer 3: Hybrid Storage   ->  store.py     (SQLite + ChromaDB)

Layer 4: Interface        ->  app.py       (Streamlit dashboard)

Orchestration             ->  pipeline.py  (chains layers 1-3)

Each layer reads the previous layer's output and writes the next layer's input, so any stage can be run and inspected independently from the command line, or the whole chain can be triggered at once from the dashboard.


Key design decisions
Two models, two jobs. Sentiment and embeddings are handled by different models on purpose. FinBERT (ProsusAI/finbert) classifies tone, because general sentiment models misread financial language — "the company crushed earnings" is positive in finance, negative everywhere else. MiniLM (all-MiniLM-L6-v2) generates the embeddings, because it produces better general-purpose vector geometry for similarity search than a classification model's output layer does.

Hybrid storage. Two kinds of data need two kinds of database. Prices, sentiment scores, and article metadata are structured and relational, so they live in SQLite. Embeddings are high-dimensional vectors that need fast nearest-neighbour search, so they live in ChromaDB. A shared article_id (an MD5 hash of ticker + url + title) links a vector back to its full record, so a semantic hit in ChromaDB can be joined to its row in SQLite.

Logic separated from interface. All real work lives in callable functions (run_ingest, run_transform, run_store). The dashboard is a thin layer that calls them. This means the pipeline can be driven from the terminal or the UI with zero duplicated logic.

The dashboard
Company selection — pick a company already in the database, or type a new ticker to run the full pipeline live (with a progress spinner). The chosen company becomes the active context for the whole app.
Semantic search — search the active company's news by concept. Typing "raising capital" surfaces a bond-offering article even though it shares no words with the query. This is matching by meaning, which keyword search cannot do.
Price & sentiment over time — the active company's closing price overlaid with a daily-average sentiment line, reading from SQLite. This is where the text analysis connects back to the market.

What this project demonstrates
Handling messy, unstructured text and turning it into clean, structured features.
Applying domain-specific transformers (FinBERT) rather than generic models.
Generating and storing embeddings, and using a vector database for semantic search — including metadata-filtered queries.
Designing a hybrid storage architecture and reasoning about when to use a relational vs. a vector store.
Separating logic from interface so the same code runs headless or in a UI.

# Finance RAG — Time‑Series Insights

A Retrieval‑Augmented Generation (RAG) app that combines **market data** (yfinance), **vector search** (ChromaDB), and **interactive charts** (Streamlit + Matplotlib) to surface daily insights with **temporal context** (SMA, 1‑day, 1‑month deltas).

> **Live stack**: Streamlit • ChromaDB • SentenceTransformers • yfinance • Matplotlib

---

##  What this app does

- **Embeds** daily market facts (close/open/high/low/volume) and stores them in a **Chroma** vector DB.  
- **Retrieves** facts via semantic search with optional **ticker filter**.  
- **Analyzes time‑series** (1‑month window): last close, 1‑day Δ, 1‑month Δ, and **SMA5/SMA10** overlays.  
- **Ready for Streamlit Cloud** deployment.

---

## Repository structure
```
fin_rag_project/
├─ app.py                  # Streamlit app (entrypoint)
├─ vector_store/           # Chroma persistent store (created locally)
├─ requirements.txt        # Python deps for Streamlit Cloud
├─ .gitignore
└─ README.md               # This file
```

> If you used a Notebook to ingest data, you can later move that code into an `ingest.py` script. For now it’s fine that you ingested locally.

---

##  How it works (high level)

1. **Ingestion** (done locally):  
   - Pull last month of OHLCV data from `yfinance` for tickers (e.g., `TSLA`, `GOOGL`, `MSFT`, `TECHM.NS`, `INFY.NS`, `TCS.NS`, `RELIANCE.NS`).  
   - Normalize data (handles MultiIndex, ensures `Date` column, uses `Close`/`Adj Close`).  
   - Create daily “facts” strings and embed with **SentenceTransformers/all‑MiniLM‑L6‑v2**.  
   - Store in **Chroma** persistent collection: `finance-docs`.

2. **Query** (Streamlit app – `app.py`):  
   - User enters a **natural language** question and optional **ticker**.  
   - We do a vector search in Chroma and show the top‑k matches.  
   - We also fetch **1‑month** of latest prices and plot **SMA5/SMA10** + summary metrics.

---

##  Local development (Windows / Anaconda)

```bash
# 1) Create & activate env (if you don’t already have one)
conda create -n finrag python=3.11 -y
conda activate finrag

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

The app expects a local **Chroma** collection at:  
`vector_store/` with collection name **finance-docs**.

If the collection is empty (first run on a new machine), `app.py` has a **tiny seed** so queries don’t crash on Cloud (see “Seeding” below).

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (you already did).  
2. Go to **share.streamlit.io** → **Create app**.  
3. Select your repo: `sripranay/fin_rag_project`  
4. Branch: `main`  
5. App file path: `app.py`  
6. Hit **Deploy**.

> Streamlit Cloud will install from `requirements.txt` and run `app.py`.

---

## Requirements

Minimal `requirements.txt`

```
streamlit>=1.36
chromadb>=0.5
sentence-transformers>=2.7
yfinance>=0.2.40
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
```

---

## Seeding the vector store (for first-time Cloud runs)

Streamlit Cloud doesn’t have your local `vector_store/` yet. To avoid errors on the very first query, `app.py` contains a tiny **seed** that runs if the collection is empty:

```python
# After: collection = client.get_collection(COLLECTION)
try:
    existing = collection.peek(1)
except Exception:
    existing = {"documents": []}

if not existing.get("documents"):
    seed_docs = ["Seed fact: RELIANCE.NS is listed on NSE."]
    seed_meta = [{"ticker": "RELIANCE.NS", "date": "seed"}]
    seed_ids  = ["seed-1"]
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seed_emb = _m.encode(seed_docs, normalize_embeddings=True).tolist()
    collection.add(documents=seed_docs, metadatas=seed_meta, ids=seed_ids, embeddings=seed_emb)
```

This ensures the app still **responds** even before you ingest real data into Chroma on the server.

---

##  Ingestion (what we ran)

You ran ingestion locally (Notebook) with logic equivalent to:

- Download 1‑month daily bars via `yfinance`  
- Normalize columns/dates  
- Compose fact strings per date  
- Embed with **all‑MiniLM‑L6‑v2**  
- `col.add()` to the `finance-docs` collection

> You can move this into a script (e.g., `ingest.py`) later if you want a CLI like:
> `python ingest.py --tickers TSLA,GOOGL,MSFT`

---

##  Query flow (at runtime)

- Encode the user question → query Chroma (**top_k**).  
- Optional `{ "ticker": "RELIANCE.NS" }` filter.  
- Show the ranked hits (doc + metadata + distance).  
- Render chart & metrics via `yfinance` live pull (1 month window).

---

##  Notes on data / compliance

- This app uses **public market data** via `yfinance`; it is **educational**.  
- No PII is collected.  
- If you add news/filings in the future, verify **usage rights** and **attribution**.

---

##  Troubleshooting

- **“No results”**: Remove ticker filter or ingest more data.  
- **Chroma not found**: Ensure `vector_store/` exists and is readable.  
- **Cloud errors**: Check `requirements.txt` formatting and app path (`app.py`).  
- **Plots not showing**: Do not set custom Matplotlib styles on Cloud; the app uses default settings.

---

##  Roadmap

- Add **news** and **filings** chunking/ingestion.  
- Add **risk metrics** (volatility, drawdown).  
- Add **RAGAS** for retrieval evaluation.  
- Add **period dropdown** (1/3/6/12 months).

---

## 📄 License

MIT (or add your preferred license).

---

### Authors

**Sri Pranay** — project owner  
**Assistant** — setup guidance & docs

---

##  Quick commands (Git)

```bash
# from your repo folder
git add README.md
git commit -m "Add complete README with setup, usage, and deployment"
git push
```

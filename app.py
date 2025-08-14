import os
import sys
import streamlit as st

# ---- Patch sqlite for environments with old sqlite (Streamlit Cloud, etc.) ----
try:
    import pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
# ------------------------------------------------------------------------------

import chromadb
from sentence_transformers import SentenceTransformer
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------- Name → ticker resolver (accept company names too) ----------
NAME_TO_TICKER = {
    # US
    "TESLA": "TSLA",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "META": "META", "FACEBOOK": "META",
    # India (NSE)
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS", "INFOSYS": "INFY.NS",
    "TCS": "TCS.NS", "TATA CONSULTANCY SERVICES": "TCS.NS",
    "TECHM": "TECHM.NS", "TECH MAHINDRA": "TECHM.NS",
}
def resolve_symbol(s: str | None) -> str | None:
    if not s:
        return s
    key = s.strip().upper().replace("&", "AND").replace(".", "")
    return NAME_TO_TICKER.get(key, s.strip())

# ---------- Paths & Chroma ----------
BASE = os.path.dirname(os.path.abspath(__file__))
store_dir = os.path.join(BASE, "vector_store")
os.makedirs(store_dir, exist_ok=True)
COLLECTION = "finance-docs"

try:
    client = chromadb.PersistentClient(path=store_dir)
except Exception:
    # Fallback for older Chroma installs
    from chromadb.config import Settings
    client = chromadb.Client(Settings(persist_directory=store_dir, anonymized_telemetry=False))

# Open or create the collection
try:
    collection = client.get_collection(COLLECTION)
except Exception:
    collection = client.create_collection(COLLECTION)

# If empty, add a tiny seed so the app can answer something immediately
try:
    existing = collection.peek(1)
except Exception:
    existing = {"documents": []}
if not existing.get("documents"):
    seed_docs  = ["Seed fact: RELIANCE.NS is listed on NSE."]
    seed_meta  = [{"ticker": "RELIANCE.NS", "date": "seed"}]
    seed_ids   = ["seed-1"]
    _m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seed_emb   = _m.encode(seed_docs, normalize_embeddings=True).tolist()
    collection.add(documents=seed_docs, metadatas=seed_meta, ids=seed_ids, embeddings=seed_emb)

# ---------- Model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ---------- Helper: Ingest market data into Chroma ----------
def ingest_more(tickers, period="1mo"):
    total = 0
    for raw in tickers:
        tkr = resolve_symbol(raw) or raw
        try:
            df = yf.download(
                tkr, period=period, interval="1d",
                progress=False, auto_adjust=True
            )
        except Exception:
            df = None

        if df is None or df.empty:
            st.warning(f"Skipping {tkr}: no data.")
            continue

        # Normalize
        df = df.reset_index()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Build docs
        docs, metas, ids = [], [], []
        for _, r in df.iterrows():
            d = r["Date"].strftime("%Y-%m-%d")
            o = float(r.get("Open", np.nan))
            h = float(r.get("High", np.nan))
            l = float(r.get("Low",  np.nan))
            c = float(r.get("Close", np.nan))
            v = r.get("Volume", 0)
            try:
                v = int(float(v))
            except Exception:
                v = 0
            docs.append(
                f"On {d}, {tkr} closed at {c:.2f}, opened at {o:.2f}, "
                f"high {h:.2f}, low {l:.2f}, with volume {v}."
            )
            metas.append({"ticker": tkr, "date": d})
            ids.append(f"{tkr}-{d}")

        if not docs:
            st.warning(f"Skipping {tkr}: nothing to ingest.")
            continue

        embs = model.encode(docs, normalize_embeddings=True).tolist()
        collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        st.success(f"✅ Ingested {tkr}: {len(docs)} facts")
        total += len(docs)
    if total:
        st.info(f"Done. Added {total} facts in total.")

# ---------- UI ----------
st.title("Finance RAG — Quick Search")
st.write("Query the facts and daily summaries you ingested.")

# Sidebar: data manager to ingest on the cloud
with st.sidebar:
    st.header("Data manager")
    ing_tkr = st.text_input("Ticker or company name to ingest", "TSLA")
    ing_period = st.selectbox("Ingest period", ["1mo", "3mo", "6mo", "1y"], index=0)
    if st.button("Ingest"):
        ingest_more([ing_tkr], period=ing_period)

# Main search inputs
query  = st.text_input("Your question:", "What's the latest daily summary for TSLA?")
ticker = st.text_input("Ticker filter (optional):", "TSLA")
top_k  = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    ticker_clean = resolve_symbol(ticker)
    where = {"ticker": ticker_clean} if ticker_clean else None

    res = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)

    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = list(zip(docs, metas, dists))

    if not hits:
        st.warning("No results. Use the sidebar to ingest this ticker first, or clear the filter.")
    else:
        for i, (doc, meta, dist) in enumerate(hits, start=1):
            st.markdown(f"**{i}.** {doc}")
            st.caption(f"meta={meta} · distance={dist:.4f}")

# ---------- Price chart ----------
st.divider()
st.subheader("Price chart")

period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=0)
chart_ticker = resolve_symbol(ticker) or "RELIANCE.NS"

try:
    data = yf.download(chart_ticker, period=period, interval="1d",
                       progress=False, auto_adjust=True).dropna()
    if not data.empty:
        close = data["Close"]
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
        day_change = last_close - prev_close
        day_pct = (day_change / prev_close * 100.0) if prev_close else 0.0

        period_first = float(close.iloc[0])
        period_change = last_close - period_first
        period_pct = (period_change / period_first * 100.0) if period_first else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Last close", f"{last_close:.2f}")
        c2.metric("1-day Δ", f"{day_change:+.2f}", f"{day_pct:+.2f}%")
        c3.metric(f"{period} Δ", f"{period_change:+.2f}", f"{period_pct:+.2f}%")

        sma5  = close.rolling(5,  min_periods=1).mean()
        sma10 = close.rolling(10, min_periods=1).mean()

        fig, ax = plt.subplots()
        ax.plot(close.index, close, label="Close")
        ax.plot(sma5.index, sma5, linestyle="--", label="SMA5")
        ax.plot(sma10.index, sma10, linestyle=":", label="SMA10")
        ax.set_title(f"{chart_ticker} Close (last {period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No price data found for that ticker.")
except Exception as e:
    st.warning(f"Could not load chart: {e}")

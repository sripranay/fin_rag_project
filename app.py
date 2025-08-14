import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import yfinance as yf
import matplotlib.pyplot as plt

os.makedirs(store_dir, exist_ok=True)

# ---------- Paths & Chroma ----------
BASE = os.path.dirname(os.path.abspath(__file__))
store_dir = os.path.join(BASE, "vector_store")
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
# After: collection = client.get_collection(COLLECTION)
# If empty collection, add a tiny seed so queries work in the cloud:
try:
    existing = collection.peek(1)
except Exception:
    existing = {"documents": []}

if not existing.get("documents"):
    # seed with minimal facts (or call a tiny ingestion function)
    seed_docs = ["Seed fact: RELIANCE.NS is listed on NSE."]
    seed_meta = [{"ticker": "RELIANCE.NS", "date": "seed"}]
    seed_ids  = ["seed-1"]
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seed_emb = _m.encode(seed_docs, normalize_embeddings=True).tolist()
    collection.add(documents=seed_docs, metadatas=seed_meta, ids=seed_ids, embeddings=seed_emb)



# ---------- Model ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


# ---------- UI: RAG search ----------
st.title("Finance RAG — Quick Search")
st.write("Query the facts and daily summaries you ingested.")

query  = st.text_input("Your question:", "What's the latest daily summary for RELIANCE.NS?")
ticker = st.text_input("Ticker filter (optional):", "RELIANCE.NS")
top_k  = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    # Clean ticker; allow blank to search across all
    ticker_clean = (ticker or "").strip()
    where = {"ticker": ticker_clean} if ticker_clean else None

    res = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)

    # Pack results safely
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = list(zip(docs, metas, dists))

    if not hits:
        st.warning("No results. Try removing the ticker filter or adding more data.")
    else:
        for i, (doc, meta, dist) in enumerate(hits, start=1):
            st.markdown(f"**{i}.** {doc}")
            st.caption(f"meta={meta} · distance={dist:.4f}")


# ---------- UI: Price chart + quick stats ----------
st.divider()
st.subheader("Price chart")

# Let the user choose the chart period
period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=0)

# If a ticker box is empty, default to RELIANCE.NS for the chart
chart_ticker = (ticker or "").strip() or "RELIANCE.NS"

try:
    # Fetch prices for selected period
    data = yf.download(chart_ticker, period=period, interval="1d",
                       progress=False, auto_adjust=True).dropna()

    if not data.empty:
        close = data["Close"]

        # ---- Quick stats
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

        # ---- SMA overlays
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

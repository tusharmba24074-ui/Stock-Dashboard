# app.py — Indian Stocks Intelligence (Streamlit) — Deploy to Streamlit Cloud
# Run on Streamlit Cloud (share.streamlit.io). Optional NewsAPI key for headlines.
# Educational prototype — not financial advice.

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Indian Stocks Intelligence", layout="wide")
st.title("📊 Indian Stocks Intelligence — Buy / Hold / Sell")
st.caption("Prototype: blended signals (trend, RSI, PE, volume) + news sentiment. Not investment advice.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    newsapi_key_input = st.text_input("NewsAPI key (optional) — or leave blank", value="")
    interval = st.selectbox("Price interval", ["1d", "1h"], index=0)
    lookback_days = st.slider("Lookback days", 60, 720, 180)
    rsi_period = st.number_input("RSI period", 7, 30, 14)
    ma_short = st.number_input("Short MA days (for charts)", 5, 60, 20)
    ma_long = st.number_input("Long MA days (for scoring)", 50, 200, 200)
    use_pe = st.checkbox("Use PE in scoring (when available)", value=True)
    use_volume = st.checkbox("Use volume thrust in scoring", value=True)
    batch_size = st.slider("Batch size (tickers per fetch)", 50, 300, 150)
    max_stocks = st.slider("Max stocks to scan", 100, 2600, 600, step=100)
    st.markdown("---")
    custom_list = st.text_area("Optional: paste tickers (comma-separated) e.g. RELIANCE.NS, TCS.NS", "")
    uploaded = st.file_uploader("Optional: upload tickers CSV (column: ticker)", type=["csv"])
    peps_file = st.file_uploader("Optional: upload peps.csv (cols: name,companies,notes,source)", type=["csv"])
    run_btn = st.button("Run Scan")

# Use secret if user provided in Streamlit secrets; prefer secrets over manual input
NEWSAPI_KEY = ""
if "NEWSAPI_KEY" in st.secrets:
    NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
if newsapi_key_input.strip():
    NEWSAPI_KEY = newsapi_key_input.strip()

analyzer = SentimentIntensityAnalyzer()

# ---------------- Helpers ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_universe_from_github():
    """
    Load NSE equity list from a public GitHub mirror to avoid NSE TLS/anti-bot issues.
    """
    url = "https://raw.githubusercontent.com/rohanrao619/NSE_Stocks/main/EQUITY_L.csv"
    df = pd.read_csv(url)
    symbol_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
    syms = df[symbol_col].astype(str).str.strip().dropna().unique().tolist()
    yahoo = [s + ".NS" for s in syms if s and "." not in s]
    return sorted(yahoo)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_batch(tickers, start, end, interval="1d"):
    """Fetch OHLCV for tickers using yfinance — returns dict[ticker]=DataFrame"""
    data = {}
    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, interval=interval, progress=False, threads=False)
            if not df.empty:
                data[tk] = df
        except Exception:
            pass
    return data

@st.cache_data(ttl=900, show_spinner=False)
def fetch_pe_mcap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("trailingPE", np.nan), info.get("marketCap", np.nan)
    except Exception:
        return np.nan, np.nan

def compute_indicators(df, ma_s, ma_l, rsi_n):
    d = df.copy()
    d["MA_S"] = d["Close"].rolling(int(ma_s)).mean()
    d["MA_L"] = d["Close"].rolling(int(ma_l)).mean()
    delta = d["Close"].diff()
    up = delta.where(delta > 0, 0).rolling(int(rsi_n)).mean()
    down = (-delta.where(delta < 0, 0)).rolling(int(rsi_n)).mean()
    rs = up / down.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    if "Volume" in d.columns:
        d["VOL20"] = d["Volume"].rolling(20).mean()
    else:
        d["VOL20"] = np.nan
    return d

@st.cache_data(ttl=600, show_spinner=False)
def get_news_sentiment(query, newsapi_key):
    """Return mean VADER compound score for latest headlines via NewsAPI."""
    if not newsapi_key:
        return np.nan
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 6, "apiKey": newsapi_key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return np.nan
        arts = r.json().get("articles", [])
        if not arts:
            return np.nan
        scores = []
        for a in arts:
            text = (a.get("title") or "") + ". " + (a.get("description") or "")
            scores.append(analyzer.polarity_scores(text)["compound"])
        return float(np.mean(scores)) if scores else np.nan
    except Exception:
        return np.nan

def score_row(close, ma_l, rsi, pe, vol, vol20, sent, use_pe=True, use_vol=True):
    score = 0
    # Trend/MA logic
    if not np.isnan(ma_l) and close > ma_l:
        score += 2
    # RSI windows
    if not np.isnan(rsi):
        if 40 < rsi < 65:
            score += 1
        if rsi > 70:
            score -= 1
    # PE
    if use_pe:
        if np.isnan(pe):
            score += 0
        elif pe < 30:
            score += 1
        elif pe > 60:
            score -= 1
    # Volume thrust
    if use_vol and not (np.isnan(vol) or np.isnan(vol20) or vol20 == 0):
        if (vol / vol20) > 1.3:
            score += 1
    # Sentiment
    if not np.isnan(sent):
        if sent > 0.15:
            score += 1
        elif sent < -0.15:
            score -= 1
    return int(score)

def label_from_score(score):
    if score >= 3:
        return "BUY"
    if score <= -1:
        return "SELL"
    return "HOLD"

# ---------------- Build universe ----------------
if uploaded is not None:
    try:
        df_u = pd.read_csv(uploaded)
        col = next((c for c in df_u.columns if c.lower() in ("ticker","symbol")), df_u.columns[0])
        universe_all = df_u[col].dropna().astype(str).str.strip().tolist()
    except Exception:
        st.warning("Uploaded CSV unreadable — falling back to default universe.")
        universe_all = load_nse_universe_from_github()
else:
    if custom_list.strip():
        universe_all = [t.strip() for t in custom_list.replace("\n", ",").split(",") if t.strip()]
    else:
        with st.spinner("Loading NSE universe (from GitHub mirror)…"):
            universe_all = load_nse_universe_from_github()

universe = universe_all[:max_stocks]
st.caption(f"Loaded {len(universe)} tickers (showing up to {max_stocks}).")

# PEP mapping (optional)
peps_df = None
if peps_file is not None:
    try:
        peps_df = pd.read_csv(peps_file)
        if "companies" in peps_df.columns:
            peps_df["companies"] = peps_df["companies"].fillna("").astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    except Exception:
        peps_df = None

# ---------------- Run scan ----------------
if run_btn:
    start = dt.date.today() - dt.timedelta(days=int(lookback_days))
    end = dt.date.today()
    results = []

    with st.spinner("Fetching data & computing signals (this can take time)…"):
        for batch in chunks(universe, batch_size):
            data_map = fetch_history_batch(batch, start, end, interval=interval)
            for tk, df in data_map.items():
                if df.empty:
                    continue
                ind = compute_indicators(df, ma_short, ma_long, rsi_period)
                last = ind.iloc[-1]
                price = float(last["Close"])
                ma_l = float(last["MA_L"]) if not np.isnan(last["MA_L"]) else np.nan
                rsi = float(last["RSI"]) if not np.isnan(last["RSI"]) else np.nan
                vol = float(last["Volume"]) if "Volume" in ind.columns else np.nan
                vol20 = float(last["VOL20"]) if "VOL20" in ind.columns else np.nan
                pe, mcap = fetch_pe_mcap(tk) if use_pe else (np.nan, np.nan)

                # headlines sentiment (ticker without suffix)
                q = tk.replace(".NS","").replace(".BO","")
                sent = get_news_sentiment(q, NEWSAPI_KEY) if NEWSAPI_KEY else np.nan

                score = score_row(price, ma_l, rsi, pe, vol, vol20, sent, use_pe=use_pe, use_vol=use_volume)
                label = label_from_score(score)

                results.append({
                    "Ticker": tk,
                    "Price": round(price,2),
                    "RSI": round(rsi,2) if not np.isnan(rsi) else np.nan,
                    "PE": round(pe,2) if not np.isnan(pe) else np.nan,
                    "Vol/20d": round((vol/vol20),2) if (not np.isnan(vol) and not np.isnan(vol20) and vol20!=0) else np.nan,
                    "Sentiment": round(sent,3) if not np.isnan(sent) else np.nan,
                    "Score": score,
                    "Label": label,
                    "MarketCap": mcap
                })

    if not results:
        st.error("No data returned. Try reducing Max stocks or increasing lookback.")
        st.stop()

    df_res = pd.DataFrame(results)
    order_map = {"BUY":0, "SELL":1, "HOLD":2}
    df_res["Order"] = df_res["Label"].map(order_map)
    df_res = df_res.sort_values(["Order","Score","MarketCap"], ascending=[True, False, False]).reset_index(drop=True)

    # PEP flags
    if peps_df is not None and "companies" in peps_df.columns:
        tk_to_peps = {}
        for _, row in peps_df.iterrows():
            nm = row.get("name","")
            for c in row.get("companies", []):
                tk_to_peps.setdefault(c, []).append(nm)
        df_res["PEP_Flag"] = df_res["Ticker"].apply(lambda t: "; ".join(tk_to_peps.get(t, [])) if t in tk_to_peps else "")
    else:
        df_res["PEP_Flag"] = ""

    st.success("Scan complete ✅")
    st.subheader("✅ BUY (top) → ❌ SELL → ⚖️ HOLD")
    st.dataframe(df_res[["Ticker","PEP_Flag","Price","RSI","PE","Vol/20d","Sentiment","Score","Label"]], use_container_width=True)

    # detail view
    st.markdown("---")
    st.header("🔎 Stock detail")
    sel = st.selectbox("Pick a ticker", df_res["Ticker"].tolist())
    if sel:
        hist = fetch_history_batch([sel], start, end, interval=interval).get(sel)
        if hist is not None and not hist.empty:
            ind = compute_indicators(hist, ma_short, ma_long, rsi_period)
            st.line_chart(ind[["Close","MA_S","MA_L"]])
            st.caption("RSI")
            st.line_chart(ind[["RSI"]])
else:
    st.info("Configure options in the sidebar and click Run Scan.")

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

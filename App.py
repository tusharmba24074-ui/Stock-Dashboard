# 📊 Indian Stock Market Dashboard (All NSE + BSE Stocks)
# - Loads NSE + BSE tickers automatically from exchange CSV files
# - Buy / Hold / Sell signals using MA + RSI
# - Candlestick charts with MA + RSI
# - Screener for all stocks

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np

st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")

# ---------------------------
# Load tickers from NSE + BSE
# ---------------------------
@st.cache_data
def load_all_tickers():
    # NSE all stocks
    nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    nse = pd.read_csv(nse_url)
    nse_symbols = nse["SYMBOL"].dropna().unique().tolist()
    nse_symbols = [s + ".NS" for s in nse_symbols]

    # BSE all stocks
    bse_url = "https://www.bseindia.com/corporates/List_Scrips.aspx"
    try:
        bse = pd.read_excel(bse_url)  # BSE provides XLS format
        bse_symbols = bse["Security Id"].dropna().unique().tolist()
        bse_symbols = [s + ".BO" for s in bse_symbols]
    except Exception:
        bse_symbols = []

    return list(set(nse_symbols + bse_symbols))

tickers = load_all_tickers()

# ---------------------------
# Stock Data Functions
# ---------------------------
@st.cache_data(ttl=300)
def get_stock_data(ticker, period="6mo"):
    try:
        return yf.download(ticker, period=period, interval="1d")
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    if df.empty:
        return df
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def get_signal(df):
    if df.empty:
        return "HOLD"
    latest = df.iloc[-1]
    if latest["Close"] > latest["MA20"] > latest["MA50"] and latest["RSI"] < 70:
        return "BUY"
    elif latest["Close"] < latest["MA20"] < latest["MA50"] and latest["RSI"] > 30:
        return "SELL"
    else:
        return "HOLD"

# ---------------------------
# UI Layout
# ---------------------------
st.title("📊 Indian Stock Market Dashboard — All NSE + BSE")
st.markdown("**Buy/Hold/Sell signals based on Moving Averages + RSI**")

tab1, tab2 = st.tabs(["📈 Stock Explorer", "📋 Market Screener"])

# ---------------------------
# Tab 1 - Single Stock Explorer
# ---------------------------
with tab1:
    stock_choice = st.selectbox("Choose a stock", tickers)
    period = st.selectbox("Select period", ["1mo", "3mo", "6mo", "1y", "2y"])

    df = get_stock_data(stock_choice, period)
    if not df.empty:
        df = compute_indicators(df)
        signal = get_signal(df)

        st.metric("Recommendation", signal)

        # Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df["Open"],
                                             high=df["High"],
                                             low=df["Low"],
                                             close=df["Close"],
                                             name="Candles")])
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], line=dict(color="blue", width=1), name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], line=dict(color="orange", width=1), name="MA50"))
        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # RSI Chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple"), name="RSI"))
        fig2.add_hline(y=70, line_dash="dot", line_color="red")
        fig2.add_hline(y=30, line_dash="dot", line_color="green")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available.")

# ---------------------------
# Tab 2 - Screener
# ---------------------------
with tab2:
    st.subheader("Market Screener (All NSE + BSE Stocks)")
    max_scan = st.slider("Number of stocks to scan (for speed)", 50, 1000, 200)

    signals = []
    for t in tickers[:max_scan]:
        df = get_stock_data(t, "3mo")
        if not df.empty:
            df = compute_indicators(df)
            sig = get_signal(df)
            signals.append({"Ticker": t, "Signal": sig, "Price": df["Close"].iloc[-1]})

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        buy_df = signals_df[signals_df["Signal"] == "BUY"]
        hold_df = signals_df[signals_df["Signal"] == "HOLD"]
        sell_df = signals_df[signals_df["Signal"] == "SELL"]

        st.markdown("### ✅ Buy Recommendations")
        st.dataframe(buy_df)

        st.markdown("### ⚪ Hold Recommendations")
        st.dataframe(hold_df)

        st.markdown("### ❌ Sell Recommendations")
        st.dataframe(sell_df)
    else:
        st.warning("No screener data available.")

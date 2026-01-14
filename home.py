import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# ================= CONFIG =================
st.set_page_config("AI High Gain Dashboard – KSA", layout="wide")

symbols_file = "tadawul_symbols.csv"
symbols_df = pd.read_csv(symbols_file)

symbols = symbols_df["Symbol"].dropna().unique().tolist()

# ================= DATA =================
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)

    if df.empty or len(df) < 60:
        return None

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.astype(float)

    df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
    df["EMA50"] = EMAIndicator(close, 50).ema_indicator()
    df["EMA200"] = EMAIndicator(close, 200).ema_indicator()

    df["RSI"] = RSIIndicator(close, 14).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd_diff()

    df["AvgVolume10"] = df["Volume"].rolling(10).mean()

    df.dropna(inplace=True)
    return df


# ================= ANALYSIS =================
rows = []

for sym in symbols:
    df = fetch_data(sym)
    if df is None:
        continue

    last = df.iloc[-1]
    prev10 = df.tail(10)

    score = 0
    reasons = []

    if last["EMA20"] > last["EMA50"]:
        score += 20; reasons.append("EMA20 > EMA50")
    if last["EMA50"] > last["EMA200"]:
        score += 20; reasons.append("EMA50 > EMA200")

    if 50 <= last["RSI"] <= 65:
        score += 15; reasons.append("RSI صحي")

    if last["MACD"] > 0:
        score += 15; reasons.append("MACD إيجابي")

    if last["Volume"] > last["AvgVolume10"]:
        score += 15; reasons.append("سيولة أعلى من المتوسط")

    green = (prev10["Close"].pct_change() > 0).sum()
    if green >= 6:
        score += 15; reasons.append("تجميع 10 شموع")

    rows.append({
        "Symbol": sym,
        "Price": round(last["Close"], 2),
        "EMA20": round(last["EMA20"], 2),
        "EMA50": round(last["EMA50"], 2),
        "EMA200": round(last["EMA200"], 2),
        "RSI": round(last["RSI"], 2),
        "MACD": round(last["MACD"], 4),
        "Volume": int(last["Volume"]),
        "AvgVolume10": int(last["AvgVolume10"]),
        "NextDayScore": score,
        "سبب الترشيح": " + ".join(reasons)
    })

df_all = pd.DataFrame(rows)

# ================= STRONG SETUPS =================
def trade_setup(row):
    if not (
        row["EMA20"] > row["EMA50"] and
        row["EMA50"] > row["EMA200"] and
        row["MACD"] > 0 and
        50 <= row["RSI"] <= 65 and
        row["Volume"] > row["AvgVolume10"]
    ):
        return None

    entry = row["Price"]
    stop = row["EMA20"]

    return pd.Series({
        "Entry": entry,
        "Stop Loss": round(stop, 2),
        "Target 1": round(entry * 1.05, 2),
        "Target 2": round(entry * 1.10, 2)
    })

setups = df_all.apply(trade_setup, axis=1)
strong_df = pd.concat([df_all, setups], axis=1).dropna(subset=["Entry"])

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

tab1, tab2, tab3 = st.tabs([
    "🔥 أفضل 20 سهم للغد",
    "💎 فرص قوية (دخول / وقف / أهداف)",
    "📊 السوق كامل"
])

# ---------- TAB 1 ----------
with tab1:
    top20 = df_all.sort_values("NextDayScore", ascending=False).head(20)
    st.dataframe(
        top20[
            [
                "Symbol","Price","EMA20","EMA50","EMA200",
                "RSI","MACD","NextDayScore","سبب الترشيح"
            ]
        ],
        use_container_width=True
    )

# ---------- TAB 2 ----------
with tab2:
    st.subheader("فرص قوية مكتملة فنيًا")
    st.dataframe(
        strong_df[
            [
                "Symbol","Price",
                "Entry","Stop Loss","Target 1","Target 2",
                "RSI","MACD","NextDayScore","سبب الترشيح"
            ]
        ].sort_values("NextDayScore", ascending=False),
        use_container_width=True
    )

# ---------- TAB 3 ----------
with tab3:
    st.dataframe(df_all, use_container_width=True)

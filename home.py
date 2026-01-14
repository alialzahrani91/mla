import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
import os

# ================= CONFIG =================
st.set_page_config("AI KSA Trading Dashboard", layout="wide")

SYMBOLS_FILE = "tadawul_symbols.csv"

# ================= LOAD SYMBOLS =================
@st.cache_data
def load_symbols():
    if not os.path.exists(SYMBOLS_FILE):
        return pd.DataFrame()
    df = pd.read_csv(SYMBOLS_FILE)
    if "Symbol" not in df.columns:
        return pd.DataFrame()
    return df

symbols_df = load_symbols()
if symbols_df.empty:
    st.error("❌ ملف symbols غير موجود أو العمود Symbol مفقود")
    st.stop()

# ================= FETCH DATA =================
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="9mo", interval="1d", progress=False)
        if df is None or df.empty or "Close" not in df:
            return None

        close = df["Close"]

        if close.ndim != 1 or len(close) < 200:
            return None

        df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA50"] = EMAIndicator(close, 50).ema_indicator()
        df["EMA200"] = EMAIndicator(close, 200).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()

        macd = MACD(close)
        df["MACD"] = macd.macd_diff()

        df["VolumeAvg20"] = df["Volume"].rolling(20).mean()

        df.dropna(inplace=True)
        if len(df) < 5:
            return None

        return df

    except Exception:
        return None

# ================= ANALYSIS =================
def analyze_stock(symbol, company):
    df = fetch_data(symbol)
    if df is None or df.empty or len(df) < 2:
        return None

    last = df.iloc[-1]

    score = 0
    reasons = []

    # Trend
    if last["EMA20"] > last["EMA50"]:
        score += 20
        reasons.append("EMA20 > EMA50")
    if last["EMA50"] > last["EMA200"]:
        score += 20
        reasons.append("EMA50 > EMA200")

    # Momentum
    if 50 <= last["RSI"] <= 68:
        score += 15
        reasons.append("RSI صحي")

    if last["MACD"] > 0:
        score += 15
        reasons.append("MACD إيجابي")

    # Volume
    if last["Volume"] > last["VolumeAvg20"] * 1.3:
        score += 20
        reasons.append("سيولة مفاجئة")

    # Entry / SL / Targets
    entry = round(last["Close"], 2)
    stop = round(entry * 0.96, 2)
    target1 = round(entry * 1.05, 2)
    target2 = round(entry * 1.10, 2)

    return {
        "Symbol": symbol,
        "Company": company,
        "Close": round(last["Close"], 2),
        "EMA20": round(last["EMA20"], 2),
        "EMA50": round(last["EMA50"], 2),
        "EMA200": round(last["EMA200"], 2),
        "RSI": round(last["RSI"], 2),
        "MACD": round(last["MACD"], 4),
        "Score": score,
        "Reasons": " + ".join(reasons),
        "Entry": entry,
        "Stop": stop,
        "Target 1": target1,
        "Target 2": target2
    }

# ================= RUN ANALYSIS =================
results = []

with st.spinner("🔍 تحليل الأسهم..."):
    for _, r in symbols_df.iterrows():
        res = analyze_stock(r["Symbol"], r.get("Company", ""))
        if res:
            results.append(res)

df_all = pd.DataFrame(results)

# ================= UI =================
st.title("📈 AI KSA Trading Dashboard")

tab1, tab2, tab3 = st.tabs([
    "📊 السوق كامل",
    "🔥 أفضل فرص قوية",
    "🎯 فرصة قوية (دخول/وقف/أهداف)"
])

# ---------- TAB 1 ----------
with tab1:
    st.dataframe(df_all, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    top = df_all.sort_values("Score", ascending=False).head(20)
    st.subheader("أفضل 20 سهم محتمل للغد")
    st.dataframe(
        top[[
            "Symbol","Company","Close",
            "EMA20","EMA50","EMA200",
            "RSI","MACD","Score","Reasons"
        ]],
        use_container_width=True
    )

# ---------- TAB 3 ----------
with tab3:
    strong = df_all[df_all["Score"] >= 70].sort_values("Score", ascending=False)
    st.subheader("فرص قوية مكتملة فنياً")
    st.dataframe(
        strong[[
            "Symbol","Company",
            "Entry","Stop","Target 1","Target 2",
            "Score","Reasons"
        ]],
        use_container_width=True
    )

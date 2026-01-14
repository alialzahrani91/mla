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
    st.error("❌ ملف tadawul_symbols.csv غير موجود أو العمود Symbol مفقود")
    st.stop()

# ================= FETCH DATA =================

def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="9mo", interval="1d", progress=False)
        if df is None or df.empty or "Close" not in df:
            return None

        close = df["Close"]

        if close.ndim != 1 or len(close) < 40:
            return None

        df["EMA20"] = EMAIndicator(close, 7).ema_indicator()
        df["EMA50"] = EMAIndicator(close, 21).ema_indicator()
        df["EMA200"] = EMAIndicator(close, 40).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()

        macd = MACD(close)
        df["MACD"] = macd.macd_diff()

        df["VolumeAvg20"] = df["Volume"].rolling(20).mean()
        df["Change %"] = df["Close"].pct_change() * 100

        df.dropna(inplace=True)
        if len(df) < 5:
            return None

        return df

    except Exception:
        return None

# ================= ANALYSIS =================
def analyze_stock(symbol, company):
    df = fetch_data(symbol)
    if df is None or df.empty:
        # صف فارغ مع Symbol و Company فقط
        return {
            "Symbol": symbol,
            "Company": company,
            "Close": np.nan,
            "Change %": np.nan,
            "EMA20": np.nan,
            "EMA50": np.nan,
            "EMA200": np.nan,
            "RSI": np.nan,
            "MACD": np.nan,
            "Volume": np.nan,
            "Score": np.nan,
            "Reasons": "",
            "Entry": np.nan,
            "Stop": np.nan,
            "Target 1": np.nan,
            "Target 2": np.nan,
            "History": []
        }

    last = df.iloc[-1]
    score = 0
    reasons = []

    if last["EMA20"] > last["EMA50"]:
        score += 20; reasons.append("EMA20 > EMA50")
    if last["EMA50"] > last["EMA200"]:
        score += 20; reasons.append("EMA50 > EMA200")
    if 50 <= last["RSI"] <= 68:
        score += 15; reasons.append("RSI صحي")
    if last["MACD"] > 0:
        score += 15; reasons.append("MACD إيجابي")
    if last["Volume"] > last["VolumeAvg20"] * 1.3:
        score += 20; reasons.append("سيولة مفاجئة")

    entry = round(last["Close"], 2)
    stop = round(entry * 0.96, 2)
    t1 = round(entry * 1.05, 2)
    t2 = round(entry * 1.10, 2)

    history = df.tail(10)[["Close","Volume","Change %"]].to_dict(orient="records")

    return {
        "Symbol": symbol,
        "Company": company,
        "Close": round(last["Close"], 2),
        "Change %": round(last["Change %"], 2),
        "EMA20": round(last["EMA20"], 2),
        "EMA50": round(last["EMA50"], 2),
        "EMA200": round(last["EMA200"], 2),
        "RSI": round(last["RSI"], 2),
        "MACD": round(last["MACD"], 4),
        "Volume": int(last["Volume"]),
        "Score": score,
        "Reasons": " + ".join(reasons),
        "Entry": entry,
        "Stop": stop,
        "Target 1": t1,
        "Target 2": t2,
        "History": history
    }

# ================= RUN =================
results = []
st.title("🧠 AI KSA Trading Dashboard")
progress_bar = st.progress(0)
total = len(symbols_df)

for idx, r in symbols_df.iterrows():
    try:
        res = analyze_stock(r["Symbol"], r.get("Company", ""))
        results.append(res)
    except Exception as e:
        # في حال فشل السهم بالكامل
        results.append({
            "Symbol": r["Symbol"],
            "Company": r.get("Company", ""),
            "Close": np.nan,
            "Change %": np.nan,
            "EMA20": np.nan,
            "EMA50": np.nan,
            "EMA200": np.nan,
            "RSI": np.nan,
            "MACD": np.nan,
            "Volume": np.nan,
            "Score": np.nan,
            "Reasons": "",
            "Entry": np.nan,
            "Stop": np.nan,
            "Target 1": np.nan,
            "Target 2": np.nan,
            "History": []
        })
    progress_bar.progress((idx+1)/total)

df_all = pd.DataFrame(results)

# ================= UI =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 السوق كامل",
    "📈 +5% اليوم",
    "🔥 أفضل 20 للغد",
    "🎯 فرص قوية",
    "📉 تحليل 10 إغلاقات",
    "⚡ فرص +2%",
    "🧠 ملخص السوق"
])

with tab1:
    st.dataframe(df_all.drop(columns=["History"], errors="ignore"), use_container_width=True)

with tab2:
    st.dataframe(df_all[df_all["Change %"].notna() & (df_all["Change %"] >= 5)], use_container_width=True)

with tab3:
    top20 = df_all[df_all["Score"].notna()].sort_values("Score", ascending=False).head(20)
    st.dataframe(top20, use_container_width=True)

with tab4:
    strong = df_all[df_all["Score"].notna() & (df_all["Score"] >= 70)].sort_values("Score", ascending=False)
    st.dataframe(strong[["Symbol","Company","Entry","Stop","Target 1","Target 2","Score","Reasons"]], use_container_width=True)

with tab5:
    for _, r in df_all[df_all["Score"].notna() & (df_all["Score"] >= 70)].iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        hist = pd.DataFrame(r.get("History", []))
        if not hist.empty:
            st.dataframe(hist, use_container_width=True)

with tab6:
    st.dataframe(df_all[df_all["Change %"].notna() & (df_all["Change %"] >= 2)], use_container_width=True)

with tab7:
    st.metric("عدد الأسهم", len(df_all))
    if df_all["Score"].notna().any():
        st.metric("أعلى Score", int(df_all["Score"].max()))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
    required = {"Symbol", "Yahoo"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df

symbols_df = load_symbols()
if symbols_df.empty:
    st.error("❌ ملف tadawul_symbols.csv غير صحيح")
    st.stop()

# ================= FETCH YAHOO DATA =================
@st.cache_data(ttl=3600)
def fetch_yahoo_full(yahoo_symbol):
    try:
        df = yf.download(
            yahoo_symbol,
            period="6mo",
            interval="1d",
            progress=False
        )

        if df.empty or len(df) < 50:
            raise Exception

        close = df["Close"]
        volume = df["Volume"]

        change_pct = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
        volume_avg = volume.rolling(20).mean().iloc[-1]

        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))

        volatility = close.pct_change().rolling(20).std() * 100

        return {
            "Close": round(close.iloc[-1], 2),
            "Change %": round(change_pct, 2),
            "Volume": int(volume.iloc[-1]),
            "VolumeAvg20": int(volume_avg),
            "EMA20": round(ema20, 2),
            "EMA50": round(ema50, 2),
            "RSI": round(rsi.iloc[-1], 2),
            "Volatility": round(volatility.iloc[-1], 2),
        }

    except Exception:
        return {
            "Close": np.nan,
            "Change %": np.nan,
            "Volume": np.nan,
            "VolumeAvg20": np.nan,
            "EMA20": np.nan,
            "EMA50": np.nan,
            "RSI": np.nan,
            "Volatility": np.nan,
        }

# ================= BUILD DATA =================
rows = []
with st.spinner("🔄 جلب بيانات السوق..."):
    for _, r in symbols_df.iterrows():
        data = fetch_yahoo_full(r["Yahoo"])
        rows.append({
            "Symbol": r["Symbol"],
            "Company": r.get("Company", ""),
            **data
        })

df_all = pd.DataFrame(rows)

# ================= SCORE ENGINE =================
def score_stock(row):
    score = 0
    reasons = []

    if pd.notna(row["Change %"]) and row["Change %"] >= 2:
        score += 15; reasons.append("+2%")

    if pd.notna(row["Volume"]) and pd.notna(row["VolumeAvg20"]) and row["Volume"] > row["VolumeAvg20"]:
        score += 25; reasons.append("سيولة")

    if pd.notna(row["Close"]) and pd.notna(row["EMA20"]) and row["Close"] > row["EMA20"]:
        score += 15; reasons.append("فوق EMA20")

    if pd.notna(row["EMA20"]) and pd.notna(row["EMA50"]) and row["EMA20"] > row["EMA50"]:
        score += 15; reasons.append("اتجاه صاعد")

    if pd.notna(row["RSI"]) and 45 <= row["RSI"] <= 68:
        score += 15; reasons.append("RSI صحي")

    if pd.notna(row["Volatility"]) and row["Volatility"] < 2.5:
        score += 15; reasons.append("تذبذب منخفض")

    return score, " + ".join(reasons)

scores = df_all.apply(score_stock, axis=1)
df_all["Score"] = scores.apply(lambda x: x[0])
df_all["Reasons"] = scores.apply(lambda x: x[1])

# ================= SIGNALS =================
def trade_signal(row):
    if row["Score"] >= 70 and pd.notna(row["Change %"]) and row["Change %"] >= 2:
        return "BUY 🟢"
    elif row["Score"] >= 55:
        return "WATCH 🟡"
    else:
        return "IGNORE 🔴"

df_all["Signal"] = df_all.apply(trade_signal, axis=1)

# ================= UI =================
st.title("🧠 AI KSA Trading Dashboard")

tabs = st.tabs([
    "📊 السوق كامل",
    "⚡ فرص +2%",
    "🔥 إغلاق +5% اليوم",
    "🟢 فرص دخول",
    "🟡 تحت المراقبة"
])

with tabs[0]:
    st.dataframe(df_all, use_container_width=True)

with tabs[1]:
    st.dataframe(df_all[df_all["Change %"] >= 2], use_container_width=True)

with tabs[2]:
    st.dataframe(df_all[df_all["Change %"] >= 5], use_container_width=True)

with tabs[3]:
    st.dataframe(df_all[df_all["Signal"] == "BUY 🟢"], use_container_width=True)

with tabs[4]:
    st.dataframe(df_all[df_all["Signal"] == "WATCH 🟡"], use_container_width=True)

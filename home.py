import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os

# ================= CONFIG =================
st.set_page_config("AI KSA Trading Dashboard", layout="wide")
TV_FILE = "tradingview_symbols.csv"  # ملف بيانات TradingView

# ================= LOAD TRADINGVIEW DATA =================
@st.cache_data
def load_tradingview():
    if not os.path.exists(TV_FILE):
        return pd.DataFrame()
    # ملف بسيط بدون Header
    df = pd.read_csv(TV_FILE, names=["Symbol", "Company", "TV_Close"])
    return df

tv_df = load_tradingview()
if tv_df.empty:
    st.error("❌ ملف tradingview_symbols.csv غير موجود أو فارغ")
    st.stop()

# ================= FETCH YAHOO DATA =================
@st.cache_data(ttl=3600)
def fetch_yahoo_full(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
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
with st.spinner("🔄 جلب بيانات السوق من Yahoo وTradingView..."):
    for _, r in tv_df.iterrows():
        yahoo_data = fetch_yahoo_full(r["Symbol"])
        rows.append({
            "Symbol": r["Symbol"],
            "Company": r["Company"],      # من TradingView
            "TV_Close": r["TV_Close"],    # من TradingView
            **yahoo_data                  # بيانات Yahoo
        })

df_all = pd.DataFrame(rows)

# ================= SCORE ENGINE DYNAMIC =================
st.sidebar.header("⚙️ إعدادات Scoring")

change_pct_threshold = st.sidebar.number_input("Change % minimum", value=2.0, step=0.1)
volume_multiplier = st.sidebar.number_input("Volume > Avg20 multiplier", value=1.0, step=0.1)
rsi_min = st.sidebar.number_input("RSI Min", value=45)
rsi_max = st.sidebar.number_input("RSI Max", value=68)
volatility_max = st.sidebar.number_input("Max Volatility %", value=2.5)

numeric_cols = ["Change %", "Volume", "VolumeAvg20", "Close", "EMA20", "EMA50", "RSI", "Volatility"]
for col in numeric_cols:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

def score_stock_dynamic(row):
    score = 0
    reasons = []

    change = row.get("Change %", np.nan)
    volume = row.get("Volume", np.nan)
    volume_avg = row.get("VolumeAvg20", np.nan)
    close = row.get("Close", np.nan)
    ema20 = row.get("EMA20", np.nan)
    ema50 = row.get("EMA50", np.nan)
    rsi = row.get("RSI", np.nan)
    volatility = row.get("Volatility", np.nan)

    if pd.notna(change) and change >= change_pct_threshold:
        score += 15
        reasons.append(f"+{change_pct_threshold}%")

    if pd.notna(volume) and pd.notna(volume_avg) and volume > (volume_avg * volume_multiplier):
        score += 25
        reasons.append("سيولة")

    if pd.notna(close) and pd.notna(ema20) and close > ema20:
        score += 15
        reasons.append("فوق EMA20")

    if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
        score += 15
        reasons.append("اتجاه صاعد")

    if pd.notna(rsi) and rsi_min <= rsi <= rsi_max:
        score += 15
        reasons.append("RSI صحي")

    if pd.notna(volatility) and volatility < volatility_max:
        score += 15
        reasons.append("تذبذب منخفض")

    return score, " + ".join(reasons)

scores = df_all.apply(score_stock_dynamic, axis=1)
df_all["Score"] = scores.apply(lambda x: x[0])
df_all["Reasons"] = scores.apply(lambda x: x[1])

# ================= SIGNALS DYNAMIC =================
def trade_signal_dynamic(row):
    if row["Score"] >= 70 and pd.notna(row["Change %"]) and row["Change %"] >= change_pct_threshold:
        return "BUY 🟢"
    elif row["Score"] >= 55:
        return "WATCH 🟡"
    else:
        return "IGNORE 🔴"

df_all["Signal"] = df_all.apply(trade_signal_dynamic, axis=1)

# ================= UI WITH TABS =================
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
    st.dataframe(df_all[df_all["Change %"] >= change_pct_threshold], use_container_width=True)

with tabs[2]:
    st.dataframe(df_all[df_all["Change %"] >= 5], use_container_width=True)

with tabs[3]:
    st.dataframe(df_all[df_all["Signal"] == "BUY 🟢"], use_container_width=True)

with tabs[4]:
    st.dataframe(df_all[df_all["Signal"] == "WATCH 🟡"], use_container_width=True)

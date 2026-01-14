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
    return df if "Symbol" in df.columns else pd.DataFrame()

symbols_df = load_symbols()
if symbols_df.empty:
    st.error("❌ ملف tadawul_symbols.csv غير موجود أو العمود Symbol مفقود")
    st.stop()

# ================= MOCK TradingView DATA =================
# ⚠️ يفترض أنك جلبت هذه القيم من TradingView
def fetch_tv_data(symbol, company):
    try:
        return {
            "Symbol": symbol,
            "Company": company,
            "Close": np.random.uniform(5, 300),
            "Change %": np.random.uniform(-3, 7),
            "Volume": np.random.randint(500_000, 20_000_000),
            "VolumeAvg20": np.random.randint(500_000, 10_000_000),
            "EMA20": np.random.uniform(5, 300),
            "EMA50": np.random.uniform(5, 300)
        }
    except Exception:
        return None

# ================= YAHOO DATA =================
def fetch_yahoo(symbol):
    try:
        df = yf.Ticker(symbol).history(period="6mo", interval="1d")
        if df.empty or len(df) < 20:
            raise Exception

        close = df["Close"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))

        vol = close.pct_change().rolling(20).std() * 100

        return {
            "RSI": round(rsi.iloc[-1], 2),
            "Volatility": round(vol.iloc[-1], 2)
        }
    except Exception:
        return {"RSI": np.nan, "Volatility": np.nan}

# ================= BUILD DATA =================
rows = []
for _, r in symbols_df.iterrows():
    tv = fetch_tv_data(r["Symbol"], r.get("Company", ""))
    if not tv:
        continue

    yahoo = fetch_yahoo(r["Symbol"])

    rows.append({**tv, **yahoo})

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

# ================= PREDICTIONS =================
def predict(row):
    s = 0
    if pd.notna(row["Change %"]) and row["Change %"] >= 2: s += 1
    if pd.notna(row["RSI"]) and 50 <= row["RSI"] <= 65: s += 1
    if pd.notna(row["Volatility"]) and row["Volatility"] < 2.5: s += 1

    return "Bullish 🔼" if s >= 2 else "Neutral ⚪" if s == 1 else "Risky 🔴"

df_all["Prediction"] = df_all.apply(predict, axis=1)

# ================= UI =================
st.title("🧠 AI KSA Trading Dashboard")

tabs = st.tabs([
    "📊 السوق كامل",
    "⚡ فرص +2%",
    "🔥 إغلاق +5% اليوم",
    "🟢 فرص دخول",
    "🟡 تحت المراقبة",
    "🔮 التنبؤات"
])

# TAB 1
with tabs[0]:
    st.dataframe(df_all, use_container_width=True)

# TAB 2 +2%
with tabs[1]:
    st.dataframe(df_all[df_all["Change %"] >= 2], use_container_width=True)

# TAB 3 +5% (جديد ✅)
with tabs[2]:
    st.dataframe(df_all[df_all["Change %"] >= 5], use_container_width=True)

# TAB 4 BUY
with tabs[3]:
    st.dataframe(df_all[df_all["Signal"] == "BUY 🟢"], use_container_width=True)

# TAB 5 WATCH
with tabs[4]:
    st.dataframe(df_all[df_all["Signal"] == "WATCH 🟡"], use_container_width=True)

# TAB 6 Predictions
with tabs[5]:
    st.dataframe(df_all.sort_values("Score", ascending=False), use_container_width=True)

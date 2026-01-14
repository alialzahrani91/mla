import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

# ================= CONFIG =================
st.set_page_config("AI High Gain Dashboard – KSA", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HIGH_GAIN_FILE = f"{DATA_DIR}/high_gain_today.csv"

# ================= HELPERS =================
def safe_save(file, df):
    if df.empty:
        return
    df = df.copy()
    df["Date"] = datetime.today().date()
    if os.path.exists(file):
        old = pd.read_csv(file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol","Date"])
    df.to_csv(file, index=False)
    st.success(f"تم الحفظ في {file}")

# ================= TRADINGVIEW =================
@st.cache_data(ttl=600)
def fetch_ksa():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "columns": [
            "name","description","close","change",
            "relative_volume_10d_calc","volume"
        ],
        "sort": {"sortBy": "change","sortOrder": "desc"},
        "range": [0,400]
    }
    r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
    data = r.json().get("data", [])

    rows = []
    for d in data:
        try:
            rows.append({
                "Symbol": d["s"],
                "Company": d["d"][1],
                "Price": float(d["d"][2]),
                "Change %": float(d["d"][3]),
                "Relative Volume": float(d["d"][4] or 0),
                "Volume": float(d["d"][5] or 0)
            })
        except:
            pass
    return pd.DataFrame(rows)

# ================= INDICATORS =================
def compute_indicators(df):
    df = df.copy()
    df["EMA20"] = EMAIndicator(df["Price"],20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Price"],50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Price"],200).ema_indicator()
    df["RSI"] = RSIIndicator(df["Price"],14).rsi()
    df["MACD"] = MACD(df["Price"]).macd_diff()
    df["Liquidity"] = df["Price"] * df["Volume"]
    df.fillna(method="bfill", inplace=True)
    return df

# ================= LAST 10 DAYS =================
def last_10_days(symbol):
    try:
        s = symbol.replace("TADAWUL:","") + ".SR"
        hist = yf.Ticker(s).history(period="20d")
        if len(hist) < 10:
            return None

        df = hist.tail(10).copy()
        df["Change %"] = df["Close"].pct_change() * 100
        df["Liquidity"] = df["Close"] * df["Volume"]

        if df["Close"].nunique() > 1:
            df["RSI"] = RSIIndicator(df["Close"],14).rsi()
            df["EMA20"] = EMAIndicator(df["Close"],20).ema_indicator()
        else:
            df["RSI"] = np.nan
            df["EMA20"] = np.nan

        df.fillna(method="bfill", inplace=True)
        return df
    except:
        return None

# ================= NEXT DAY SCORE =================
def next_day_score(row, last10=None):
    score = 0
    reasons = []

    # ===== Trend =====
    if row["EMA20"] > row["EMA50"]:
        score += 12; reasons.append("اتجاه صاعد قصير")
    if row["EMA50"] > row["EMA200"]:
        score += 12; reasons.append("اتجاه صاعد متوسط")

    # ===== Momentum =====
    if 50 <= row["RSI"] <= 68:
        score += 12; reasons.append("RSI صحي")
    if row["MACD"] > 0:
        score += 10; reasons.append("MACD إيجابي")

    # ===== Liquidity =====
    if row["Relative Volume"] > 1.3:
        score += 12; reasons.append("سيولة مفاجئة")

    # ===== Price Action =====
    if row["Price"] > row["EMA20"]:
        score += 8; reasons.append("إغلاق أعلى EMA20")

    # ===== 10 Candles Behavior =====
    if last10 is not None:
        green = (last10["Change %"] > 0).sum()
        avg_liq = last10["Liquidity"].mean()
        if green >= 6:
            score += 10; reasons.append("تجميع 10 شموع")
        if last10["Liquidity"].iloc[-1] > avg_liq:
            score += 10; reasons.append("سيولة آخر جلسة أعلى من المتوسط")

    # ===== Risk =====
    if row["RSI"] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score,0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

df = compute_indicators(fetch_ksa())

scores, reasons = [], []
for _, r in df.iterrows():
    l10 = last_10_days(r["Symbol"])
    s, rs = next_day_score(r, l10)
    scores.append(s)
    reasons.append(rs)

df["NextDayScore"] = scores
df["سبب الترشيح"] = reasons

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2%",
    "🧠 Score ذكي"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5]
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_save(HIGH_GAIN_FILE, high_gain)

# ---------- TAB 2 ----------
with tab2:
    top20 = df.sort_values("NextDayScore", ascending=False).head(20)
    st.dataframe(top20[[
        "Symbol","Company","Price",
        "NextDayScore","RSI","MACD",
        "Relative Volume","سبب الترشيح"
    ]], use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        l10 = last_10_days(r["Symbol"])
        if l10 is not None:
            view = l10[[
                "Close","Change %","Volume",
                "Liquidity","RSI","EMA20"
            ]]
            st.dataframe(view, use_container_width=True)
            st.info(f"📌 تأثيره على الغد: {r['سبب الترشيح']}")

# ---------- TAB 4 ----------
with tab4:
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.dataframe(df[df["Change %"] >= 2], use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    st.metric("عدد الأسهم", len(df))
    st.metric("أعلى NextDayScore", df["NextDayScore"].max())

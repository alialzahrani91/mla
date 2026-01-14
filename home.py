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
PRED_FILE = f"{DATA_DIR}/predictions.csv"
TRAIN_FILE = f"{DATA_DIR}/training.csv"

# ================= HELPERS =================
def safe_save(file, df):
    if df.empty:
        return
    df["Date"] = datetime.today().date()
    if os.path.exists(file) and os.stat(file).st_size > 0:
        old = pd.read_csv(file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol", "Date"])
    df.to_csv(file, index=False)
    st.success(f"تم الحفظ في {file}")

def fetch_ksa():
    """جلب جميع الأسهم السعودية من TradingView"""
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "columns": ["name", "description", "close", "change", "relative_volume_10d_calc", "volume"],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 400]
    }
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
        data = r.json().get("data", [])
    except Exception as e:
        st.error(f"⚠️ خطأ في جلب البيانات: {e}")
        return pd.DataFrame()
    
    rows = []
    for d in data:
        try:
            rows.append({
                "Symbol": d["s"],
                "Company": d["d"][1],
                "Price": float(d["d"][2]) if len(d["d"])>2 else 0.0,
                "Change %": float(d["d"][3]) if len(d["d"])>3 else 0.0,
                "Relative Volume": float(d["d"][4]) if len(d["d"])>4 else 0.0,
                "Volume": float(d["d"][5]) if len(d["d"])>5 else 0.0
            })
        except:
            continue
    return pd.DataFrame(rows)

# ================= FETCH HISTORICAL =================
@st.cache_data(ttl=600)
def fetch_history(symbol, period="200d"):
    """جلب البيانات التاريخية من Yahoo Finance"""
    try:
        s = symbol.replace("TADAWUL:", "") + ".SR"
        hist = yf.Ticker(s).history(period=period)
        hist = hist.reset_index()
        return hist
    except:
        return pd.DataFrame()

# ================= INDICATORS =================
def compute_indicators_hist(hist):
    hist = hist.copy()
    hist["EMA20"] = hist["Close"].ewm(span=20, adjust=False).mean()
    hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
    hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
    hist["RSI"] = RSIIndicator(hist["Close"], window=14).rsi()
    macd = MACD(hist["Close"])
    hist["MACD"] = macd.macd_diff()
    hist["VolumeMA"] = hist["Volume"].rolling(10).mean()
    hist.fillna(method="bfill", inplace=True)
    return hist

# ================= NEXT DAY SCORE =================
def next_day_score(row, hist):
    score = 0
    reasons = []

    if row["Price"] > hist["EMA20"].iloc[-1]:
        score += 15; reasons.append("إغلاق أعلى EMA20")
    if hist["EMA20"].iloc[-1] > hist["EMA50"].iloc[-1]:
        score += 10; reasons.append("EMA20 > EMA50")
    if hist["EMA50"].iloc[-1] > hist["EMA200"].iloc[-1]:
        score += 10; reasons.append("EMA50 > EMA200")
    if 50 <= hist["RSI"].iloc[-1] <= 68:
        score += 15; reasons.append("RSI صحي")
    if hist["MACD"].iloc[-1] > 0:
        score += 10; reasons.append("MACD إيجابي")
    if row["Relative Volume"] > 1.3:
        score += 15; reasons.append("سيولة مرتفعة")
    if (hist["Close"].pct_change()[-10:] > 0).sum() >= 6:
        score += 10; reasons.append("تجميع 10 شموع")
    if hist["Volume"].iloc[-1] > hist["VolumeMA"].iloc[-1]:
        score += 10; reasons.append("سيولة آخر جلسة أعلى المتوسط")
    if hist["RSI"].iloc[-1] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score, 0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

df = fetch_ksa()
if df.empty:
    st.stop()

# حساب المؤشرات لكل سهم باستخدام البيانات التاريخية
ema20s, ema50s, ema200s, rsis, macds, scores, reasons = [], [], [], [], [], [], []
for _, row in df.iterrows():
    hist = fetch_history(row["Symbol"])
    if hist.empty:
        ema20s.append(np.nan); ema50s.append(np.nan); ema200s.append(np.nan); rsis.append(np.nan); macds.append(np.nan)
        scores.append(0); reasons.append("")
        continue
    hist = compute_indicators_hist(hist)
    ema20s.append(hist["EMA20"].iloc[-1])
    ema50s.append(hist["EMA50"].iloc[-1])
    ema200s.append(hist["EMA200"].iloc[-1])
    rsis.append(hist["RSI"].iloc[-1])
    macds.append(hist["MACD"].iloc[-1])
    score, reason = next_day_score(row, hist)
    scores.append(score)
    reasons.append(reason)

df["EMA20"] = ema20s
df["EMA50"] = ema50s
df["EMA200"] = ema200s
df["RSI"] = rsis
df["MACD"] = macds
df["NextDayScore"] = scores
df["سبب الترشيح"] = reasons

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 سهم للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2% غدًا",
    "🧠 Score ذكي",
    "🚀 فرص قوية (MACD+سيولة+EMA20+10 شموع)"
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
    st.subheader("أفضل 20 سهم مرشح للغد (+5%)")
    st.dataframe(top20[[
        "Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"
    ]], use_container_width=True)
    if st.button("💾 حفظ التنبؤات"):
        safe_save(PRED_FILE, top20)

# ---------- TAB 3 ----------
with tab3:
    st.subheader("تحليل آخر 10 إغلاقات لكل سهم + حجم السيولة والمؤشرات")
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        hist = fetch_history(r["Symbol"], period="15d")
        if not hist.empty:
            hist = compute_indicators_hist(hist)
            view = hist.tail(10)[["Close","Volume","EMA20","EMA50","EMA200","RSI","MACD"]].copy()
            st.dataframe(view, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.dataframe(df[df["Change %"] >= 2], use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    st.metric("عدد الأسهم المحللة", len(df))
    st.metric("أفضل Score", df["NextDayScore"].max())

# ---------- TAB 7 ----------
with tab7:
    strong_ops = df[
        (df["MACD"] > 0) &
        (df["Relative Volume"] > 1.3) &
        (df["Price"] > df["EMA20"])
    ].sort_values("NextDayScore", ascending=False).head(20)
    st.subheader("فرص قوية للغد بناء على التحليل الفني")
    st.dataframe(strong_ops[[
        "Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"
    ]], use_container_width=True)

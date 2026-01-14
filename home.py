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
    df["Date"] = datetime.today().date()
    if os.path.exists(file):
        try:
            old = pd.read_csv(file)
            df = pd.concat([old, df]).drop_duplicates(subset=["Symbol", "Date"])
        except pd.errors.EmptyDataError:
            pass
    df.to_csv(file, index=False)
    st.success(f"تم الحفظ في {file}")

# ================= TRADINGVIEW =================
@st.cache_data(ttl=600)
def fetch_ksa():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "columns": [
            "name", "description", "close", "change",
            "relative_volume_10d_calc", "volume"
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 400]
    }
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
        data = r.json().get("data", [])
    except:
        return pd.DataFrame()

    rows = []
    for d in data:
        try:
            rows.append({
                "Symbol": d["s"],
                "Company": d["d"][1] if len(d["d"])>1 else "",
                "Price": float(d["d"][2]) if len(d["d"])>2 and d["d"][2] else np.nan,
                "Change %": float(d["d"][3]) if len(d["d"])>3 and d["d"][3] else np.nan,
                "Relative Volume": float(d["d"][4]) if len(d["d"])>4 and d["d"][4] else np.nan,
                "Volume": float(d["d"][5]) if len(d["d"])>5 and d["d"][5] else np.nan
            })
        except:
            continue
    return pd.DataFrame(rows)

# ================= INDICATORS =================
def compute_indicators(df):
    df = df.copy()
    df["EMA20"] = EMAIndicator(df["Price"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Price"], 50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Price"], 200).ema_indicator()
    df["RSI"] = RSIIndicator(df["Price"], 14).rsi()
    macd = MACD(df["Price"])
    df["MACD"] = macd.macd_diff()
    df.fillna(method="bfill", inplace=True)
    return df

# ================= LAST 10 DAYS =================
def last_10_days(symbol):
    try:
        s = symbol.replace("TADAWUL:", "") + ".SR"
        hist = yf.Ticker(s).history(period="15d")
        if len(hist) < 10:
            return None
        last10 = hist.tail(10)
        last10["Change %"] = last10["Close"].pct_change() * 100
        return last10
    except:
        return None

# ================= NEXT DAY SCORE =================
def next_day_score(row, last10=None):
    score = 0
    reasons = []

    # Trend
    if row.get("EMA20",0) > row.get("EMA50",0):
        score += 15; reasons.append("EMA20 > EMA50")
    if row.get("EMA50",0) > row.get("EMA200",0):
        score += 15; reasons.append("EMA50 > EMA200")

    # Momentum
    if 50 <= row.get("RSI",0) <= 68:
        score += 15; reasons.append("RSI صحي")
    if row.get("MACD",0) > 0:
        score += 10; reasons.append("MACD إيجابي")

    # Volume
    if row.get("Relative Volume",0) > 1.3:
        score += 15; reasons.append("سيولة مرتفعة")

    # Price Action
    if row.get("Price",0) > row.get("EMA20",0):
        score += 10; reasons.append("إغلاق فوق EMA20")

    if last10 is not None:
        green = (last10["Change %"] > 0).sum()
        if green >= 6:
            score += 10; reasons.append("تجميع 10 أيام")

    # Risk
    if row.get("RSI",0) > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score, 0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")
df = fetch_ksa()
df = compute_indicators(df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 سهم للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2% غدًا",
    "🧠 Score ذكي",
    "💹 فرص MACD + سيولة"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5]
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_save(HIGH_GAIN_FILE, high_gain)

# ---------- TAB 2 ----------
with tab2:
    scores, reasons = [], []
    for _, r in df.iterrows():
        l10 = last_10_days(r["Symbol"])
        s, reason = next_day_score(r, l10)
        scores.append(s)
        reasons.append(reason)
    df["NextDayScore"] = scores
    df["سبب الترشيح"] = reasons
    top20 = df.sort_values("NextDayScore", ascending=False).head(20)
    cols = ["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"]
    top20 = top20.reindex(columns=cols, fill_value=np.nan)
    st.dataframe(top20, use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        l10 = last_10_days(r["Symbol"])
        if l10 is not None:
            view = l10[["Close"]].copy()
            view["Change %"] = l10["Close"].pct_change() * 100
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
    macd_opportunities = df[
        (df["MACD"]>0) &
        (df["Relative Volume"]>1.3) &
        (df["Price"]>df["EMA20"])
    ].copy()
    top_macd = []
    for _, r in macd_opportunities.iterrows():
        l10 = last_10_days(r["Symbol"])
        if l10 is not None and (l10["Change %"]>0).sum()>=6 and l10["Volume"].iloc[-1] > l10["Volume"].mean():
            top_macd.append(r)
    top_macd = pd.DataFrame(top_macd).head(20)
    cols7 = ["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"]
    top_macd = top_macd.reindex(columns=cols7, fill_value=np.nan)
    st.subheader("فرص MACD إيجابي + سيولة مرتفعة + تجميع 10 شموع")
    st.dataframe(top_macd, use_container_width=True)

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
        old = pd.read_csv(file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol", "Date"])
    df.to_csv(file, index=False)
    st.success(f"تم الحفظ في {file}")

# ================= TRADINGVIEW =================
@st.cache_data(ttl=600)
def fetch_ksa():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"}
        ],
        "columns": [
            "name", "description", "close", "change",
            "relative_volume_10d_calc", "volume"
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 400]
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
                "Relative Volume": float(d["d"][4]) if d["d"][4] else 0,
                "Volume": float(d["d"][5]) if d["d"][5] else 0
            })
        except:
            pass

    return pd.DataFrame(rows)

# ================= HISTORICAL DATA =================
def fetch_history(symbol, period="200d"):
    try:
        s = symbol.replace("TADAWUL:", "") + ".SR"
        hist = yf.Ticker(s).history(period=period)
        if hist.empty:
            return None
        return hist
    except:
        return None

# ================= INDICATORS =================
def compute_indicators(hist):
    df = hist.copy()
    df["EMA20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd_diff()
    df["Relative Volume"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df.fillna(method="bfill", inplace=True)
    return df

# ================= NEXT DAY SCORE =================
def next_day_score(row, hist_last10=None):
    score = 0
    reasons = []

    # Trend
    if row["EMA20"] > row["EMA50"]:
        score += 15; reasons.append("EMA20 > EMA50")
    if row["EMA50"] > row["EMA200"]:
        score += 15; reasons.append("EMA50 > EMA200")

    # Momentum
    if 50 <= row["RSI"] <= 68:
        score += 15; reasons.append("RSI صحي")
    if row["MACD"] > 0:
        score += 10; reasons.append("MACD إيجابي")

    # Volume
    if row["Relative Volume"] > 1.3:
        score += 15; reasons.append("سيولة مرتفعة")

    # Price Action
    if row["Close"] > row["EMA20"]:
        score += 10; reasons.append("إغلاق فوق EMA20")

    # Last 10 days
    if hist_last10 is not None:
        green = (hist_last10["Close"].pct_change()*100 > 0).sum()
        if green >= 6:
            score += 10; reasons.append("تجميع 10 أيام")

    # Risk
    if row["RSI"] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score, 0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

# جلب بيانات السوق
df = fetch_ksa()
df["NextDayScore"] = 0
df["سبب الترشيح"] = ""
df["نوع التداول"] = ""

# حساب المؤشرات لكل سهم
for idx, row in df.iterrows():
    hist = fetch_history(row["Symbol"])
    if hist is not None and not hist.empty:
        ind = compute_indicators(hist)
        last_row = ind.iloc[-1]
        score, reason = next_day_score(last_row, ind.tail(10))
        df.at[idx,"NextDayScore"] = score
        df.at[idx,"سبب الترشيح"] = reason
        df.at[idx,"نوع التداول"] = "مضاربي" if score>=40 else "استثماري"

# التابات
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 سهم للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2% غدًا",
    "🧠 Score ذكي",
    "💹 فرص متقدمة"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5].copy()
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_save(HIGH_GAIN_FILE, high_gain)

# ---------- TAB 2 ----------
with tab2:
    top20 = df.sort_values("NextDayScore", ascending=False).head(20)
    st.subheader("أفضل 20 سهم للغد (+5%)")
    st.dataframe(
        top20[["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح","نوع التداول"]],
        use_container_width=True
    )

# ---------- TAB 3 ----------
with tab3:
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        hist = fetch_history(r["Symbol"])
        if hist is not None:
            view = hist.tail(10)[["Close","Volume"]].copy()
            view["Change %"] = view["Close"].pct_change()*100
            st.dataframe(view, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    potential = df[df["Change %"] >= 2]
    st.dataframe(potential, use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    st.metric("عدد الأسهم المحللة", len(df))
    st.metric("أفضل Score", df["NextDayScore"].max())

# ---------- TAB 7 ----------
with tab7:
    adv = df[
        (df["MACD"] > 0) &
        (df["Relative Volume"] > 1.3) &
        (df["Close"] > df["EMA20"]) 
    ].copy()
    adv_scores = []
    adv_reasons = []
    for idx, row in adv.iterrows():
        hist = fetch_history(row["Symbol"])
        score, reason = next_day_score(row, hist.tail(10) if hist is not None else None)
        adv_scores.append(score)
        adv_reasons.append(reason)
    adv["NextDayScore"] = adv_scores
    adv["سبب الترشيح"] = adv_reasons
    st.subheader("فرص متقدمة – MACD إيجابي + سيولة + إغلاق أعلى EMA20 + تجميع 10 شموع")
    st.dataframe(adv[["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","Volume","سبب الترشيح","نوع التداول"]], use_container_width=True)

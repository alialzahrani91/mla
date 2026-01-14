import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
import os

# ================= CONFIG =================
st.set_page_config("AI High Gain Dashboard – KSA", layout="wide")
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

def last_10_days(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="15d")
        if len(hist) < 10:
            return None
        last10 = hist.tail(10)
        last10["Change %"] = last10["Close"].pct_change() * 100
        return last10
    except:
        return None

def next_day_score(row, last10=None):
    score = 0
    reasons = []

    if row["EMA20"] > row["EMA50"]:
        score += 15; reasons.append("EMA20 > EMA50")
    if row["EMA50"] > row["EMA200"]:
        score += 15; reasons.append("EMA50 > EMA200")
    if 50 <= row["RSI"] <= 68:
        score += 15; reasons.append("RSI صحي")
    if row["MACD"] > 0:
        score += 10; reasons.append("MACD إيجابي")
    if row["Price"] > row["EMA20"]:
        score += 10; reasons.append("إغلاق فوق EMA20")
    if row["Relative Volume"] > 1.3:
        score += 15; reasons.append("سيولة مرتفعة")

    if last10 is not None:
        green = (last10["Change %"] > 0).sum()
        if green >= 6:
            score += 10; reasons.append("تجميع 10 أيام")

    if row["RSI"] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score,0), " + ".join(reasons)

# ================= LOAD SYMBOLS =================
# CSV يحتوي عمود Symbol بصيغة Yahoo (مثال: 4327.SR)
symbols_file = "tadawul_symbols.csv"
symbols_df = pd.read_csv(symbols_file)
symbols = symbols_df["Symbol"].tolist()

# ================= FETCH DATA =================
rows = []
for s in symbols:
    try:
        df_hist = yf.Ticker(s).history(period="5d")
        price = df_hist["Close"].iloc[-1]
        volume = df_hist["Volume"].iloc[-1]
        rel_vol = volume / df_hist["Volume"].rolling(10).mean().iloc[-1]
        rows.append({
            "Symbol": s,
            "Company": s,  # يمكن تعديل لو عندك أسماء الشركات
            "Price": price,
            "Relative Volume": rel_vol,
            "Volume": volume,
            "Change %": (price - df_hist["Close"].iloc[-2]) / df_hist["Close"].iloc[-2] * 100
        })
    except:
        continue

df = pd.DataFrame(rows)
df = compute_indicators(df)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 سهم للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2% غدًا",
    "🧠 Score ذكي",
    "💎 فرص قوية (دخول/وقف/هدف)"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5]
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_save(HIGH_GAIN_FILE, high_gain)

# ---------- TAB 2 ----------
with tab2:
    scores = []
    reasons = []
    for _, r in df.iterrows():
        l10 = last_10_days(r["Symbol"])
        s, reason = next_day_score(r, l10)
        scores.append(s)
        reasons.append(reason)
    df["NextDayScore"] = scores
    df["سبب الترشيح"] = reasons
    top20 = df.sort_values("NextDayScore", ascending=False).head(20)
    st.dataframe(top20[["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"]], use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']}")
        l10 = last_10_days(r["Symbol"])
        if l10 is not None:
            view = l10[["Close"]].copy()
            view["Change %"] = l10["Close"].pct_change() * 100
            view["Volume"] = l10["Volume"]
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
    top_scores = df[df["NextDayScore"] >= 50].head(20).copy()
    top_scores["دخول"] = top_scores["Price"]  # سعر الدخول الحالي
    top_scores["وقف"] = top_scores["EMA20"] - (top_scores["Price"]*0.02)  # مثال وقف 2%
    top_scores["هدف"] = top_scores["Price"] + (top_scores["Price"]*0.05)  # هدف 5%
    st.dataframe(top_scores[["Symbol","Company","Price","NextDayScore","دخول","وقف","هدف","سبب الترشيح"]], use_container_width=True)

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
HIGH_GAIN_FILE = os.path.join(DATA_DIR, "high_gain_today.csv")

# ================= HELPERS =================
def safe_save(file, df):
    if df.empty:
        return
    df["Date"] = datetime.today().date()
    if os.path.exists(file):
        try:
            old = pd.read_csv(file)
            df = pd.concat([old, df]).drop_duplicates(subset=["Symbol", "Date"])
        except:
            pass
    df.to_csv(file, index=False)
    st.success(f"تم الحفظ في {file}")

# ================= TRADINGVIEW =================
@st.cache_data(ttl=600)
def fetch_ksa():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [{"left": "type","operation": "equal","right": "stock"}],
        "columns": ["name","description","close","change","relative_volume_10d_calc","volume"],
        "sort": {"sortBy": "change","sortOrder": "desc"},
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
                "Company": d["d"][1],
                "Price": float(d["d"][2]),
                "Change %": float(d["d"][3]),
                "Relative Volume": float(d["d"][4]) if d["d"][4] else 0,
                "Volume": float(d["d"][5]) if d["d"][5] else 0
            })
        except:
            continue
    return pd.DataFrame(rows)

# ================= HISTORICAL DATA =================
def get_historical(symbol, days=250):
    s = symbol.replace("TADAWUL:","")+".SR"
    try:
        hist = yf.Ticker(s).history(period=f"{days}d")
        if hist.empty: return None
        hist['Change %'] = hist['Close'].pct_change()*100
        hist['EMA20'] = EMAIndicator(hist['Close'],20).ema_indicator()
        hist['EMA50'] = EMAIndicator(hist['Close'],50).ema_indicator()
        hist['EMA200'] = EMAIndicator(hist['Close'],200).ema_indicator()
        hist['RSI'] = RSIIndicator(hist['Close'],14).rsi()
        macd = MACD(hist['Close'])
        hist['MACD'] = macd.macd_diff()
        hist['Relative Volume'] = hist['Volume']/hist['Volume'].rolling(10).mean()
        hist.fillna(method="bfill", inplace=True)
        return hist
    except:
        return None

# ================= NEXT DAY SCORE =================
def next_day_score(row, last10=None):
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
    if row["Price"] > row["EMA20"]:
        score += 10; reasons.append("إغلاق فوق EMA20")

    if last10 is not None:
        green = (last10["Change %"] > 0).sum()
        if green >= 6:
            score += 10; reasons.append("تجميع 10 أيام")

    # Risk
    if row["RSI"] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score,0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

df = fetch_ksa()
if df.empty:
    st.stop()

# جلب المؤشرات التاريخية لكل سهم
indicators = []
reasons_all = []
scores_all = []
for _, r in df.iterrows():
    hist = get_historical(r["Symbol"])
    if hist is not None:
        today = hist.iloc[-1]
        l10 = hist.tail(10)
        s, reason = next_day_score(today, l10)
        indicators.append(today)
        scores_all.append(s)
        reasons_all.append(reason)
    else:
        scores_all.append(0)
        reasons_all.append("بيانات غير كافية")
        indicators.append(r)

df["NextDayScore"] = scores_all
df["سبب الترشيح"] = reasons_all
df["EMA20"] = [i["EMA20"] if "EMA20" in i else 0 for i in indicators]
df["EMA50"] = [i["EMA50"] if "EMA50" in i else 0 for i in indicators]
df["EMA200"] = [i["EMA200"] if "EMA200" in i else 0 for i in indicators]
df["RSI"] = [i["RSI"] if "RSI" in i else 0 for i in indicators]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 سهم للغد",
    "📉 تحليل 10 إغلاقات",
    "📊 السوق كامل",
    "⚡ فرص +2% غدًا",
    "🧠 Score ذكي"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >=5]
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_save(HIGH_GAIN_FILE, high_gain)

# ---------- TAB 2 ----------
with tab2:
    top20 = df.sort_values("NextDayScore", ascending=False).head(20)
    st.dataframe(top20[["Symbol","Company","Price","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"]], use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    for _, r in high_gain.iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        hist = get_historical(r["Symbol"])
        if hist is not None:
            view = hist.tail(10)[["Close","Change %","Volume"]]
            st.dataframe(view, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.dataframe(df[df["Change %"] >=2], use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    st.metric("عدد الأسهم المحللة", len(df))
    st.metric("أفضل Score", df["NextDayScore"].max())

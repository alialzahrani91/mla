import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

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
        "range": [0, 500]
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

    df["EMA20"] = EMAIndicator(df["Price"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Price"], 50).ema_indicator()
    df["EMA200"] = EMAIndicator(df["Price"], 200).ema_indicator()

    df["RSI"] = RSIIndicator(df["Price"], 14).rsi()
    df["MACD"] = MACD(df["Price"]).macd_diff()

    df.fillna(method="bfill", inplace=True)
    return df

# ================= LAST 10 DAYS (تقريبي) =================
def last_10_mock(row):
    """محاكاة سلوك 10 جلسات بناءً على الزخم"""
    green = 0
    if row["RSI"] > 55: green += 3
    if row["MACD"] > 0: green += 3
    if row["Relative Volume"] > 1.2: green += 2
    if row["Price"] > row["EMA20"]: green += 2
    return min(green, 10)

# ================= NEXT DAY SCORE =================
def next_day_score(row):
    score = 0
    reasons = []

    if row["EMA20"] > row["EMA50"]:
        score += 15; reasons.append("EMA20 > EMA50")
    if row["EMA50"] > row["EMA200"]:
        score += 15; reasons.append("ترند صاعد")

    if 50 <= row["RSI"] <= 68:
        score += 15; reasons.append("RSI صحي")
    if row["MACD"] > 0:
        score += 10; reasons.append("MACD إيجابي")

    if row["Relative Volume"] > 1.3:
        score += 15; reasons.append("سيولة مرتفعة")

    if row["Price"] > row["EMA20"]:
        score += 10; reasons.append("إغلاق أعلى EMA20")

    green = last_10_mock(row)
    if green >= 6:
        score += 10; reasons.append("تجميع واضح")

    if row["RSI"] > 72:
        score -= 20; reasons.append("تشبع شراء")

    return max(score, 0), " + ".join(reasons)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – KSA")

df = fetch_ksa()
if df.empty:
    st.stop()

df = compute_indicators(df)

# حساب Score
scores, reasons = [], []
for _, r in df.iterrows():
    s, reason = next_day_score(r)
    scores.append(s)
    reasons.append(reason)

df["NextDayScore"] = scores
df["سبب الترشيح"] = reasons

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔥 أفضل 20 للغد",
    "📉 تحليل 10 إشارات",
    "📊 السوق كامل",
    "⚡ فرص +2%",
    "🧠 Score ذكي",
    "🚀 فرص فنية مكتملة"
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
    st.dataframe(
        top20[[
            "Symbol","Company","Price","NextDayScore",
            "RSI","MACD","Relative Volume","سبب الترشيح"
        ]],
        use_container_width=True
    )

# ---------- TAB 3 ----------
with tab3:
    for _, r in df[df["Change %"] >= 5].iterrows():
        st.markdown(f"### {r['Symbol']} – {r['Company']}")
        st.write({
            "RSI": round(r["RSI"],2),
            "MACD": round(r["MACD"],4),
            "Relative Volume": r["Relative Volume"],
            "EMA20": round(r["EMA20"],2),
            "EMA50": round(r["EMA50"],2),
            "NextDayScore": r["NextDayScore"]
        })

# ---------- TAB 4 ----------
with tab4:
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.dataframe(df[df["Change %"] >= 2], use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    st.metric("عدد الأسهم", len(df))
    st.metric("أعلى Score", df["NextDayScore"].max())

# ---------- TAB 7 ----------
with tab7:
    qualified = []

    for _, r in df.iterrows():
        green = last_10_mock(r)

        if (
            r["MACD"] > 0 and
            r["Relative Volume"] > 1.3 and
            r["Price"] > r["EMA20"] and
            green >= 6
        ):
            trade_type = "مضاربي"
            if (
                r["EMA20"] > r["EMA50"] > r["EMA200"] and
                45 <= r["RSI"] <= 60 and
                r["Relative Volume"] <= 1.8
            ):
                trade_type = "استثماري"

            qualified.append({
                "Symbol": r["Symbol"],
                "Company": r["Company"],
                "Price": r["Price"],
                "NextDayScore": r["NextDayScore"],
                "نوع التداول": trade_type,
                "سبب الترشيح":
                    "MACD إيجابي + سيولة مفاجئة + "
                    "إغلاق أعلى EMA20 + تجميع"
            })

    qdf = pd.DataFrame(qualified)

    if qdf.empty:
        st.warning("لا توجد فرص فنية مكتملة حالياً")
    else:
        st.dataframe(
            qdf.sort_values("NextDayScore", ascending=False),
            use_container_width=True
        )

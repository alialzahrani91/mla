import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import date
import xgboost as xgb

# ================= CONFIG =================
st.set_page_config("تركي وحمد", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HIGH_GAIN_FILE = f"{DATA_DIR}/high_gain_today.csv"
PRED_FILE = f"{DATA_DIR}/predictions.csv"
NEXT_DAY_FILE = f"{DATA_DIR}/next_day_candidates.csv"

# ================= UTILS =================
def save_csv(file, df):
    if df.empty:
        return
    df = df.copy()
    df["Date"] = date.today()
    if os.path.exists(file):
        old = pd.read_csv(file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol","Date"])
    df.to_csv(file, index=False)

def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0

# ================= TRADINGVIEW =================
def scan_tv(columns, sort_col):
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [
            {"left":"exchange","operation":"equal","right":"TADAWUL"},
            {"left":"type","operation":"equal","right":"stock"}
        ],
        "columns": columns,
        "sort":{"sortBy":sort_col,"sortOrder":"desc"},
        "range":[0,500]
    }
    r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
    data = r.json().get("data",[])
    rows=[]
    for d in data:
        rows.append({
            "Symbol": d["s"],
            **{columns[i]: safe_float(d["d"][i]) for i in range(len(columns))}
        })
    return pd.DataFrame(rows)

# ================= DATA =================
def fetch_daily():
    return scan_tv(
        ["close","change","volume","RSI","MACD.macd"],
        "change"
    )

def fetch_15m():
    return scan_tv(
        ["close","change|15","RSI|15","MACD.macd|15"],
        "change|15"
    )

def fetch_1h():
    return scan_tv(
        ["close","change|60","RSI|60","MACD.macd|60"],
        "change|60"
    )

# ================= ML =================
def xgboost_predict(df):
    df = df.copy()
    df["Target"] = (df["change"] >= 5).astype(int)
    features = ["change","volume","RSI","MACD.macd"]
    X = df[features]
    y = df["Target"]

    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X, y)
    df["Prediction"] = model.predict_proba(X)[:,1]
    return df.sort_values("Prediction", ascending=False)

# ================= NEXT DAY BREAKOUT =================
def next_day_breakout_candidates(df):
    df = df.copy()

    # استبعاد الأسهم التي انفجرت اليوم
    df = df[(df["change"] < 3) & (df["change"] > 0.3)]

    score = pd.Series(0, index=df.index)

    # Volume Accumulation
    score += (df["volume"] > df["volume"].median()).astype(int) * 2

    # RSI في منطقة الانطلاق
    score += ((df["RSI"] > 45) & (df["RSI"] < 60)).astype(int) * 2

    # MACD قريب من التقاطع
    score += (df["MACD.macd"].between(-0.2, 0.15)).astype(int) * 2

    # حركة سعرية إيجابية خفيفة
    score += ((df["change"] > 0.5) & (df["change"] < 2)).astype(int)

    df["Breakout Score"] = score

    return df[df["Breakout Score"] >= 5] \
        .sort_values("Breakout Score", ascending=False) \
        .head(10)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – Saudi Market")

daily = fetch_daily()
m15 = fetch_15m()
h1 = fetch_1h()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 +5% اليوم",
    "🔮 XGBoost تنبؤ",
    "⏱️ Multi-Timeframe",
    "⚡ فرص +2%",
    "🔮 مرشحي اختراق الغد +5%",
    "📊 السوق كامل",
    "💾 الملفات"
])

# -------- TAB 1 --------
with tab1:
    hg = daily[daily["change"] >= 5]
    st.dataframe(hg, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        save_csv(HIGH_GAIN_FILE, hg)
        st.success("تم الحفظ")

# -------- TAB 2 --------
with tab2:
    pred = xgboost_predict(daily)
    top = pred[pred["Prediction"] > 0.7].head(10)
    st.dataframe(top, use_container_width=True)
    if st.button("💾 حفظ تنبؤات XGBoost"):
        save_csv(PRED_FILE, top)
        st.success("تم الحفظ")

# -------- TAB 3 --------
with tab3:
    df = daily.merge(m15, on="Symbol").merge(h1, on="Symbol")
    df["Score"] = (
        (df["change|15"] > 0.5).astype(int) +
        (df["change|60"] > 1).astype(int) +
        (df["RSI|15"] > 55).astype(int) +
        (df["RSI|60"] > 55).astype(int)
    )
    st.dataframe(df.sort_values("Score", ascending=False).head(10), use_container_width=True)

# -------- TAB 4 --------
with tab4:
    st.dataframe(daily[daily["change"] >= 2], use_container_width=True)

# -------- TAB 5 (الجديد) --------
with tab5:
    st.subheader("🔮 أفضل 10 أسهم مرشحة لاختراق +5% في التداول القادم")
    next_day = next_day_breakout_candidates(daily)
    st.dataframe(
        next_day[
            ["Symbol","change","RSI","MACD.macd","volume","Breakout Score"]
        ],
        use_container_width=True
    )
    if st.button("💾 حفظ مرشحي الغد"):
        save_csv(NEXT_DAY_FILE, next_day)
        st.success("تم حفظ قائمة مرشحي الغد")

# -------- TAB 6 --------
with tab6:
    st.dataframe(daily, use_container_width=True)

# -------- TAB 7 --------
with tab7:
    st.write("📁 +5% اليوم")
    if os.path.exists(HIGH_GAIN_FILE):
        st.dataframe(pd.read_csv(HIGH_GAIN_FILE))
    st.write("📁 تنبؤات XGBoost")
    if os.path.exists(PRED_FILE):
        st.dataframe(pd.read_csv(PRED_FILE))
    st.write("📁 مرشحي اختراق الغد")
    if os.path.exists(NEXT_DAY_FILE):
        st.dataframe(pd.read_csv(NEXT_DAY_FILE))

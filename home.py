import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import date
import xgboost as xgb

# ================= CONFIG =================
st.set_page_config("AI High Gain – KSA", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HIGH_GAIN_FILE = f"{DATA_DIR}/high_gain_today.csv"
PRED_FILE = f"{DATA_DIR}/predictions.csv"

# ================= UTILS =================
def save_csv(file, df):
    if df.empty:
        return
    df["Date"] = date.today()
    if os.path.exists(file):
        old = pd.read_csv(file)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol","Date"])
    df.to_csv(file, index=False)

def safe_float(v):
    try: return float(v)
    except: return 0.0

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

# ================= MULTI TF =================
def multi_tf(d, m15, h1):
    df = d.merge(m15, on="Symbol").merge(h1, on="Symbol")
    df["Score"] = 0

    df.loc[df["change|15"] > 0.5, "Score"] += 1
    df.loc[df["change|60"] > 1.0, "Score"] += 1
    df.loc[df["change"] > 1.5, "Score"] += 1

    df.loc[df["RSI|15"] > 55, "Score"] += 1
    df.loc[df["RSI|60"] > 55, "Score"] += 1
    df.loc[df["RSI"] > 50, "Score"] += 1

    df.loc[df["MACD.macd|15"] > 0, "Score"] += 1
    df.loc[df["MACD.macd|60"] > 0, "Score"] += 1
    df.loc[df["MACD.macd"] > 0, "Score"] += 1

    return df.sort_values("Score", ascending=False)

# ================= UI =================
st.title("🧠 AI High Gain Dashboard – Saudi Market")

daily = fetch_daily()
m15 = fetch_15m()
h1 = fetch_1h()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 +5% اليوم",
    "🔮 XGBoost تنبؤ",
    "⏱️ Multi-Timeframe",
    "⚡ فرص +2%",
    "📊 السوق كامل",
    "💾 الملفات"
])

# -------- TAB 1 --------
with tab1:
    hg = daily[daily["change"] >= 5]
    st.dataframe(hg)
    if st.button("💾 حفظ"):
        save_csv(HIGH_GAIN_FILE, hg)
        st.success("تم الحفظ")

# -------- TAB 2 --------
with tab2:
    pred = xgboost_predict(daily)
    top = pred[pred["Prediction"] > 0.7].head(10)
    st.dataframe(top)
    if st.button("💾 حفظ التنبؤات"):
        save_csv(PRED_FILE, top)
        st.success("تم الحفظ")

# -------- TAB 3 --------
with tab3:
    mtf = multi_tf(daily, m15, h1).head(10)
    st.dataframe(mtf[[
        "Symbol","change","change|15","change|60","RSI","RSI|15","RSI|60","Score"
    ]])

# -------- TAB 4 --------
with tab4:
    st.dataframe(daily[daily["change"] >= 2])

# -------- TAB 5 --------
with tab5:
    st.dataframe(daily)

# -------- TAB 6 --------
with tab6:
    st.write("📁 High Gain File")
    st.dataframe(pd.read_csv(HIGH_GAIN_FILE) if os.path.exists(HIGH_GAIN_FILE) else [])
    st.write("📁 Predictions File")
    st.dataframe(pd.read_csv(PRED_FILE) if os.path.exists(PRED_FILE) else [])

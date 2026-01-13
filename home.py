import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
import xgboost as xgb

# ================= CONFIG =================
st.set_page_config("AI High Gain Dashboard – KSA", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HIGH_GAIN_FILE = os.path.join(DATA_DIR, "high_gain_today.csv")
PRED_FILE = os.path.join(DATA_DIR, "predictions.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "training.csv")

# ================= HELPERS =================
def safe_read(file):
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def safe_append(file, df, subset_cols=None):
    if df.empty:
        return
    df_to_save = df.copy()
    df_to_save["Date"] = pd.to_datetime("today").date()
    if os.path.exists(file) and os.stat(file).st_size > 0:
        try:
            existing = pd.read_csv(file)
            if subset_cols:
                df_to_save = pd.concat([existing, df_to_save]).drop_duplicates(subset=subset_cols)
            else:
                df_to_save = pd.concat([existing, df_to_save]).drop_duplicates()
        except pd.errors.EmptyDataError:
            pass
    df_to_save.to_csv(file, index=False)
    st.success(f"تم حفظ {len(df_to_save)} سهم في {file}")

# ================= TRADINGVIEW =================
def fetch_ksa():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "TADAWUL"},
            {"left": "type", "operation": "equal", "right": "stock"}
        ],
        "columns": [
            "name", "description", "close", "change",
            "relative_volume_10d_calc", "volume", "market_cap_basic"
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 400]
    }
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        st.error(f"⚠️ خطأ في جلب البيانات: {e}")
        return pd.DataFrame()
    rows = []
    for d in data:
        try:
            rows.append({
                "Symbol": d.get("s",""),
                "Company": str(d["d"][1]) if len(d["d"])>1 else "",
                "Price": float(d["d"][2]) if len(d["d"])>2 and d["d"][2] else 0.0,
                "Change %": float(d["d"][3]) if len(d["d"])>3 and d["d"][3] else 0.0,
                "Relative Volume": float(d["d"][4]) if len(d["d"])>4 and d["d"][4] else 0.0,
                "Volume": float(d["d"][5]) if len(d["d"])>5 and d["d"][5] else 0.0,
                "Market Cap": float(d["d"][6]) if len(d["d"])>6 and d["d"][6] else 0.0
            })
        except:
            continue
    return pd.DataFrame(rows)

# ================= INDICATORS =================
def compute_indicators(df):
    df = df.copy()
    try:
        df["EMA20"] = EMAIndicator(df["Price"], window=20).ema_indicator()
        df["EMA50"] = EMAIndicator(df["Price"], window=50).ema_indicator()
        df["EMA200"] = EMAIndicator(df["Price"], window=200).ema_indicator()
    except:
        df[["EMA20","EMA50","EMA200"]] = 0
    try:
        df["RSI"] = RSIIndicator(df["Price"]).rsi()
    except:
        df["RSI"] = 50
    try:
        df["MACD"] = MACD(df["Price"]).macd_diff()
    except:
        df["MACD"] = 0
    df.fillna(method="bfill", inplace=True)
    return df

# ================= SCORE & REASON =================
def score_stocks(df):
    df = df.copy()
    reasons = []
    scores = []
    for _, row in df.iterrows():
        score = 0
        reason = []
        if row["EMA20"] > row["EMA50"] > row["EMA200"]:
            score += 3
            reason.append("EMA صاعد")
        if 30 < row["RSI"] < 70:
            score += 2
            reason.append("RSI مناسب")
        if row["MACD"] > 0:
            score += 2
            reason.append("MACD إيجابي")
        if row["Relative Volume"] > 1.2:
            score += 1
            reason.append("حجم تداول مرتفع")
        reasons.append(", ".join(reason))
        scores.append(score)
    df["Score"] = scores
    df["سبب الترشيح"] = reasons
    return df

# ================= TOP 20 NEXT DAY =================
def top_20_next_day(df, timeframe="1H"):
    df_scored = score_stocks(df)
    df_sorted = df_scored.sort_values(by="Score", ascending=False)
    top20 = df_sorted.head(20).copy()
    # سعر الدخول والوقف والأهداف محسوبة حسب Timeframe (يمكن تعديل الصيغ لاحقًا)
    top20["سعر الدخول"] = top20["Price"]
    top20["وقف الخسارة"] = (top20["Price"] * 0.975).round(2)
    top20["جني الأرباح"] = (top20["Price"] * 1.05).round(2)
    top20["Timeframe"] = timeframe
    return top20

# ================= STREAMLIT =================
st.title("🧠 AI High Gain Dashboard – KSA")

df = fetch_ksa()
if df.empty:
    st.stop()

df = compute_indicators(df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 +5% اليوم",
    "🔮 تنبؤ الغد (Ensemble)",
    "🧠 التعلم والتقييم",
    "📊 Dashboard",
    "⚡ فرص +2% غدًا",
    "🔥 أفضل 20 سهم للغد"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5].copy()
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_append(HIGH_GAIN_FILE, high_gain, subset_cols=["Symbol","Date"])

# ---------- TAB 2 ----------
with tab2:
    features = ["Change %","Relative Volume","Volume","EMA20","EMA50","EMA200","RSI","MACD"]
    pred_df = df.copy()
    pred_df["Target"] = (pred_df["Change %"].shift(-1) >= 5).astype(int)
    pred_df.dropna(inplace=True)
    X = pred_df[features]
    y = pred_df["Target"]
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    pred_df["Predicted"] = model.predict(X)
    st.subheader("التنبؤ بالأسهم التي قد تحقق +5% غدًا")
    st.dataframe(pred_df[["Symbol","Company","Predicted"]], use_container_width=True)
    if st.button("💾 حفظ التنبؤات"):
        safe_append(PRED_FILE, pred_df[["Symbol","Company","Predicted"]], subset_cols=["Symbol","Date"])

# ---------- TAB 3 ----------
with tab3:
    st.subheader("تقييم التعلم")
    training_data = safe_read(TRAIN_FILE)
    st.dataframe(training_data, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    st.subheader("Dashboard")
    st.dataframe(df, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    potential = df[df["Change %"] >= 2].copy()
    st.subheader("فرص +2% تحليل مباشر")
    st.dataframe(potential, use_container_width=True)

# ---------- TAB 6 ----------
with tab6:
    timeframe_choice = st.selectbox("اختر Timeframe للتحليل", ["15m","1H"], index=1)
    top20 = top_20_next_day(df, timeframe=timeframe_choice)
    st.subheader(f"أفضل 20 سهم للغد – Timeframe {timeframe_choice}")
    st.dataframe(top20[[
        "Symbol","Company","Price","سعر الدخول","وقف الخسارة","جني الأرباح",
        "Score","سبب الترشيح","Timeframe"
    ]], use_container_width=True)

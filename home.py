import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ================= CONFIG =================
st.set_page_config("AI High Gain Dashboard – KSA", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

# مجلد البيانات
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
        st.warning("لا توجد بيانات للحفظ")
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
def fetch_ksa_stocks():
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
    df = pd.DataFrame(rows)
    df.dropna(subset=["Price"], inplace=True)
    return df

# ================= INDICATORS =================
def compute_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Price"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Price"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Price"].ewm(span=200, adjust=False).mean()

    # RSI يدوي
    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Price"].ewm(span=12, adjust=False).mean()
    ema26 = df["Price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ATR تقريبي
    df["ATR"] = df["Price"].rolling(14).max() - df["Price"].rolling(14).min()
    df.fillna(method="bfill", inplace=True)
    return df

# ================= ML =================
def train_xgboost(df):
    features = ["Change %","Relative Volume","Volume","EMA20","EMA50","EMA200","RSI","MACD","ATR"]
    df = df.copy()
    df["Target"] = (df["Change %"].shift(-1) >= 5).astype(int)
    df.dropna(inplace=True)
    if df.empty:
        return pd.DataFrame()
    X = df[features]
    y = df["Target"]
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    df["Predicted"] = model.predict(X)
    return df[["Symbol","Company","Predicted"]]

# ================= STREAMLIT ==================
st.title("🧠 AI High Gain Dashboard – KSA")

df = fetch_ksa_stocks()
if df.empty:
    st.stop()

df = compute_indicators(df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 +5% اليوم",
    "🔮 تنبؤ الغد (Ensemble)",
    "🧠 التعلم والتقييم",
    "📊 Dashboard",
    "⚡ فرص +2% غدًا",
    "⭐ أفضل 10 فرص الغد"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5].copy()
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        safe_append(HIGH_GAIN_FILE, high_gain, subset_cols=["Symbol","Date"])

# ---------- TAB 2 ----------
with tab2:
    pred = train_xgboost(df)
    st.subheader("التنبؤ بالأسهم التي قد تحقق +5% غدًا")
    st.dataframe(pred, use_container_width=True)
    if st.button("💾 حفظ التنبؤات"):
        safe_append(PRED_FILE, pred, subset_cols=["Symbol","Date"])

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
    # أفضل 10 أسهم للغد بناء على Change %, Relative Volume, RSI, MACD, ATR
    df_sorted = df.sort_values(
        by=["Change %","Relative Volume","RSI","MACD","ATR"], ascending=False
    ).head(10)
    st.subheader("أفضل 10 أسهم محتملة +5% الغد")
    st.dataframe(df_sorted[["Symbol","Company","Price","Change %","Relative Volume","RSI","MACD","ATR"]], use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import requests, os
from datetime import datetime

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ================== CONFIG ==================
st.set_page_config("AI High Gain Dashboard – TASI Only", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HIGH_GAIN_FILE = f"{DATA_DIR}/high_gain_today.csv"
PRED_FILE = f"{DATA_DIR}/predictions.csv"
TRAIN_FILE = f"{DATA_DIR}/training.csv"

BASE_FEATURES = ["Change %", "Relative Volume", "Volume"]
LSTM_FEATURES = [
    "Change %", "Relative Volume", "Volume",
    "EMA20", "EMA50", "EMA200", "RSI", "MACD", "ATR"
]

TIME_STEPS = 20

# ================== TASI SYMBOLS ==================
TASI_SYMBOLS = [
    "1010","1020","1030","1040","1050","1060","1080","1120","1140","1150",
    "1180","1201","1210","1211","1301","1302","1320","1330","1350",
    "2010","2020","2030","2040","2050","2060","2070","2080","2090","2100",
    "2110","2120","2130","2140","2150","2160","2170","2180","2190","2200",
    "2210","2220","2230","2240","2250","2270","2280","2290","2300",
    "2310","2320","2330","2340","2350","2360","2370","2380","3001","3002",
    "3003","3004","3010","3020","3030","3040","3050","3060","3080",
    "3090","3100","4001","4002","4003","4004","4011","4020","4030","4040",
    "4050","4061","4070","4080","4090","4100","4110","4130","4140","4150",
    "4160","4170","4180","4190","4200","4210","4220","4230","4240","4250",
    "4260","4270","4280","4290","4300","4310","4320","4330","4340","5110",
    "6001","6002","6004","6010","6020","6040","6050","6060","6070","6090",
    "7010","7020","7030","7040","8010","8020","8030","8040","8050","8060",
    "8070","8100","8120","8150","8160","8170","8180","8190","8200","8210",
    "8230","8240","8250","8260","8270","8280","8300","8310"
]

# ================== HELPERS ==================
def safe_read(file):
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(file)

def safe_append(file, df):
    if df.empty:
        return
    if os.path.exists(file):
        df = pd.concat([pd.read_csv(file), df]).drop_duplicates()
    df.to_csv(file, index=False)

# ================== TRADINGVIEW ==================
def fetch_tasi():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "TADAWUL"},
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "market_cap_basic", "operation": "greater", "right": 1_000_000_000}
        ],
        "columns": [
            "name", "description", "close",
            "change", "relative_volume_10d_calc",
            "volume", "market_cap_basic"
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 400]
    }

    r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
    data = r.json().get("data", [])

    rows = []
    for d in data:
        desc = str(d["d"][1])
        # Filter out Nomu / نمو
        if "نمو" in desc or "nomu" in desc.lower():
            continue

        rows.append({
            "Symbol": d["s"],
            "Company": desc,
            "Price": float(d["d"][2]),
            "Change %": float(d["d"][3]),
            "Relative Volume": float(d["d"][4]),
            "Volume": float(d["d"][5]),
            "Market Cap": float(d["d"][6])
        })

    df = pd.DataFrame(rows)
    # Apply strict TASI filter
    df["Code"] = df["Symbol"].str.split(":").str[-1]
    df = df[df["Code"].isin(TASI_SYMBOLS)]
    return df

# ================== INDICATORS ==================
def compute_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Price"].ewm(span=20).mean()
    df["EMA50"] = df["Price"].ewm(span=50).mean()
    df["EMA200"] = df["Price"].ewm(span=200).mean()

    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MACD"] = df["Price"].ewm(span=12).mean() - df["Price"].ewm(span=26).mean()
    df["ATR"] = df["Price"].rolling(14).max() - df["Price"].rolling(14).min()
    df.fillna(method="bfill", inplace=True)
    return df

# ================== XGBOOST / LSTM ==================
def train_xgb(df):
    X = df[BASE_FEATURES]
    y = df["Target"]
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_xgb(model, df):
    df["XGB_Prob"] = model.predict_proba(df[BASE_FEATURES])[:, 1]
    return df

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[LSTM_FEATURES])
    X, y = [], []
    for i in range(TIME_STEPS, len(df)):
        X.append(scaled[i-TIME_STEPS:i])
        y.append(df["Target"].iloc[i])
    return np.array(X), np.array(y), scaler

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_lstm(df):
    X, y, scaler = prepare_lstm_data(df)
    if len(X) < 30:
        return None, None
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    cw = dict(enumerate(weights))
    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(
        X, y, epochs=40, batch_size=32,
        validation_split=0.2,
        class_weight=cw,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    return model, scaler

def predict_lstm(model, scaler, df):
    if model is None:
        df["LSTM_Prob"] = 0.0
        return df
    scaled = scaler.transform(df[LSTM_FEATURES])
    seq = []
    for i in range(TIME_STEPS, len(df)):
        seq.append(scaled[i-TIME_STEPS:i])
    preds = model.predict(np.array(seq), verbose=0).flatten()
    df = df.iloc[TIME_STEPS:].copy()
    df["LSTM_Prob"] = preds
    return df

# ================== UI ==================
st.title("🧠 AI High Gain Dashboard – TASI Only")

df = fetch_tasi()
df = compute_indicators(df)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 +5% اليوم",
    "🔮 تنبؤ الغد (Ensemble)",
    "🧠 التعلم والتقييم",
    "📊 Dashboard",
    "⚡ فرص +2% غدًا (تحليل مباشر)"
])

# ---------- TAB 1 ----------
with tab1:
    high_gain = df[df["Change %"] >= 5].copy()
    st.dataframe(high_gain, use_container_width=True)
    if st.button("💾 حفظ +5% اليوم"):
        high_gain["Date"] = datetime.today().date()
        safe_append(HIGH_GAIN_FILE, high_gain)
        st.success("تم الحفظ")

# ---------- TAB 2 ----------
with tab2:
    training = safe_read(TRAIN_FILE)
    if len(training) < 50:
        st.warning("البيانات غير كافية للتنبؤ")
    else:
        xgb_model = train_xgb(training)
        lstm_model, scaler = train_lstm(training)
        df_pred = predict_xgb(xgb_model, df.copy())
        df_pred = predict_lstm(lstm_model, scaler, df_pred)
        df_pred["Final_Prob"] = np.where(
            df_pred["RSI"] > 60,
            0.6*df_pred["XGB_Prob"] + 0.4*df_pred["LSTM_Prob"],
            0.75*df_pred["XGB_Prob"] + 0.25*df_pred["LSTM_Prob"]
        )
        winners = df_pred[df_pred["Final_Prob"] >= 0.6].sort_values("Final_Prob", ascending=False)
        st.dataframe(winners, use_container_width=True)
        if st.button("💾 حفظ التوقعات"):
            winners["Date"] = datetime.today().date()
            safe_append(PRED_FILE, winners)
            st.success("تم الحفظ")

# ---------- TAB 3 ----------
with tab3:
    preds = safe_read(PRED_FILE)
    if preds.empty:
        st.info("لا توجد بيانات")
    else:
        merged = preds.merge(df[["Symbol", "Change %"]], on="Symbol", how="left")
        merged["Target"] = ((merged["Change %"] >= 5) & (merged["Volume"] > merged["Volume"].rolling(20).mean())).astype(int)
        merged["Reason"] = np.where(
            merged["Target"] == 1,
            "زخم قوي + حجم مرتفع + اتجاه إيجابي",
            "فشل اختراق / ضعف حجم"
        )
        st.dataframe(merged, use_container_width=True)
        if st.button("🧠 تحديث قاعدة التعلم"):
            safe_append(TRAIN_FILE, merged[LSTM_FEATURES + ["Target"]])
            st.success("تم تحديث التعلم")

# ---------- TAB 4 ----------
with tab4:
    col1, col2, col3 = st.columns(3)
    col1.metric("+5% اليوم", len(high_gain))
    col2.metric("توقعات محفوظة", len(safe_read(PRED_FILE)))
    col3.metric("حجم التعلم", len(safe_read(TRAIN_FILE)))
    st.bar_chart(df["Change %"])

# ---------- TAB 5 ----------
with tab5:
    st.subheader("⚡ أسهم مرشحة +2% غدًا (تاسي فقط – بدون تعلم)")
    candidates = df.copy()
    # ---------- Strict TASI Filter ----------
    candidates["Code"] = candidates["Symbol"].str.split(":").str[-1]
    candidates = candidates[candidates["Code"].isin(TASI_SYMBOLS)]

    candidates["Score"] = 0
    candidates.loc[(candidates["RSI"] >= 45) & (candidates["RSI"] <= 65), "Score"] += 1
    candidates.loc[candidates["Price"] > candidates["EMA20"], "Score"] += 1
    candidates.loc[candidates["EMA20"] > candidates["EMA50"], "Score"] += 1
    candidates.loc[candidates["MACD"] > 0, "Score"] += 1
    candidates.loc[candidates["Relative Volume"] >= 1.3, "Score"] += 1
    candidates.loc[(candidates["Change %"] >= -1) & (candidates["Change %"] <= 3), "Score"] += 1

    result = candidates[candidates["Score"] >= 3].sort_values(["Score","Relative Volume"], ascending=False)
    if result.empty:
        st.info("لا توجد فرص قوية حاليًا")
    else:
        st.dataframe(
            result[[
                "Symbol","Company","Price","Change %","RSI","Relative Volume","EMA20","EMA50","MACD","Score"
            ]],
            use_container_width=True
        )
    st.caption("""
    🧠 **المنهجية**:
    - زخم صحي
    - اتجاه صاعد قصير
    - حجم تداول داعم
    - لم يتحرك بقوة بعد
    """)

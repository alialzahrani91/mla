import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import xgboost as xgb

# ================= CONFIG =================
st.set_page_config("High Gain AI Dashboard", layout="wide")
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HIGH_GAIN_FILE = f"{DATA_DIR}/high_gain_today.csv"
PREDICT_FILE = f"{DATA_DIR}/predictions_tomorrow.csv"
TRAIN_FILE = f"{DATA_DIR}/training_data.csv"

FEATURES = ["Change %", "Relative Volume", "Volume"]

# ================= TRADINGVIEW =================
def fetch_tasi():
    url = "https://scanner.tradingview.com/ksa/scan"
    payload = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "TADAWUL"},
            {"left": "type", "operation": "equal", "right": "stock"}
        ],
        "columns": [
            "name", "description", "close",
            "change", "relative_volume_10d_calc",
            "volume", "exchange"
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 300]
    }

    r = requests.post(url, json=payload, headers=HEADERS, timeout=20)
    data = r.json().get("data", [])

    rows = []
    for d in data:
        if d["d"][-1] != "TADAWUL":
            continue
        rows.append({
            "Symbol": d["s"],
            "Company": d["d"][1],
            "Price": d["d"][2],
            "Change %": d["d"][3],
            "Relative Volume": d["d"][4],
            "Volume": d["d"][5]
        })
    return pd.DataFrame(rows)

# ================= MODEL =================
def train_xgb(df):
    X = df[FEATURES]
    y = df["Target"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_xgb(model, df):
    df["Probability"] = model.predict_proba(df[FEATURES])[:, 1]
    return df.sort_values("Probability", ascending=False)

# ================= FILE HELPERS =================
def safe_read(file):
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(file)

def safe_append(file, df):
    if df.empty:
        return
    if os.path.exists(file):
        df_all = pd.concat([pd.read_csv(file), df]).drop_duplicates()
    else:
        df_all = df
    df_all.to_csv(file, index=False)

# ================= UI =================
st.title("📊 High Gain AI Dashboard (TASI Only)")

df = fetch_tasi()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 +5% اليوم",
    "🔮 تنبؤ الغد (XGBoost)",
    "🧠 التعلم والتقييم",
    "📊 Dashboard"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("الأسهم التي حققت +5% اليوم")
    high_gain = df[df["Change %"] >= 5].copy()

    if high_gain.empty:
        st.info("لا توجد أسهم +5% اليوم")
    else:
        st.dataframe(high_gain, use_container_width=True)
        if st.button("💾 حفظ قائمة اليوم"):
            high_gain["Date"] = datetime.today().date()
            safe_append(HIGH_GAIN_FILE, high_gain)
            st.success("تم الحفظ")

# ================= TAB 2 =================
with tab2:
    st.subheader("الأسهم المتوقع تحقيق +5% غدًا")

    training = safe_read(TRAIN_FILE)

    if len(training) < 20:
        st.warning("قاعدة التعلم غير كافية بعد")
    else:
        model = train_xgb(training)
        preds = predict_xgb(model, df.copy())
        candidates = preds[preds["Probability"] >= 0.6]

        st.dataframe(candidates, use_container_width=True)

        if st.button("💾 حفظ التوقعات"):
            candidates["Date"] = datetime.today().date()
            safe_append(PREDICT_FILE, candidates)
            st.success("تم حفظ التوقعات")

# ================= TAB 3 =================
with tab3:
    st.subheader("تحليل نتائج التوقعات السابقة")

    preds_old = safe_read(PREDICT_FILE)
    if preds_old.empty:
        st.info("لا توجد توقعات محفوظة")
    else:
        merged = preds_old.merge(
            df[["Symbol", "Change %"]],
            on="Symbol",
            how="left",
            suffixes=("", "_Actual")
        )

        merged["Target"] = (merged["Change %_Actual"] >= 5).astype(int)
        merged["Reason"] = np.where(
            merged["Target"] == 1,
            "زخم قوي + حجم تداول",
            "فشل الاختراق / ضعف زخم"
        )

        st.dataframe(merged, use_container_width=True)

        if st.button("🧠 تحديث قاعدة التعلم"):
            safe_append(TRAIN_FILE, merged[FEATURES + ["Target"]])
            st.success("تم تحديث قاعدة التعلم")

# ================= TAB 4 =================
with tab4:
    st.subheader("📊 ملخص النظام")

    col1, col2, col3 = st.columns(3)
    col1.metric("+5% اليوم", len(high_gain))
    col2.metric("توقعات محفوظة", len(safe_read(PREDICT_FILE)))
    col3.metric("حجم التعلم", len(safe_read(TRAIN_FILE)))

    st.bar_chart(df["Change %"])

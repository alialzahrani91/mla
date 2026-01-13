import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime

from ta.momentum import RSIIndicator
from ta.trend import MACD

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ======================
# إعدادات عامة
# ======================
st.set_page_config(page_title="High Gain Stocks AI", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TODAY_FILE = f"{DATA_DIR}/today_5pct.csv"
PREDICT_FILE = f"{DATA_DIR}/predicted_next.csv"
LEARN_FILE = f"{DATA_DIR}/learning_db.csv"

# ======================
# جلب الأسهم (مثال سوق سعودي عبر Yahoo)
# ======================
@st.cache_data(ttl=3600)
def fetch_ksa_stocks():
    tickers = [
        "2222.SR", "1120.SR", "2010.SR", "1211.SR", "1180.SR",
        "7010.SR", "1060.SR", "5110.SR", "2290.SR", "4260.SR",
        "1050.SR", "4140.SR", "3002.SR", "2330.SR", "4002.SR"
    ]

    rows = []

    for t in tickers:
        df = yf.download(t, period="30d", interval="1d", progress=False)
        if df.empty:
            continue

        df.dropna(inplace=True)

        close = df["Close"]
        volume = df["Volume"]

        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd_val = MACD(close).macd().iloc[-1]

        change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100

        rows.append({
            "Symbol": t,
            "Company": t.replace(".SR", ""),
            "price": round(close.iloc[-1], 2),
            "change": round(change, 2),
            "volume": int(volume.iloc[-1]),
            "RSI": round(rsi, 2),
            "MACD": round(macd_val, 3)
        })

    return pd.DataFrame(rows)

# ======================
# حفظ CSV مع إنشاء تلقائي
# ======================
def safe_save(df, path):
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df]).drop_duplicates(subset=["Symbol"])
    df.to_csv(path, index=False)

# ======================
# مرشحي اختراق الغد (تحليل بدون تعلم)
# ======================
def next_day_breakout_candidates(df):
    df = df.copy()
    df = df[(df["change"] < 3) & (df["change"] > 0.3)]

    scores, reasons = [], []
    vol_med = df["volume"].median()

    for _, r in df.iterrows():
        score = 0
        reason = []

        if r["volume"] > vol_med:
            score += 2
            reason.append("تجميع حجم تداول")

        if 45 < r["RSI"] < 60:
            score += 2
            reason.append("RSI مناسب للانطلاق")

        if -0.2 < r["MACD"] < 0.15:
            score += 2
            reason.append("MACD قريب من التقاطع")

        if 0.5 < r["change"] < 2:
            score += 1
            reason.append("تحرك سعري هادئ")

        scores.append(score)
        reasons.append(" + ".join(reason))

    df["Breakout Score"] = scores
    df["Reason"] = reasons

    return (
        df[df["Breakout Score"] >= 5]
        .sort_values("Breakout Score", ascending=False)
        .head(20)
    )

# ======================
# واجهة التطبيق
# ======================
st.title("📊 High Gain Stocks AI Dashboard")

df = fetch_ksa_stocks()

tabs = st.tabs([
    "📈 +5% اليوم",
    "🔮 توقع الغد",
    "🧠 التعلّم",
    "🤖 تنبؤ ذكي",
    "🚀 مرشحي اختراق الغد"
])

# ======================
# TAB 1
# ======================
with tabs[0]:
    st.subheader("الأسهم التي حققت +5% اليوم")
    high_gain = df[df["change"] >= 5]

    st.dataframe(high_gain, use_container_width=True)

    if st.button("💾 حفظ +5% اليوم"):
        if not high_gain.empty:
            safe_save(high_gain, TODAY_FILE)
            st.success(f"تم الحفظ في {TODAY_FILE}")
        else:
            st.warning("لا توجد أسهم")

# ======================
# TAB 2
# ======================
with tabs[1]:
    st.subheader("الأسهم المتوقعة +5% الغد (بدون تعلم)")
    predicted = df[(df["RSI"] < 60) & (df["volume"] > df["volume"].median())]

    st.dataframe(predicted, use_container_width=True)

    if st.button("💾 حفظ التوقعات"):
        safe_save(predicted, PREDICT_FILE)
        st.success(f"تم الحفظ في {PREDICT_FILE}")

# ======================
# TAB 3
# ======================
with tabs[2]:
    st.subheader("تحليل نتائج التوقعات")

    if os.path.exists(PREDICT_FILE):
        pred = pd.read_csv(PREDICT_FILE)
        merged = pred.merge(df[["Symbol", "change"]], on="Symbol", how="left")
        merged["Success"] = merged["change"] >= 5
        merged["Date"] = datetime.today().date()

        safe_save(merged, LEARN_FILE)
        st.dataframe(merged, use_container_width=True)
    else:
        st.info("لا توجد بيانات تعلم")

# ======================
# TAB 4 (XGBoost)
# ======================
with tabs[3]:
    st.subheader("تنبؤ ذكي باستخدام XGBoost")

    if os.path.exists(LEARN_FILE):
        data = pd.read_csv(LEARN_FILE)

        features = ["RSI", "MACD", "volume", "change"]
        data.dropna(inplace=True)

        X = data[features]
        y = data["Success"].astype(int)

        if len(y.unique()) > 1:
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X, y)

            probs = model.predict_proba(df[features])[:, 1]
            df["AI Probability"] = probs

            st.dataframe(
                df.sort_values("AI Probability", ascending=False).head(10),
                use_container_width=True
            )
        else:
            st.warning("البيانات غير كافية للتعلم")
    else:
        st.info("لا توجد قاعدة تعلم بعد")

# ======================
# TAB 5 (الجديد)
# ======================
with tabs[4]:
    st.subheader("🚀 مرشحي اختراق الغد 5%")
    next_day = next_day_breakout_candidates(df)

    st.dataframe(
        next_day[
            ["Symbol", "Company", "change", "RSI", "MACD", "volume",
             "Breakout Score", "Reason"]
        ],
        use_container_width=True
    )

    if st.button("💾 حفظ مرشحي الغد"):
        safe_save(next_day, f"{DATA_DIR}/breakout_candidates.csv")
        st.success("تم الحفظ بنجاح")

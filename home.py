import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os

st.set_page_config(page_title="High Gain Stocks Auto Update & Prediction", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
TRAINING_FILE = "training_data.csv"
PREDICTED_FILE = "predicted_today.csv"

# =============================
# جلب بيانات السوق من TradingView
# =============================
@st.cache_data(ttl=300)
def fetch_tradingview_market(market_code):
    url = f"https://scanner.tradingview.com/{market_code}/scan"
    payload = {
        "filter": [],
        "symbols": {"query":{"types":[]}, "tickers":[]},
        "columns":["name","description","close","change","relative_volume_10d_calc",
                   "price_earnings_ttm","volume"],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 300]  # أول 300 سهم
    }
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
    except:
        st.warning("⚠️ فشل الاتصال بـ TradingView")
        return pd.DataFrame()
    
    rows = []
    for item in data:
        try:
            rows.append({
                "Symbol": item["s"],
                "Company": item["d"][1],
                "Price": float(item["d"][2]),
                "Change %": float(item["d"][3]),
                "Relative Volume": float(item["d"][4]),
                "PE": float(item["d"][5]) if item["d"][5] else None,
                "Volume": float(item["d"][6]) if len(item["d"])>6 else None
            })
        except: continue
    return pd.DataFrame(rows)

# =============================
# حفظ High Gain كقاعدة تدريب
# =============================
def save_training_data(df):
    if os.path.exists(TRAINING_FILE):
        existing = pd.read_csv(TRAINING_FILE)
        df_all = pd.concat([existing, df], ignore_index=True)
    else:
        df_all = df
    df_all.to_csv(TRAINING_FILE, index=False)
    st.success(f"✅ تم تحديث قاعدة التدريب بعدد {len(df)} سهم")

# =============================
# تدريب النموذج والتنبؤ بالأسهم المتوقع +5%
# =============================
def train_predict_high_gain(df_current):
    if not os.path.exists(TRAINING_FILE) or os.path.getsize(TRAINING_FILE)==0:
        st.info("❌ لا يوجد بيانات تدريب للتنبؤ")
        return pd.DataFrame()
    
    training = pd.read_csv(TRAINING_FILE)
    if training.empty:
        st.info("❌ ملف التدريب فارغ")
        return pd.DataFrame()
    
    le = LabelEncoder()
    training["symbol_code"] = le.fit_transform(training["Symbol"].astype(str))
    
    features = ["Price","Change %","Relative Volume","PE","Volume","symbol_code"]
    X = training[features].fillna(0)
    y = (training["Change %"].shift(-1)>=5).astype(int)[:-1]
    X = X[:-1]
    
    if len(y)<5:
        st.info("❌ بيانات التدريب قليلة للتنبؤ")
        return pd.DataFrame()
    
    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)
    
    df_curr = df_current.copy()
    df_curr["symbol_code"] = le.transform(df_curr["Symbol"].astype(str))
    X_curr = df_curr[features].fillna(0)
    df_curr["Probability +5%"] = model.predict_proba(X_curr)[:,1]
    
    predicted = df_curr[df_curr["Probability +5%"]>=0.5].sort_values("Probability +5%", ascending=False)
    
    if not predicted.empty:
        predicted.to_csv(PREDICTED_FILE, index=False)
    
    return predicted

# =============================
# تحديث يومي تلقائي
# =============================
def daily_update(market_code):
    df = fetch_tradingview_market(market_code)
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    high_gain = df[df["Change %"]>=5]
    if not high_gain.empty:
        save_training_data(high_gain)
    
    predicted = train_predict_high_gain(df)
    return df, high_gain, predicted

# =============================
# واجهة المستخدم
# =============================
st.title("📈 High Gain Stocks Auto Update & Prediction")

market = st.selectbox("اختر السوق", ["السعودي","الأمريكي"])
market_code = "ksa" if market=="السعودي" else "america"

# جلب وعرض بيانات السوق فور اختيار السوق
with st.spinner("جارٍ جلب بيانات السوق..."):
    df, high_gain, predicted = daily_update(market_code)

st.subheader("📊 أسهم السوق")
if df.empty:
    st.info("لا توجد بيانات حالياً")
else:
    st.dataframe(df,use_container_width=True,hide_index=True)

st.subheader("📈 أسهم حققت +5% اليوم")
if high_gain.empty:
    st.info("لا توجد أسهم حققت +5% اليوم")
else:
    st.dataframe(high_gain,use_container_width=True,hide_index=True)

st.subheader("🔮 الأسهم المتوقع +5% غدًا")
if predicted.empty:
    st.info("لا توجد أسهم متوقعة تحقيق +5% اليوم")
else:
    st.dataframe(predicted[["Symbol","Company","Price","Change %","Relative Volume","PE","Volume","Probability +5%"]],
                 use_container_width=True,hide_index=True)

# زر لتحديث البيانات مرة أخرى (اختياري)
if st.button("🔄 تحديث البيانات"):
    with st.spinner("جارٍ تحديث البيانات وتحليل الأسهم..."):
        df, high_gain, predicted = daily_update(market_code)
        st.experimental_rerun()

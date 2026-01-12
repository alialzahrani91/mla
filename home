import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os
from datetime import datetime

st.set_page_config(page_title="High Gain Stocks Auto Update", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
WATCHLIST_FILE = "watchlist.csv"
TRAINING_FILE = "training_data.csv"
PREDICTED_FILE = "predicted_today.csv"

# =============================
# Helper Functions
# =============================
@st.cache_data(ttl=300)
def load_watchlist():
    try:
        return pd.read_csv(WATCHLIST_FILE)["Symbol"].dropna().tolist()
    except:
        st.error("❌ خطأ في قراءة ملف CSV")
        return []

@st.cache_data(ttl=300)
def fetch_tradingview_data(market, tickers):
    if not tickers: return pd.DataFrame()
    url = f"https://scanner.tradingview.com/{market}/scan"
    payload = {"filter": [], "symbols": {"tickers": tickers},
               "columns":["name","description","close","change","relative_volume_10d_calc","price_earnings_ttm","volume"]}
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=15)
        r.raise_for_status()
        raw = r.json().get("data", [])
    except:
        st.warning("⚠️ فشل الاتصال بـ TradingView")
        return pd.DataFrame()
    rows=[]
    for item in raw:
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

def save_training_data(df):
    """حفظ High Gain لتكون قاعدة تعلم"""
    if os.path.exists(TRAINING_FILE):
        existing = pd.read_csv(TRAINING_FILE)
        df_all = pd.concat([existing, df], ignore_index=True)
    else:
        df_all = df
    df_all.to_csv(TRAINING_FILE, index=False)
    st.success(f"✅ تم تحديث قاعدة التدريب بعدد {len(df)} سهم")

def train_predict_high_gain(df_current):
    """تدريب نموذج XGBoost والتنبؤ بالأسهم المتوقع +5%"""
    if not os.path.exists(TRAINING_FILE):
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

def daily_update(market_code, tickers):
    """تحديث يومي تلقائي للبيانات + قاعدة التدريب + ML"""
    df = fetch_tradingview_data(market_code, tickers)
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    high_gain = df[df["Change %"]>=5]
    if not high_gain.empty:
        save_training_data(high_gain)
    
    predicted = train_predict_high_gain(df)
    return high_gain, predicted

# =============================
# UI
# =============================
st.title("📈 High Gain Stocks Auto Update & Prediction")

market = st.selectbox("اختر السوق", ["السعودي","الأمريكي"])
market_code = "ksa" if market=="السعودي" else "america"
tickers = load_watchlist()
st.write(f"✅ عدد الأسهم في Watchlist: {len(tickers)}")

if st.button("🔄 تحديث اليوم"):
    with st.spinner("جارٍ تحديث البيانات وتحليل الأسهم..."):
        high_gain, predicted = daily_update(market_code, tickers)
        
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

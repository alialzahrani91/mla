import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

st.set_page_config(page_title="High Gain Stocks Dashboard", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
HIGH_GAIN_FILE = "high_gain_today.csv"
PREDICTED_FILE = "predicted.csv"
TRAINING_FILE = "training_data.csv"

# =============================
# جلب بيانات السوق من TradingView مع فلترة أسهم تاسي فقط
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
        "range": [0, 300]
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
            symbol = item["s"]
            # تصفية أسهم تاسي فقط للسوق السعودي
            if market_code == "ksa" and not symbol.startswith("TADAWUL:"):
                continue
            
            rows.append({
                "Symbol": symbol,
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
# حفظ CSV بحماية من EmptyDataError
# =============================
def safe_save_csv(df, filename):
    if df.empty:
        st.info(f"⚠️ الملف {filename} فارغ ولم يتم الحفظ")
        return
    df.to_csv(filename, index=False)
    st.success(f"✅ تم حفظ الملف: {filename} ({len(df)} صفوف)")

def safe_read_csv(filename, columns=None):
    try:
        df = pd.read_csv(filename)
        if df.empty and columns:
            df = pd.DataFrame(columns=columns)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        if columns:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame()

# =============================
# التاب الأول: High Gain اليوم
# =============================
def tab_high_gain_today(market_code):
    st.subheader("📈 أسهم اليوم +5%")
    df = fetch_tradingview_market(market_code)
    if df.empty:
        st.info("لا توجد بيانات حالياً")
        return pd.DataFrame()
    
    high_gain = df[df["Change %"] >= 5]
    st.dataframe(high_gain,use_container_width=True,hide_index=True)
    
    safe_save_csv(high_gain, HIGH_GAIN_FILE)
    return high_gain

# =============================
# التاب الثاني: التنبؤ بالأسهم المحتملة غداً
# =============================
def tab_predicted(df_current):
    st.subheader("🔮 أسهم متوقعة +5% غدًا")
    if df_current.empty:
        st.info("لا توجد أسهم لتحليل التنبؤ")
        return pd.DataFrame()
    
    # نموذج مبسط: اختيار أسهم قوية نسبياً من حيث Change% و Relative Volume
    df_current["Score"] = df_current["Change %"]*0.6 + df_current["Relative Volume"]*0.4
    predicted = df_current.sort_values("Score", ascending=False).head(30)
    
    st.dataframe(predicted,use_container_width=True,hide_index=True)
    safe_save_csv(predicted, PREDICTED_FILE)
    return predicted

# =============================
# التاب الثالث: تقييم الأسهم بعد الإغلاق
# =============================
def tab_evaluate():
    st.subheader("📊 تقييم الأسهم المتوقعة")
    columns = ["Symbol","Company","Price","Change %","Relative Volume","PE","Volume"]
    predicted = safe_read_csv(PREDICTED_FILE, columns=columns)
    if predicted.empty:
        st.info("لا توجد أسهم للتقييم")
        return pd.DataFrame()
    
    # جلب السوق الحالي لتقييم الأداء
    df_market = fetch_tradingview_market("ksa")  # للسوق السعودي، أسهم تاسي فقط
    merged = pd.merge(predicted, df_market[["Symbol","Change %"]], on="Symbol", how="left", suffixes=("","_today"))
    
    merged["Achieved +5%"] = merged["Change %_today"] >= 5
    merged["Reason"] = np.where(merged["Achieved +5%"], "📈 تحقق +5%", "⚠️ لم يتحقق +5%")
    
    st.dataframe(merged,use_container_width=True,hide_index=True)
    
    # حفظ كقاعدة للتعلم
    safe_save_csv(merged, TRAINING_FILE)
    return merged

# =============================
# التاب الرابع: التنبؤ بناءً على التعلم
# =============================
def tab_predict_learning():
    st.subheader("🔮 تنبؤات غدًا بناءً على التعلم")
    training = safe_read_csv(TRAINING_FILE)
    if training.empty:
        st.info("❌ لا توجد بيانات تعلم كافية")
        return pd.DataFrame()
    
    le = LabelEncoder()
    training["symbol_code"] = le.fit_transform(training["Symbol"].astype(str))
    
    features = ["Price","Change %","Relative Volume","PE","Volume","symbol_code"]
    X = training[features].fillna(0)
    y = training["Achieved +5%"].astype(int)
    
    if y.nunique() < 2:
        st.info("❌ بيانات غير كافية للتنبؤ")
        return pd.DataFrame()
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # التنبؤ باستخدام آخر يوم High Gain
    last_day = safe_read_csv(HIGH_GAIN_FILE)
    if last_day.empty:
        st.info("❌ لا توجد بيانات High Gain اليوم للتنبؤ")
        return pd.DataFrame()
    
    last_day["symbol_code"] = le.transform(last_day["Symbol"].astype(str))
    X_last = last_day[features].fillna(0)
    last_day["Probability +5%"] = model.predict_proba(X_last)[:,1]
    
    predicted = last_day[last_day["Probability +5%"] >= 0.5]
    st.dataframe(predicted,use_container_width=True,hide_index=True)
    safe_save_csv(predicted, "predicted_learning.csv")
    return predicted

# =============================
# واجهة Streamlit
# =============================
st.title("📊 High Gain Stocks Dashboard")
tabs = st.tabs(["High Gain اليوم","تنبؤ محتمل غدًا","تقييم الأسهم","تنبؤ التعلم"])

market_choice = st.selectbox("اختر السوق", ["السعودي","الأمريكي"])
market_code = "ksa" if market_choice=="السعودي" else "america"

with tabs[0]:
    high_gain = tab_high_gain_today(market_code)

with tabs[1]:
    predicted = tab_predicted(high_gain)

with tabs[2]:
    evaluated = tab_evaluate()

with tabs[3]:
    predicted_learning = tab_predict_learning()

# ---------- TAB 8: فرص قوية مع أهداف ----------
tab8 = st.tab("🚀 فرص قوية مع أهداف")

with tab8:
    # الفرص القوية: MACD إيجابي + سيولة عالية + إغلاق أعلى EMA20
    strong_ops = df[
        (df["MACD"] > 0) &
        (df["Relative Volume"] > 1.3) &
        (df["Price"] > df["EMA20"])
    ].sort_values("NextDayScore", ascending=False).head(20)

    # حساب سعر الدخول والوقف والهدف
    strong_ops = strong_ops.copy()
    strong_ops["سعر الدخول"] = strong_ops["Price"]
    strong_ops["وقف الخسارة"] = strong_ops["EMA50"]  # أو EMA20 حسب رغبتك
    strong_ops["الهدف"] = strong_ops["Price"] * 1.05  # هدف 5% تقريبًا

    st.subheader("فرص قوية للغد مع سعر الدخول والوقف والأهداف")
    st.dataframe(strong_ops[[
        "Symbol","Company","سعر الدخول","وقف الخسارة","الهدف","NextDayScore","RSI","MACD","Relative Volume","سبب الترشيح"
    ]], use_container_width=True)

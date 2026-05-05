import streamlit as st
import joblib
import pandas as pd

# =========================
# Load trained models
# =========================
models = {
    "Logistic Regression": joblib.load("linreg_best_model.pkl"),
    "Random Forest": joblib.load("rf_best_model.pkl"),
    "SVM": joblib.load("svm_best_model.pkl")
}

# =========================
# Feature names
# =========================
feature_names = [
    'LB','AC','FM','UC','DL','DS','DP',
    'ASTV','MSTV','ALTV','MLTV',
    'Width','Min','Max','Nmax','Nzeros',
    'Mode','Mean','Median','Variance','Tendency'
]

# =========================
# UI Header
# =========================
st.set_page_config(page_title="ML Model Comparison", layout="wide")

st.title("🧠 Cardiotocography Prediction Dashboard")
st.markdown("Compare predictions from multiple ML models in real-time.")

# =========================
# Input Section
# =========================
st.subheader("📥 Enter Patient Data")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])[feature_names]

# =========================
# Prediction Section
# =========================
if st.button("🚀 Predict with All Models"):

    st.subheader("📊 Model Predictions")

    results = {}

    for name, model in models.items():
        pred = model.predict(input_df)[0]
        results[name] = pred

    # Display in columns (fancy layout)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Logistic Regression", results["Logistic Regression"])

    with col2:
        st.metric("Random Forest", results["Random Forest"])

    with col3:
        st.metric("SVM", results["SVM"])

    # =========================
    # Summary Table
    # =========================
    st.subheader("📋 Comparison Table")

    result_df = pd.DataFrame([results])
    st.dataframe(result_df)

    # =========================
    # Agreement Check
    # =========================
    if len(set(results.values())) == 1:
        st.success("✅ All models agree on the prediction!")
    else:
        st.warning("⚠ Models disagree — check uncertainty in data.")

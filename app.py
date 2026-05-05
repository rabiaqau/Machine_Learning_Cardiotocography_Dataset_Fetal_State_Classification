import streamlit as st
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Patient Risk Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================
models = {
    "Logistic Regression": joblib.load("linreg_best_model.pkl"),
    "Random Forest": joblib.load("rf_best_model.pkl"),
    "SVM": joblib.load("svm_best_model.pkl")
}

rf_model = models["Random Forest"]

# =========================
# FULL FEATURE SET
# =========================
feature_names = [
    'LB','AC','FM','UC','DL','DS','DP',
    'ASTV','MSTV','ALTV','MLTV',
    'Width','Min','Max','Nmax','Nzeros',
    'Mode','Mean','Median','Variance','Tendency'
]

# =========================
# FEATURE IMPORTANCE (TOP FEATURES)
# =========================
importances = rf_model.named_steps["model"].feature_importances_
feature_importance = pd.Series(importances, index=feature_names)
top_features = feature_importance.sort_values(ascending=False).head(7).index.tolist()

# =========================
# UI HEADER
# =========================
st.title("🧠 AI Patient Risk Prediction System")
st.markdown("Choose analysis type and compare multiple ML models")

st.markdown("---")

# =========================
# MODE SELECTION
# =========================
mode = st.radio(
    "Select Analysis Mode",
    ["⚡ Quick Analysis", "📊 Detailed Analysis"]
)

st.markdown("---")

# =========================
# FEATURE SELECTION BASED ON MODE
# =========================
if mode == "⚡ Quick Analysis":
    selected_features = top_features
    st.info(f"Using top important features: {selected_features}")

else:
    selected_features = feature_names
    st.warning("Using full feature set for detailed analysis")

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Enter Patient Data")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(selected_features):
    with cols[i % 3]:
        input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Patient Risk"):

    st.subheader("📊 Model Predictions")

    results = {}

    for name, model in models.items():
        pred = model.predict(input_df)[0]
        results[name] = pred

    # =========================
    # DISPLAY RESULTS
    # =========================
    col1, col2, col3 = st.columns(3)

    def show_card(title, value, col):
        with col:
            if value == 1:
                st.success(f"{title} → Normal 🟢")
            elif value == 2:
                st.warning(f"{title} → Suspicious 🟡")
            else:
                st.error(f"{title} → Pathological 🔴")

    show_card("Logistic Regression", results["Logistic Regression"], col1)
    show_card("Random Forest", results["Random Forest"], col2)
    show_card("SVM", results["SVM"], col3)

    st.markdown("---")

    # =========================
    # SUMMARY TABLE
    # =========================
    st.subheader("📋 Prediction Summary")

    df_results = pd.DataFrame([results])
    st.dataframe(df_results)

    # =========================
    # AGREEMENT CHECK
    # =========================
    if len(set(results.values())) == 1:
        st.success("✅ All models agree on diagnosis")
    else:
        st.warning("⚠ Models disagree → review patient risk carefully")

    st.markdown("---")

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Model Comparison")

    st.bar_chart(df_results.T)

    # =========================
    # FEATURE INFO (ONLY FOR QUICK MODE)
    # =========================
    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Feature Importance Insight")

        st.write(feature_importance.sort_values(ascending=False))

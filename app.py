import streamlit as st
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Mother & Baby Risk Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# BACKGROUND IMAGE + STYLE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1584515933487-779824d29309");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: rgba(0, 0, 0, 0.60);
        z-index: 0;
    }

    .main {
        position: relative;
        z-index: 1;
        color: white;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.title("🧠 AI Mother & Baby Risk Prediction System")
st.markdown("Multi-model clinical decision support dashboard")

st.markdown("---")

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
# FEATURES
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
# MODE SELECTION
# =========================
mode = st.radio(
    "Select Analysis Mode",
    ["⚡ Quick Analysis", "📊 Detailed Analysis"]
)

st.markdown("---")

# =========================
# SELECT FEATURES BASED ON MODE
# =========================
if mode == "⚡ Quick Analysis":
    selected_features = top_features
    st.info(f"Using key features: {selected_features}")
else:
    selected_features = feature_names
    st.warning("Using full clinical feature set")

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Patient Input")

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
if st.button("🚀 Predict Risk Level"):

    st.subheader("📊 Model Predictions")

    results = {}

    for name, model in models.items():
        results[name] = model.predict(input_df)[0]

    # =========================
    # DISPLAY RESULTS
    # =========================
    col1, col2, col3 = st.columns(3)

    def show_result(name, value, col):
        with col:
            if value == 1:
                st.success(f"{name} → Normal 🟢")
            elif value == 2:
                st.warning(f"{name} → Suspicious 🟡")
            else:
                st.error(f"{name} → Pathological 🔴")

    show_result("Logistic Regression", results["Logistic Regression"], col1)
    show_result("Random Forest", results["Random Forest"], col2)
    show_result("SVM", results["SVM"], col3)

    st.markdown("---")

    # =========================
    # TABLE
    # =========================
    st.subheader("📋 Prediction Summary")

    df_results = pd.DataFrame([results])
    st.dataframe(df_results, use_container_width=True)

    # =========================
    # AGREEMENT CHECK
    # =========================
    if len(set(results.values())) == 1:
        st.success("✅ All models agree on diagnosis")
    else:
        st.warning("⚠ Models disagree — further review needed")

    st.markdown("---")

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Model Comparison Chart")

    st.bar_chart(df_results.T)

    # =========================
    # FEATURE INSIGHT (QUICK MODE ONLY)
    # =========================
    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Key Feature Importance")

        st.write(feature_importance.sort_values(ascending=False))

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Mother & Baby Risk System",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
        url("https://raw.githubusercontent.com/rabiaqau/Machine_Learning_Cardiotocography_Dataset_Fetal_State_Classification/main/mother_baby_image.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stApp { color: #e2e8f0; }
    p, label, div, span { color: #e2e8f0 !important; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700; }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover { background-color: #1d4ed8; }
    input { border-radius: 6px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 AI Mother & Baby Risk Prediction System")
st.markdown("Clinical decision support using multiple ML models")
st.markdown("---")

# =========================
# LOAD MODELS
# =========================
models = {
    "Logistic Regression": joblib.load("linreg_best_model.pkl"),
    "Random Forest":       joblib.load("rf_best_model.pkl"),
    "SVM":                 joblib.load("svm_best_model.pkl")
}

rf_model = models["Random Forest"]

# =========================
# FEATURES
# =========================
feature_names = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

importances = rf_model.named_steps["model"].feature_importances_
feature_importance = pd.Series(importances, index=feature_names)
top_features = feature_importance.sort_values(ascending=False).head(7).index.tolist()

# =========================
# MODE
# =========================
mode = st.radio("Select Analysis Mode", ["⚡ Quick Analysis", "📊 Detailed Analysis"])
st.markdown("---")

if mode == "⚡ Quick Analysis":
    displayed_features = top_features
    st.info(f"Using key clinical indicators: {displayed_features}")
else:
    displayed_features = feature_names
    st.warning("Full diagnostic feature set enabled")

# =========================
# INPUT
# =========================
st.subheader("📥 Patient Input")

# Pre-fill ALL 21 features as Python floats
input_data = {feat: 0.0 for feat in feature_names}

cols = st.columns(3)
for i, feature in enumerate(displayed_features):
    with cols[i % 3]:
        input_data[feature] = float(st.number_input(feature, value=0.0))

# Build DataFrame with all 21 features, explicitly cast to float64
input_df = pd.DataFrame([input_data], columns=feature_names).astype(np.float64)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Run Risk Prediction"):

    results = {}
    errors = {}

    for name, model in models.items():
        try:
            results[name] = int(model.predict(input_df)[0])
        except Exception as e:
            errors[name] = str(e)

    if errors:
        st.error("⚠️ Prediction errors occurred:")
        for model_name, err in errors.items():
            st.code(f"{model_name}: {err}")
        st.stop()

    st.subheader("📊 Model Outputs")
    col1, col2, col3 = st.columns(3)

    def show(name, val, col):
        with col:
            if val == 1:
                st.success(f"{name}\nNormal 🟢")
            elif val == 2:
                st.warning(f"{name}\nSuspicious 🟡")
            else:
                st.error(f"{name}\nPathological 🔴")

    show("Logistic Regression", results["Logistic Regression"], col1)
    show("Random Forest",       results["Random Forest"],       col2)
    show("SVM",                 results["SVM"],                 col3)

    st.markdown("---")

    st.subheader("📋 Summary")
    df_results = pd.DataFrame([results])
    st.dataframe(df_results, use_container_width=True)

    if len(set(results.values())) == 1:
        st.success("✅ All models agree on prediction")
    else:
        st.warning("⚠️ Model disagreement detected — review outputs carefully")

    st.bar_chart(df_results.T)

    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Feature Importance")
        st.write(feature_importance.sort_values(ascending=False))

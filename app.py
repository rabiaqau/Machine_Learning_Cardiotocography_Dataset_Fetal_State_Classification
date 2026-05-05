import streamlit as st
import joblib
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Mother & Baby Risk System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# FIXED BACKGROUND (IMPORTANT)
# =========================
st.markdown(
    """
    <style>

    /* =========================
       BACKGROUND IMAGE (STRONGER + BRIGHTER)
    ========================= */
    [data-testid="stAppViewContainer"] {
        background: url("https://raw.githubusercontent.com/rabiaqau/Machine_Learning_Cardiotocography_Dataset_Fetal_State_Classification/main/mother_baby_image.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* STRONG DARK OVERLAY (improves contrast) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.55);  /* darker = better readability */
        top: 0;
        left: 0;
        z-index: 0;
    }

    /* keep content above background */
    .main {
        position: relative;
        z-index: 1;
    }

    /* =========================
       GLASS CARD UI
    ========================= */
    .block-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* =========================
       TEXT STYLE
    ========================= */
    h1, h2, h3 {
        color: #1f2a44 !important;
    }

    p, label, div {
        color: #2c3e50 !important;
    }

    /* =========================
       🚀 BEAUTIFUL BUTTON STYLE (FIXED)
    ========================= */
    .stButton>button {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 14px;
        height: 3.2em;
        width: 100%;
        border: none;
        box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(255, 75, 43, 0.6);
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.title("🧠 AI Mother & Baby Risk Prediction System")
st.markdown("Multi-model comparison dashboard for clinical decision support")

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
# FEATURE IMPORTANCE
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
# INPUT FIELDS
# =========================
if mode == "⚡ Quick Analysis":
    selected_features = top_features
    st.info(f"Using key features: {selected_features}")
else:
    selected_features = feature_names
    st.warning("Using full feature set")

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
if st.button("🚀 Predict Risk"):

    results = {}

    for name, model in models.items():
        results[name] = model.predict(input_df)[0]

    st.subheader("📊 Model Predictions")

    col1, col2, col3 = st.columns(3)

    def show(name, val, col):
        with col:
            if val == 1:
                st.success(f"{name} → Normal 🟢")
            elif val == 2:
                st.warning(f"{name} → Suspicious 🟡")
            else:
                st.error(f"{name} → Pathological 🔴")

    show("Logistic Regression", results["Logistic Regression"], col1)
    show("Random Forest", results["Random Forest"], col2)
    show("SVM", results["SVM"], col3)

    st.markdown("---")

    st.subheader("📋 Summary Table")

    df_results = pd.DataFrame([results])
    st.dataframe(df_results, use_container_width=True)

    if len(set(results.values())) == 1:
        st.success("✅ All models agree")
    else:
        st.warning("⚠ Model disagreement detected")

    st.markdown("---")

    st.subheader("📊 Comparison")

    st.bar_chart(df_results.T)

    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Feature Importance")
        st.write(feature_importance.sort_values(ascending=False))

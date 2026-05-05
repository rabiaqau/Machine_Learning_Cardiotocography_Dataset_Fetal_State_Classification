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
# PROFESSIONAL BACKGROUND + THEME
# =========================
st.markdown(
    """
    <style>

    /* =========================
       BACKGROUND IMAGE (FIXED)
    ========================= */
    [data-testid="stAppViewContainer"] {
        position: relative;
        background: url("https://raw.githubusercontent.com/rabiaqau/Machine_Learning_Cardiotocography_Dataset_Fetal_State_Classification/main/mother_baby_image.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* DARK OVERLAY (CLEAN & PROFESSIONAL) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(10, 15, 25, 0.65);
        z-index: 0;
    }

    /* KEEP CONTENT ABOVE */
    .main {
        position: relative;
        z-index: 2;
    }

    /* =========================
       GLASS CARD UI (MAIN CONTAINER)
    ========================= */
    .block-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 2rem 2rem;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
        backdrop-filter: blur(12px);
    }

    /* =========================
       TYPOGRAPHY
    ========================= */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700;
    }

    p, label, div {
        color: #1f2937 !important;
        font-size: 15px;
    }

    /* =========================
       BUTTONS (MEDICAL / TRUST STYLE)
    ========================= */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: white;
        font-size: 16px;
        font-weight: 600;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        border: none;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
        transition: all 0.25s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.45);
    }

    /* =========================
       INPUT FIELDS
    ========================= */
    input {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.title("🧠 AI Mother & Baby Risk Prediction System")
st.markdown("Clinical decision support using multiple ML models")

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

importances = rf_model.named_steps["model"].feature_importances_
feature_importance = pd.Series(importances, index=feature_names)
top_features = feature_importance.sort_values(ascending=False).head(7).index.tolist()

# =========================
# MODE
# =========================
mode = st.radio("Select Analysis Mode", ["⚡ Quick Analysis", "📊 Detailed Analysis"])

st.markdown("---")

if mode == "⚡ Quick Analysis":
    selected_features = top_features
    st.info(f"Using key clinical indicators: {selected_features}")
else:
    selected_features = feature_names
    st.warning("Full diagnostic feature set enabled")

# =========================
# INPUT
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
if st.button("🚀 Run Risk Prediction"):

    results = {}

    for name, model in models.items():
        results[name] = model.predict(input_df)[0]

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
    show("Random Forest", results["Random Forest"], col2)
    show("SVM", results["SVM"], col3)

    st.markdown("---")

    st.subheader("📋 Summary")
    df_results = pd.DataFrame([results])
    st.dataframe(df_results, use_container_width=True)

    if len(set(results.values())) == 1:
        st.success("All models agree on prediction")
    else:
        st.warning("Model disagreement detected")

    st.bar_chart(df_results.T)

    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Feature Importance")
        st.write(feature_importance.sort_values(ascending=False))

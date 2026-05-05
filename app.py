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
# CLEAN GLASS UI + BACKGROUND FIX
# =========================
st.markdown(
    """
    <style>

    /* Background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1584515933487-779824d29309");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Dark overlay for contrast */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: rgba(0, 0, 0, 0.45);
        z-index: 0;
    }

    /* Main container */
    .main {
        position: relative;
        z-index: 1;
    }

    /* Glass card */
    .block-container {
        background: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    /* Titles */
    h1, h2, h3 {
        color: #1f2a44 !important;
        font-family: "Arial", sans-serif;
    }

    /* Text */
    p, label, div {
        color: #2c3e50 !important;
        font-size: 15px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }

    .stButton>button:hover {
        transform: scale(1.02);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.title("🧠 AI Mother & Baby Risk Prediction System")
st.markdown("Multi-model clinical decision support dashboard with AI comparison")

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
# INPUT FEATURES
# =========================
if mode == "⚡ Quick Analysis":
    selected_features = top_features
    st.info(f"Using important features: {selected_features}")
else:
    selected_features = feature_names
    st.warning("Using full clinical dataset")

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

    def show(name, val, col):
        with col:
            if val == 1:
                st.success(f"{name}\n🟢 Normal")
            elif val == 2:
                st.warning(f"{name}\n🟡 Suspicious")
            else:
                st.error(f"{name}\n🔴 Pathological")

    show("Logistic Regression", results["Logistic Regression"], col1)
    show("Random Forest", results["Random Forest"], col2)
    show("SVM", results["SVM"], col3)

    st.markdown("---")

    # =========================
    # SUMMARY TABLE
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
        st.warning("⚠ Model disagreement detected")

    st.markdown("---")

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Model Comparison")

    st.bar_chart(df_results.T)

    # =========================
    # FEATURE INSIGHT (ONLY QUICK MODE)
    # =========================
    if mode == "⚡ Quick Analysis":
        st.subheader("🧠 Feature Importance (Top Factors)")
        st.write(feature_importance.sort_values(ascending=False))

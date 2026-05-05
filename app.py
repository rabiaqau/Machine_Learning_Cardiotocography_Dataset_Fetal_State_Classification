import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiotocography Fetal State Classifier",
    page_icon="🫀",
    layout="centered",
)

# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rf_best_model.pkl")

model = load_model()

# ── Feature metadata ───────────────────────────────────────────────────────────
# (name, min, max, default, help text)
FEATURES = [
    ("LB",       50,  200,  133.0,  "Baseline fetal heart rate (bpm)"),
    ("AC",        0,    1,    0.0,  "Accelerations per second"),
    ("FM",        0,  600,    0.0,  "Fetal movements per second"),
    ("UC",        0,    1,    0.0,  "Uterine contractions per second"),
    ("DL",        0,    1,    0.0,  "Light decelerations per second"),
    ("DS",        0,    1,    0.0,  "Severe decelerations per second"),
    ("DP",        0,    1,    0.0,  "Prolonged decelerations per second"),
    ("ASTV",      0,  100,   73.0,  "% time with abnormal short-term variability"),
    ("MSTV",      0,   10,    0.5,  "Mean short-term variability"),
    ("ALTV",      0,  100,    0.0,  "% time with abnormal long-term variability"),
    ("MLTV",      0,   50,    7.9,  "Mean long-term variability"),
    ("Width",     0,  180,   70.0,  "Width of FHR histogram"),
    ("Min",      50,  200,  122.0,  "Minimum FHR in histogram"),
    ("Max",      50,  300,  152.0,  "Maximum FHR in histogram"),
    ("Nmax",      0,   30,    3.0,  "Number of histogram peaks"),
    ("Nzeros",    0,   10,    0.0,  "Number of histogram zeros"),
    ("Mode",     50,  200,  150.0,  "Histogram mode"),
    ("Mean",     50,  200,  137.0,  "Histogram mean"),
    ("Median",   50,  200,  137.0,  "Histogram median"),
    ("Variance",  0,  300,   14.0,  "Histogram variance"),
    ("Tendency", -1,    1,    0.0,  "Histogram tendency  (−1 left / 0 symmetric / 1 right)"),
]

FEATURE_NAMES = [f[0] for f in FEATURES]

LABEL_MAP = {
    1: ("✅ Normal",      "green"),
    2: ("⚠️ Suspect",     "orange"),
    3: ("🚨 Pathological", "red"),
}

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🫀 Cardiotocography Fetal State Classifier")
st.markdown(
    "Enter the CTG signal features below and click **Predict** "
    "to classify the fetal state as **Normal**, **Suspect**, or **Pathological**."
)

st.divider()

# Split inputs into two columns for readability
col1, col2 = st.columns(2)
input_data = {}

for i, (name, lo, hi, default, help_text) in enumerate(FEATURES):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_data[name] = st.number_input(
            label=name,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=0.01,
            help=help_text,
        )

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # Build input DataFrame in the correct column order
    input_df = pd.DataFrame(
        [[input_data[f] for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
        dtype=float,
    )

    try:
        pred = model.predict(input_df)[0]
        label, color = LABEL_MAP.get(int(pred), (f"Unknown ({pred})", "gray"))

        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{label}</h2>",
            unsafe_allow_html=True,
        )

        # Show probability if the model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            prob_df = pd.DataFrame(
                {"Class": [LABEL_MAP.get(int(c), (str(c), ""))[0] for c in classes],
                 "Probability": [f"{p:.1%}" for p in proba]}
            )
            st.table(prob_df.set_index("Class"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Make sure **rf_best_model.pkl** was saved with the same version of "
            "scikit-learn that is listed in requirements.txt."
        )

st.divider()
st.caption(
    "Model: Random Forest · Dataset: UCI Cardiotocography (id=193) · "
    "Classes: 1 = Normal, 2 = Suspect, 3 = Pathological"
)

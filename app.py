import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------
# LOAD MODELS
# -------------------------
lr_model = joblib.load("linreg_best_model.pkl")
rf_model = joblib.load("rf_best_model.pkl")
svm_model = joblib.load("svm_best_model.pkl")

# -------------------------
# FEATURE NAMES (21)
# -------------------------
feature_names = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

# -------------------------
# UI
# -------------------------
st.title("Mother & Baby Risk Prediction")

st.write("Enter patient data:")

input_data = {}

# Simple layout (no columns for now → fewer UI bugs)
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data]).astype(np.float64)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):

    try:
        lr_pred = int(lr_model.predict(input_df)[0])
        rf_pred = int(rf_model.predict(input_df)[0])
        svm_pred = int(svm_model.predict(input_df)[0])

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------
    st.subheader("Results")

    def interpret(val):
        if val == 1:
            return "Normal 🟢"
        elif val == 2:
            return "Suspicious 🟡"
        else:
            return "Pathological 🔴"

    st.write(f"Logistic Regression: {interpret(lr_pred)}")
    st.write(f"Random Forest: {interpret(rf_pred)}")
    st.write(f"SVM: {interpret(svm_pred)}")

    # Summary table
    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "SVM"],
        "Prediction": [lr_pred, rf_pred, svm_pred]
    })

    st.dataframe(results_df)

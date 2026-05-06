import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fetal Health Classification", layout="wide")

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_assets():
    model = joblib.load('rf_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Model or Scaler files not found. Please ensure 'rf_best_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Data Input")

def user_input_features():
    # Adding common features from the CTG dataset
    lb = st.sidebar.slider('Baseline Fetal Heart Rate (LB)', 100, 160, 130)
    ac = st.sidebar.slider('Accelerations (AC)', 0.0, 0.02, 0.005, format="%.3f")
    fm = st.sidebar.slider('Fetal Movement (FM)', 0.0, 0.5, 0.01, format="%.3f")
    uc = st.sidebar.slider('Uterine Contractions (UC)', 0.0, 0.02, 0.005, format="%.3f")
    dl = st.sidebar.slider('Light Decelerations (DL)', 0.0, 0.02, 0.002, format="%.3f")
    ds = st.sidebar.slider('Severe Decelerations (DS)', 0.0, 0.005, 0.0, format="%.4f")
    dp = st.sidebar.slider('Prolongued Decelerations (DP)', 0.0, 0.005, 0.0, format="%.4f")
    astv = st.sidebar.slider('Abnormal Short Term Variability (ASTV)', 10, 90, 45)
    mstv = st.sidebar.slider('Mean Short Term Variability (MSTV)', 0.0, 10.0, 1.0)
    altv = st.sidebar.slider('Percentage of Time with Abnormal Long Term Variability (ALTV)', 0, 100, 10)
    
    # Create a dictionary of inputs (Ensure keys/order match your training features exactly)
    data = {
        'LB': lb, 'AC': ac, 'FM': fm, 'UC': uc, 'DL': dl, 'DS': ds, 'DP': dp,
        'ASTV': astv, 'MSTV': mstv, 'ALTV': altv
        # Note: Add all other features used during training here in the correct order
    }
    return pd.DataFrame(data, index=[0])

# --- MAIN PANEL ---
st.title("👶 Fetal Health Classification (CTG)")
st.write("""
This app predicts the **Fetal State (NSP)** based on Cardiotocogram data. 
- **Normal** (1.0)
- **Suspect** (2.0)
- **Pathologic** (3.0)
""")

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# --- PREDICTION ---
if st.button("Classify"):
    # Preprocess the input
    # Note: If your model used 21 features, you must provide all 21 here
    # For this example, we assume the model was trained on the columns in input_df
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Output Results
    st.subheader("Prediction")
    status = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathologic"}
    result = status.get(prediction[0], "Unknown")
    
    if prediction[0] == 1.0:
        st.success(f"The predicted state is: **{result}**")
    elif prediction[0] == 2.0:
        st.warning(f"The predicted state is: **{result}**")
    else:
        st.error(f"The predicted state is: **{result}**")

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=["Normal", "Suspect", "Pathologic"])
    st.table(prob_df)

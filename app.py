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
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Patient Data Input")

def user_input_features():
    # 1. Primary Metrics (Commonly used)
    st.sidebar.subheader("Primary Metrics")
    lb = st.sidebar.slider('Baseline Fetal Heart Rate (LB)', 106, 160, 133)
    ac = st.sidebar.slider('Accelerations (AC)', 0.0, 0.02, 0.003, format="%.3f")
    fm = st.sidebar.slider('Fetal Movement (FM)', 0.0, 0.5, 0.01, format="%.3f")
    uc = st.sidebar.slider('Uterine Contractions (UC)', 0.0, 0.02, 0.004, format="%.3f")
    astv = st.sidebar.slider('Abnormal Short Term Variability (ASTV)', 12, 87, 47)
    mstv = st.sidebar.slider('Mean Short Term Variability (MSTV)', 0.2, 7.0, 1.3)
    altv = st.sidebar.slider('Abnormal Long Term Variability (ALTV)', 0, 91, 10)
    mltv = st.sidebar.slider('Mean Long Term Variability (MLTV)', 0.0, 50.0, 8.0)

    # 2. Deceleration Metrics
    st.sidebar.subheader("Decelerations")
    dl = st.sidebar.slider('Light Decelerations (DL)', 0.0, 0.02, 0.002, format="%.3f")
    ds = st.sidebar.slider('Severe Decelerations (DS)', 0.0, 0.001, 0.0, format="%.4f")
    dp = st.sidebar.slider('Prolonged Decelerations (DP)', 0.0, 0.005, 0.0, format="%.4f")

    # 3. Histogram/Technical Features (Hidden in an expander to save space)
    with st.sidebar.expander("Histogram & Statistical Features"):
        width = st.slider('Histogram Width', 3, 180, 70)
        min_val = st.slider('Histogram Min', 50, 160, 93)
        max_val = st.slider('Histogram Max', 120, 240, 163)
        nmax = st.slider('Number of Peaks (Nmax)', 0, 18, 4)
        nzeros = st.slider('Number of Zeros (Nzeros)', 0, 10, 0)
        mode = st.slider('Histogram Mode', 60, 190, 137)
        mean = st.slider('Histogram Mean', 73, 180, 134)
        median = st.slider('Histogram Median', 77, 186, 138)
        variance = st.slider('Histogram Variance', 0, 270, 18)
        tendency = st.selectbox('Histogram Tendency', [-1, 0, 1], index=1)

    # CREATE DICTIONARY (MUST MATCH THE 21 COLUMNS EXACTLY)
    data = {
        'LB': lb, 'AC': ac, 'FM': fm, 'UC': uc, 'DL': dl, 'DS': ds, 'DP': dp,
        'ASTV': astv, 'MSTV': mstv, 'ALTV': altv, 'MLTV': mltv,
        'Width': width, 'Min': min_val, 'Max': max_val, 'Nmax': nmax,
        'Nzeros': nzeros, 'Mode': mode, 'Mean': mean, 'Median': median,
        'Variance': variance, 'Tendency': tendency
    }
    
    # Return as DataFrame with columns in the exact order requested
    feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 
                     'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 
                     'Median', 'Variance', 'Tendency']
    
    df = pd.DataFrame(data, index=[0])
    return df[feature_names]

# --- MAIN UI ---
st.title("👶 Fetal Health Classification (CTG)")
st.markdown("""
Predicting the fetal state from Cardiotocography. Adjust parameters in the sidebar and click **Classify**.
""")

input_df = user_input_features()

# Display input summary
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Summary")
    st.dataframe(input_df)

# --- PREDICTION LOGIC ---
with col2:
    st.subheader("Action")
    if st.button("Run Classification", use_container_width=True):
        # 1. Scale Input
        try:
            scaled_input = scaler.transform(input_df)
            
            # 2. Predict
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]

            # 3. Visual Results
            status_map = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
            result = status_map.get(prediction, "Unknown")
            
            st.write(f"### Result: **{result}**")
            
            if prediction == 1:
                st.success("The condition is likely Normal.")
            elif prediction == 2:
                st.warning("The condition is Suspect.")
            else:
                st.error("The condition is Pathologic.")

            # Probability Table
            prob_df = pd.DataFrame({
                "State": ["Normal", "Suspect", "Pathologic"],
                "Confidence (%)": [round(p * 100, 2) for p in prediction_proba]
            })
            st.table(prob_df)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.divider()
st.info("Note: This tool is for educational purposes and is based on the UCI Cardiotocography Dataset.")

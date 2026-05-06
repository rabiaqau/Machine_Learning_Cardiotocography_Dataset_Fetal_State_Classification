import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np

# --- 1. HELPER FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    try:
        bin_str = get_base64_of_bin_file(bin_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Full-width Banner for Title - Matches Screenshot 15.23.15 */
        .banner {{
            background-color: rgba(160, 120, 50, 0.85); 
            width: 100%;
            padding: 25px 0;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .banner h1 {{
            color: white !important;
            font-family: 'Arial Black', Gadget, sans-serif;
            font-size: 2.2rem;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        /* Transparent container for content readability */
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.9); 
            backdrop-filter: blur(8px);
            padding: 40px;
            border-radius: 20px;
            margin-top: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }}

        /* Centered Square Button - Matches Screenshot 15.23.15 */
        div.stButton > button {{
            background-color: #a07832 !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            padding: 15px 50px !important;
            border-radius: 4px !important;
            border: 2px solid #8a6628 !important;
            display: block;
            margin: 0 auto;
            text-transform: uppercase;
            transition: 0.3s;
        }}

        div.stButton > button:hover {{
            background-color: #8a6628 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            transform: translateY(-2px);
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(248, 249, 250, 0.98);
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.sidebar.warning("Background image not found.")

# --- 2. INITIALIZE ASSETS ---
set_png_as_page_bg('mother_baby_image.png')

@st.cache_resource
def load_assets():
    model = joblib.load('rf_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 3. SIDEBAR FEATURES ---
def user_input_features():
    st.sidebar.header("📊 Clinical Parameters")
    
    # 21 Required Features
    lb = st.sidebar.slider('Fetal Heart Rate Baseline (LB)', 100, 165, 133)
    ac = st.sidebar.slider('Accelerations (AC)', 0.0, 0.02, 0.003, format="%.3f")
    fm = st.sidebar.slider('Fetal Movement (FM)', 0.0, 0.5, 0.01, format="%.3f")
    uc = st.sidebar.slider('Uterine Contractions (UC)', 0.0, 0.02, 0.004, format="%.3f")
    dl = st.sidebar.slider('Light Decelerations (DL)', 0.0, 0.02, 0.002, format="%.3f")
    ds = st.sidebar.slider('Severe Decelerations (DS)', 0.0, 0.001, 0.0, format="%.4f")
    dp = st.sidebar.slider('Prolonged Decelerations (DP)', 0.0, 0.005, 0.0, format="%.4f")
    astv = st.sidebar.slider('Abnormal Short Term Variability (ASTV %)', 10, 90, 47)
    mstv = st.sidebar.slider('Mean Short Term Variability (MSTV)', 0.2, 7.0, 1.3)
    altv = st.sidebar.slider('Abnormal Long Term Variability (ALTV %)', 0, 95, 10)
    mltv = st.sidebar.slider('Mean Long Term Variability (MLTV)', 0.0, 50.0, 8.0)
    width = st.sidebar.slider('Histogram Width', 3, 180, 70)
    min_v = st.sidebar.slider('Histogram Min', 50, 160, 93)
    max_v = st.sidebar.slider('Histogram Max', 120, 240, 163)
    nmax = st.sidebar.number_input('Number of Peaks', value=4)
    nzeros = st.sidebar.number_input('Number of Zeros', value=0)
    mode = st.sidebar.slider('Histogram Mode', 60, 190, 137)
    mean = st.sidebar.slider('Histogram Mean', 73, 180, 134)
    median = st.sidebar.slider('Histogram Median', 77, 186, 138)
    variance = st.sidebar.slider('Histogram Variance', 0, 270, 18)
    tendency = st.sidebar.selectbox('Histogram Tendency', [-1, 0, 1], index=1)

    data = {
        'LB': lb, 'AC': ac, 'FM': fm, 'UC': uc, 'DL': dl, 'DS': ds, 'DP': dp,
        'ASTV': astv, 'MSTV': mstv, 'ALTV': altv, 'MLTV': mltv,
        'Width': width, 'Min': min_v, 'Max': max_v, 'Nmax': nmax,
        'Nzeros': nzeros, 'Mode': mode, 'Mean': mean, 'Median': median,
        'Variance': variance, 'Tendency': tendency
    }
    
    feature_names = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 
                     'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 
                     'Median', 'Variance', 'Tendency']
    
    return pd.DataFrame(data, index=[0])[feature_names]

input_df = user_input_features()

# --- 4. MAIN UI CONTENT ---
st.markdown("""
    <div class="banner">
        <h1>🛡️ FETAL HEALTH SUPPORT SYSTEM</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #555; font-style: italic; margin-bottom: 25px;'>Advanced Diagnostic Analytics Portal</p>", unsafe_allow_html=True)

# 5. DIAGNOSTIC ACTION
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("GENERATE ASSESSMENT"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        
        status_map = {1: "NORMAL", 2: "SUSPECT", 3: "PATHOLOGIC"}
        result = status_map.get(prediction)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.success(f"### DIAGNOSIS: {result}")
        elif prediction == 2:
            st.warning(f"### DIAGNOSIS: {result}")
        else:
            st.error(f"### DIAGNOSIS: {result}")

# 6. DATA OVERVIEW
st.divider()
st.subheader("📋 Current Input Vector")
st.dataframe(input_df, use_container_width=True)

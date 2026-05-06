import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
import matplotlib.pyplot as plt

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
        
        /* Full-width Banner Style */
        .banner {{
            background-color: rgba(160, 120, 50, 0.9); 
            width: 100%;
            padding: 20px 0;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .banner h1, .banner h2 {{
            color: white !important;
            font-family: 'Arial Black', Gadget, sans-serif;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        /* Results Box - Solid White to prevent mixing with background */
        .results-container {{
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            border: 3px solid #a07832;
            margin-top: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}

        /* Main Content Glass-morphism Area */
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.85); 
            backdrop-filter: blur(5px);
            padding: 40px;
            border-radius: 20px;
            margin-top: 20px;
        }}

        /* Centered Styled Button */
        div.stButton > button {{
            background-color: #a07832 !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            padding: 15px 50px !important;
            border-radius: 4px !important;
            display: block;
            margin: 0 auto;
            text-transform: uppercase;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(248, 249, 250, 0.98);
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

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
    # Features remain same for model compatibility
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
st.markdown('<div class="banner"><h1>🛡️ FETAL HEALTH SUPPORT SYSTEM</h1></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("GENERATE ASSESSMENT"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Results wrapped in a White Box
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        status_map = {1: "NORMAL", 2: "SUSPECT", 3: "PATHOLOGIC"}
        result = status_map.get(prediction)
        
        if prediction == 1:
            st.success(f"### DIAGNOSIS: {result}")
        elif prediction == 2:
            st.warning(f"### DIAGNOSIS: {result}")
        else:
            st.error(f"### DIAGNOSIS: {result}")
            
        # 5. PROBABILITY HISTOGRAM
        st.write("#### Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        classes = ['Normal', 'Suspect', 'Pathologic']
        colors = ['#28a745', '#ffc107', '#dc3545'] # Green, Yellow, Red
        
        ax.bar(classes, probabilities, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- 6. DATA OVERVIEW ---
st.markdown('<br><div class="banner"><h2>📋 CURRENT INPUT VECTOR</h2></div>', unsafe_allow_html=True)
st.dataframe(input_df, use_container_width=True)

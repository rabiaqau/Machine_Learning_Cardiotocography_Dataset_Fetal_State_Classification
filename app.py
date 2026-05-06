import streamlit as st
import pandas as pd
import joblib
import base64

# 1. HELPER FUNCTIONS (Must be at the top)
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
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.94); 
            backdrop-filter: blur(12px);
            padding: 40px;
            border-radius: 20px;
            margin-top: 50px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(248, 249, 250, 0.98);
        }}
        .main-title {{ color: #1e3d59; text-align: center; font-weight: 800; font-size: 2.5rem; }}
        .sub-title {{ color: #e4572e; text-align: center; font-size: 1.2rem; font-weight: 500; margin-bottom: 30px; }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# 2. INITIALIZE BACKGROUND
set_png_as_page_bg('mother_baby_image.png')

# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('rf_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 4. SIDEBAR INPUT FUNCTION (This is what generates the features)
def user_input_features():
    st.sidebar.header("📊 Clinical Parameters")
    
    # We use sliders here to define the 21 features required
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

# 5. CALL THE INPUT FUNCTION (Crucial step!)
input_df = user_input_features()

# 6. MAIN CONTENT UI
st.markdown('<h1 class="main-title">🛡️ Fetal Health Clinical Support System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Diagnostic Analytics Portal</p>', unsafe_allow_html=True)

st.info("The observed metrics are evaluated against a Random Forest model trained on the UCI Cardiotocography dataset.")

# Display diagnosis
if st.button("Run Diagnostic Assessment", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    status_map = {1: "NORMAL", 2: "SUSPECT", 3: "PATHOLOGIC"}
    result = status_map.get(prediction)
    
    if prediction == 1:
        st.success(f"### CLASSIFICATION: {result}")
    elif prediction == 2:
        st.warning(f"### CLASSIFICATION: {result}")
    else:
        st.error(f"### CLASSIFICATION: {result}")

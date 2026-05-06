import streamlit as st
import pandas as pd
import joblib
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="CTG Clinical Analytics", layout="wide")

# --- BACKGROUND IMAGE HELPER ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    
    /* Adding a semi-transparent overlay for readability */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    
    .main .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }}
    
    h1, h2, h3 {{
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Try to set the background
try:
    set_png_as_page_bg('mother_baby_image.png')
except FileNotFoundError:
    st.sidebar.warning("Background image 'mother_baby_image.png' not found. Using default background.")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('rf_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- HEADER SECTION ---
st.title("🛡️ Fetal Health Clinical Support System")
st.markdown("""
**Diagnostic Analytics Portal**  
This system utilizes advanced Machine Learning to evaluate Cardiotocogram (CTG) metrics, 
providing real-time classification of fetal physiological states.
""")
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Clinical Parameters")

def user_input_features():
    with st.sidebar.expander("Vital Signs & Variability", expanded=True):
        lb = st.slider('Fetal Heart Rate Baseline (BPM)', 100, 165, 133)
        ac = st.slider('Accelerations', 0.0, 0.02, 0.003, format="%.3f")
        astv = st.slider('Abnormal Short Term Variability (%)', 10, 90, 47)
        mstv = st.slider('Mean Short Term Variability', 0.2, 7.0, 1.3)
        altv = st.slider('Abnormal Long Term Variability (%)', 0, 95, 10)

    with st.sidebar.expander("Uterine & Deceleration Data"):
        uc = st.slider('Uterine Contractions', 0.0, 0.02, 0.004, format="%.3f")
        dl = st.slider('Light Decelerations', 0.0, 0.02, 0.002, format="%.3f")
        ds = st.slider('Severe Decelerations', 0.0, 0.001, 0.0, format="%.4f")
        dp = st.slider('Prolonged Decelerations', 0.0, 0.005, 0.0, format="%.4f")
        fm = st.slider('Fetal Movement', 0.0, 0.5, 0.01, format="%.3f")

    with st.sidebar.expander("Histogram Morphology"):
        # Keeping internal names for the model but user-friendly labels
        width = st.slider('Width of Histogram', 3, 180, 70)
        min_v = st.slider('Minimum Frequency', 50, 160, 93)
        max_v = st.slider('Maximum Frequency', 120, 240, 163)
        mode = st.slider('Histogram Mode', 60, 190, 137)
        mean = st.slider('Histogram Mean', 73, 180, 134)
        median = st.slider('Histogram Median', 77, 186, 138)
        variance = st.slider('Histogram Variance', 0, 270, 18)
        tendency = st.selectbox('Histogram Tendency', [-1, 0, 1], index=1)
        mltv = st.slider('Mean Long Term Var.', 0.0, 50.0, 8.0)
        nmax = st.number_input('Number of Peaks', value=4)
        nzeros = st.number_input('Number of Zeros', value=0)

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

# --- ANALYSIS SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Parameter Matrix")
    st.write("Current patient parameters selected for evaluation:")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), height=400)

with col2:
    st.subheader("🎯 Diagnostic Inference")
    if st.button("Generate Assessment", use_container_width=True):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probs = model.predict_proba(scaled_input)[0]

        # Professional Outcome Formatting
        status_map = {1: "NORMAL", 2: "SUSPECT", 3: "PATHOLOGIC"}
        result_text = status_map.get(prediction)
        
        if prediction == 1:
            st.success(f"### CLASSIFICATION: {result_text}")
            st.info("The observed patterns indicate standard physiological parameters.")
        elif prediction == 2:
            st.warning(f"### CLASSIFICATION: {result_text}")
            st.write("⚠️ *Clinical correlation and further monitoring are advised.*")
        else:
            st.error(f"### CLASSIFICATION: {result_text}")
            st.write("🚨 *Immediate clinical review recommended.*")

        # Probability Visualization
        st.write("#### Confidence Interval")
        prob_data = pd.DataFrame({
            "Classification": ["Normal", "Suspect", "Pathologic"],
            "Probability": probs
        })
        st.bar_chart(prob_data.set_index("Classification"))

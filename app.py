import streamlit as st
import pandas as pd
import joblib
import base64

# --- 1. DEFINE HELPER FIRST ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 2. DEFINE BACKGROUND FUNCTION SECOND ---
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

        .main-title {{
            color: #1e3d59;
            text-align: center;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 5px;
        }}
        
        .sub-title {{
            color: #e4572e;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 30px;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.sidebar.error("⚠️ Background image not found.")

# --- 3. CALL THE FUNCTION ---
set_png_as_page_bg('mother_baby_image.png')

# --- 4. REST OF YOUR APP (HEADER, MODEL LOADING, ETC) ---
st.markdown('<h1 class="main-title">🛡️ Fetal Health Clinical Support System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Diagnostic Analytics Portal</p>', unsafe_allow_html=True)

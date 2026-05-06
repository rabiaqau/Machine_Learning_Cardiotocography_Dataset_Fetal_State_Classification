def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* This targets the main content area */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.92); /* Higher opacity for better contrast */
        backdrop-filter: blur(10px); /* Blurs the image behind the container */
        padding: 40px;
        border-radius: 15px;
        margin-top: 50px;
        margin-bottom: 50px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }}

    /* Make sidebar clean */
    [data-testid="stSidebar"] {{
        background-color: rgba(248, 249, 250, 0.95);
    }}

    /* Professional header styling */
    .main-title {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e3d59;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }}
    
    .sub-title {{
        color: #ff6e40;
        text-align: center;
        font-style: italic;
        margin-bottom: 30px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Try to set background
try:
    set_png_as_page_bg('mother_baby_image.png')
except FileNotFoundError:
    pass

# --- UPDATED HEADER SECTION ---
st.markdown('<h1 class="main-title">🛡️ Fetal Health Clinical Support System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Diagnostic Analytics Portal</p>', unsafe_allow_html=True)

st.info("""
**Clinical Overview:** This system evaluates Cardiotocogram (CTG) metrics to provide 
real-time classification of fetal physiological states using a trained Random Forest architecture.
""")

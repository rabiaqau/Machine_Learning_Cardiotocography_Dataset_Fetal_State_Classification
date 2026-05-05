import streamlit as st
import joblib

st.title("Test App")

st.write("Starting...")

try:
    model = joblib.load("rf_best_model.pkl")
    st.success("RF model loaded")
except Exception as e:
    st.error("Error loading RF model")
    st.code(str(e))

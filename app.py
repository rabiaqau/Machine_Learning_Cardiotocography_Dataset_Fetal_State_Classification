
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("/Users/rabiashaheen/Desktop/ml_project/Machine_Learning_Cardiotocography_Dataset_Fetal_State_Classification/rf_best_model.pkl")

feature_names = [
    'LB','AC','FM','UC','DL','DS','DP',
    'ASTV','MSTV','ALTV','MLTV',
    'Width','Min','Max','Nmax','Nzeros',
    'Mode','Mean','Median','Variance','Tendency'
]

st.title("Cardiotocography Prediction")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])[feature_names]

if st.button("Predict"):
    pred = model.predict(input_df)
    st.success(f"Prediction: {pred[0]}")
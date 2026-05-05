
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("rf_best_model.pkl")

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


input_df = pd.DataFrame([[input_data[f] for f in feature_names]],
                        columns=feature_names)
if st.button("Predict"):
    pred = model.predict(input_df)
    st.success(f"Prediction: {pred[0]}")

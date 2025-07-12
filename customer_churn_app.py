# app.py
# Streamlit app for Customer Churn Prediction using pre-trained RandomForest model

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('customer_churn_model.pkl')

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Upload customer data or manually input values to predict churn using a pre-trained Random Forest model.")

# Collect user input
def user_input_features():
    st.subheader("Input Customer Features")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)
    SeniorCitizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Partner = st.selectbox("Has Partner", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Dependents = st.selectbox("Has Dependents", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    PhoneService = st.selectbox("Phone Service", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    MultipleLines = st.selectbox("Multiple Lines", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    InternetService = st.selectbox("Internet Service (0-DSL, 1-Fiber Optic, 2-No)", options=[0, 1, 2])
    OnlineSecurity = st.selectbox("Online Security", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    OnlineBackup = st.selectbox("Online Backup", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    DeviceProtection = st.selectbox("Device Protection", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    TechSupport = st.selectbox("Tech Support", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    StreamingTV = st.selectbox("Streaming TV", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    StreamingMovies = st.selectbox("Streaming Movies", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Contract = st.selectbox("Contract Type (0-Month-to-month, 1-One year, 2-Two year)", options=[0, 1, 2])
    PaperlessBilling = st.selectbox("Paperless Billing", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    PaymentMethod = st.selectbox("Payment Method (0-Bank, 1-Credit Card, 2-Electronic Check, 3-Mailed Check)", options=[0, 1, 2, 3])

    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Load user input
input_df = user_input_features()

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn with probability {prediction_proba:.2f}.")
    else:
        st.success(f"✅ The customer is not likely to churn with probability {1 - prediction_proba:.2f}.")

# Display raw input if desired
if st.checkbox("Show Input Data"):
    st.write(input_df)

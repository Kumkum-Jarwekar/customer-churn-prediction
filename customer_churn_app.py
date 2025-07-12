# customer_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ----------------------------
# Load and preprocess dataset
# ----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('Telco-Customer-Churn_dataset.csv')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(inplace=True)
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'customerID':
            data[column] = le.fit_transform(data[column])
    return data

data = load_data()

# ----------------------------
# Train model for initial deployment
# ----------------------------
@st.cache_resource
def train_model(data):
    X = data.drop(['customerID', 'Churn'], axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# ----------------------------
# Streamlit App UI
# ----------------------------

st.title("📈 Customer Churn Prediction App")
st.markdown("Upload customer data or enter manually to predict churn probability.")

# Sidebar filters for business owner
st.sidebar.header("📊 Filter Data (optional)")
contract_type = st.sidebar.selectbox("Contract Type", options=["All", "Month-to-month", "One year", "Two year"])
tenure_slider = st.sidebar.slider("Minimum Tenure (months)", 0, 72, 0)
monthly_charges_slider = st.sidebar.slider("Max Monthly Charges", 0, 150, 150)

# Filter data for insights
filtered_data = data.copy()
if contract_type != "All":
    contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    contract_rev_map = {v: k for k, v in contract_map.items()}
    filtered_data = filtered_data[filtered_data['Contract'] == contract_rev_map[contract_type]]
filtered_data = filtered_data[filtered_data['tenure'] >= tenure_slider]
filtered_data = filtered_data[filtered_data['MonthlyCharges'] <= monthly_charges_slider]

st.subheader("Filtered Data Preview")
st.dataframe(filtered_data.head())

# ----------------------------
# Upload CSV or Manual Input
# ----------------------------
option = st.radio("Choose input method:", ("Upload CSV", "Manual Entry"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_data.head())
        if st.button("Predict Churn for Uploaded Data"):
            # Preprocess uploaded data
            le = LabelEncoder()
            for column in input_data.select_dtypes(include=['object']).columns:
                if column != 'customerID':
                    input_data[column] = le.fit_transform(input_data[column])
            input_data.fillna(0, inplace=True)
            X_input = input_data.drop(['customerID'], axis=1)
            churn_pred = model.predict(X_input)
            churn_prob = model.predict_proba(X_input)[:, 1]
            input_data['Churn_Prediction'] = churn_pred
            input_data['Churn_Probability'] = churn_prob
            st.write("### Prediction Results:")
            st.dataframe(input_data[['customerID', 'Churn_Prediction', 'Churn_Probability']])

elif option == "Manual Entry":
    st.write("### Enter Customer Details Manually")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
    total_charges = st.slider("Total Charges", 0, 9000, 1000)

    if st.button("Predict Churn"):
        input_dict = {
            'gender': 0 if gender == "Female" else 1,
            'SeniorCitizen': 0 if senior_citizen == "No" else 1,
            'Partner': 0 if partner == "No" else 1,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'PhoneService': 1,
            'MultipleLines': 0,
            'InternetService': 1,
            'OnlineSecurity': 0,
            'OnlineBackup': 0,
            'DeviceProtection': 0,
            'TechSupport': 0,
            'StreamingTV': 0,
            'StreamingMovies': 0,
            'Contract': 0,
            'PaperlessBilling': 1,
            'PaymentMethod': 0,
            'Dependents': 0
        }
        input_df = pd.DataFrame([input_dict])
        churn_pred = model.predict(input_df)[0]
        churn_prob = model.predict_proba(input_df)[0][1]

        st.write(f"### Churn Probability: {churn_prob:.2f}")
        if churn_pred == 1:
            st.error("❌ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")

        # Display visual risk bar
        fig, ax = plt.subplots(figsize=(5, 0.5))
        sns.heatmap(np.array([churn_prob]), cmap="Reds", cbar=False, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.info("App developed as part of Data Science project for Customer Churn Prediction.")

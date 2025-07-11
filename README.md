# Customer Churn Prediction

## 📌 Project Overview

This project predicts customer churn for a telecom company using machine learning, allowing businesses to identify customers likely to leave and take proactive retention actions to improve revenue and reduce churn.

## 📊 Dataset

* **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
* **Records:** 7,032 customer records
* **Features:** Customer demographics, account information, and service details
* **Target:** `Churn` (Yes/No)

## 🛠️ Project Workflow

### 1️⃣ Data Loading & Cleaning

* Loaded the CSV dataset using Pandas
* Handled missing values in `TotalCharges`
* Encoded categorical variables using Label Encoding

### 2️⃣ Exploratory Data Analysis (EDA)

* Churn distribution analysis
* Correlation heatmap of numerical features
* Distribution of tenure and charges among churned customers

### 3️⃣ Model Training

* Split the data into training and testing sets
* Trained a Logistic Regression model to predict churn

### 4️⃣ Model Evaluation

* Printed classification report with precision, recall, and F1-score
* Calculated Accuracy Score and ROC-AUC Score
* Displayed:

  * Confusion Matrix
  * ROC Curve
  * Feature Importance

## 📈 Results

* The Logistic Regression model achieved good predictive performance on the test set.
* Top features influencing churn included:

  * Contract type
  * Tenure
  * Monthly charges
* This insight allows the business to design targeted retention offers.

## 🚀 How to Run Locally

1️⃣ Clone the repository:

```bash
git clone https://github.com/Kumkum-Jarwekar/customer-churn-prediction.git
cd customer-churn-prediction
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run the script:

```bash
python customer_churn.py
```

✅ Plots will appear one by one; close each plot window to proceed.
✅ Evaluation metrics will display in your terminal upon completion.

## 🌱 Future Improvements

* Try other models (Random Forest, XGBoost, LightGBM)
* Hyperparameter tuning for improved performance
* Deploy a Streamlit web app for interactive churn prediction (Phase 2)
* Add SHAP or LIME for advanced interpretability

## 🤝 License

This project is for educational purposes. Dataset sourced from Kaggle, retaining respective license.

## 🙏 Acknowledgments

* [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
* Inspiration from the telecom industry's need for customer retention strategies.

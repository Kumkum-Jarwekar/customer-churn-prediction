# 1️⃣ Import Libraries
# -----------------------------
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# -----------------------------
# 2️⃣ Load Dataset
# -----------------------------
data = pd.read_csv(r'D:\DS\data science projects\Customer Churn Prediction\Telco-Customer-Churn_dataset.csv') 
print(data.head())

# -----------------------------
# 3️⃣ Data Cleaning & Preprocessing
# -----------------------------
# Drop customerID (not useful for prediction)
data.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric, coerce errors to NaN, and drop missing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Encode categorical columns using LabelEncoder
cat_cols = data.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# -----------------------------
# 4️⃣ Exploratory Data Analysis (EDA)
# -----------------------------
print(data.describe())
print(data['Churn'].value_counts())

# Visualizing churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# -----------------------------
# 5️⃣ Feature and Target Split
# -----------------------------
X = data.drop('Churn', axis=1)
y = data['Churn']

# -----------------------------
# 6️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# 7️⃣ Model Building (Random Forest)
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 8️⃣ Evaluation
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# -----------------------------
# 9️⃣ Feature Importance
# -----------------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.show()

# -----------------------------
# 1️⃣0️⃣ Conclusion
# -----------------------------
print("The model has been successfully trained and evaluated.")
print("You can now use this script to predict churn and interpret feature importance to guide business decisions.")

joblib.dump(model, 'customer_churn_model.pkl')
print("Model saved as 'customer_churn_model.pkl' using joblib.")
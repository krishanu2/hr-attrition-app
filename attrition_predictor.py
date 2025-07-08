# attrition_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import ExcelWriter
import os

print("🔄 Loading dataset...")

# -------------------- LOAD DATA --------------------
file_path = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)

# -------------------- CHECK MISSING --------------------
missing = df.isnull().sum()
print("\nMissing Values:\n", missing[missing > 0] if missing.sum() > 0 else missing)

# -------------------- ENCODE TARGET --------------------
df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# -------------------- DROP UNNECESSARY COLUMNS --------------------
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# -------------------- ENCODE CATEGORICAL --------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------- SPLIT DATA --------------------
X = df.drop(['Attrition', 'AttritionFlag'], axis=1)
y = df['AttritionFlag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- SCALE FEATURES --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- TRAIN MODEL --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- EVALUATE --------------------
y_pred = model.predict(X_test)
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------- CREATE PREDICTION OUTPUT --------------------
pred_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
pred_df.reset_index(drop=True, inplace=True)

# -------------------- ADD RISK SCORE --------------------
risk_scores = model.predict_proba(X_test)[:, 1]
pred_df['Attrition Risk Score'] = (risk_scores * 100).round(2)

# Add back department and job role for HR
X_test_df = pd.DataFrame(X_test, columns=X.columns)
original_info = df[['Department', 'JobRole']].iloc[y_test.index].reset_index(drop=True)
pred_df = pd.concat([original_info, pred_df], axis=1)

# -------------------- ATTRITION SUMMARY TABLES --------------------
attrition_by_dept = df.groupby('Department')['AttritionFlag'].mean().sort_values(ascending=False) * 100
attrition_by_role = df.groupby('JobRole')['AttritionFlag'].mean().sort_values(ascending=False) * 100

# -------------------- HIGH RISK EMPLOYEES --------------------
high_risk_df = pred_df[pred_df['Attrition Risk Score'] > 70]

# -------------------- EXPORT MULTI-SHEET EXCEL --------------------
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'attrition_report.xlsx')

with ExcelWriter(output_path, engine='openpyxl') as writer:
    pred_df.to_excel(writer, sheet_name='Predictions', index=False)
    high_risk_df.to_excel(writer, sheet_name='High Risk', index=False)
    attrition_by_dept.to_frame(name='Attrition %').to_excel(writer, sheet_name='By Department')
    attrition_by_role.to_frame(name='Attrition %').to_excel(writer, sheet_name='By Job Role')

print(f"\n✅ Full HR report exported: {output_path}")

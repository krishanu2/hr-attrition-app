import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import base64

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

raw_data = load_data()

# Preprocess data
def preprocess_data(df):
    df = df.copy()
    df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

processed_df = preprocess_data(raw_data)
X = processed_df.drop("Attrition", axis=1)
y = processed_df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# SHAP explainer
X_train_numeric = X_train.astype(np.float64)
explainer = shap.Explainer(model, X_train_numeric)
shap_values = explainer(X_train_numeric)

# UI: App Title
st.title("🧠 HR Attrition Prediction App")
st.markdown("Predict whether an employee is likely to leave the company using ML and interpret the prediction.")

# Section 1: Model performance
st.subheader("📊 Model Performance")
report = classification_report(y_test, model.predict(X_test), output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}'}))

# Section 2: Global Feature Importance
st.subheader("🌍 SHAP Global Feature Importance")
fig, ax = plt.subplots()
shap.plots.bar(shap_values, max_display=10, show=False)
st.pyplot(fig)

# Section 3: Predict Individual Example
st.subheader("🔍 Individual Prediction")
index = st.slider("Select Test Sample", 0, len(X_test) - 1, 0)
sample = X_test.iloc[[index]]
sample_display = raw_data.iloc[X_test.index[index]]

pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0][pred]
st.write(f"🔮 Prediction: **{'Yes' if pred == 1 else 'No'}** (Attrition)")
st.write(f"🔢 Confidence: `{proba*100:.2f}%`")

# SHAP Force Plot for individual prediction
st.write("📌 **Explanation of Prediction**")
shap_sample = explainer(sample.astype(np.float64))
fig2, ax2 = plt.subplots()
shap.plots.waterfall(shap_sample[0], max_display=10, show=False)
st.pyplot(fig2)

# Section 4: Export predictions
def create_download_link(df):
    df.to_csv("output/output_attrition_report.csv", index=False)
    with open("output/output_attrition_report.csv", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="attrition_report.csv">📥 Download Report</a>'
    return href

st.subheader("📝 Export Report")
st.markdown(create_download_link(processed_df), unsafe_allow_html=True)

# Section 5: Help & Info
with st.expander("ℹ️ Help & Info"):
    st.info("This tool uses **XGBoost + SHAP** to predict employee attrition. "
            "You can explore global importance and individual explanations. "
            "**Precision**: How many predicted leavers actually left. "
            "**Recall**: How many actual leavers were caught by the model.")

st.markdown("---")
st.caption("Developed with ❤️ by Krishanu Mahapatra")

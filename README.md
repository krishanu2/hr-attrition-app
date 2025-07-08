\# 🔍 HR Attrition Prediction App



This is a powerful and interactive \*\*Streamlit web application\*\* that helps HR professionals predict employee attrition using machine learning, particularly the \*\*XGBoost\*\* classifier. The app also provides explanations of predictions using \*\*SHAP values\*\*, making the results interpretable and actionable.



---



\## 🧠 What It Does



\- ✅ Predicts whether an employee is likely to \*\*leave (attrition)\*\* or \*\*stay\*\*.

\- 📊 Displays \*\*SHAP feature importance\*\* globally and per individual.

\- 📈 Shows \*\*model performance\*\* via classification report.

\- 💾 Allows users to \*\*download prediction reports\*\* in Excel.

\- 🧭 User-friendly interface with guided help sections.



---



\## 🏗️ Tech Stack



| Tool        | Usage                         |

|-------------|-------------------------------|

| Python      | Core programming              |

| Streamlit   | Web app framework             |

| XGBoost     | Machine learning model        |

| SHAP        | Explainable AI (XAI)          |

| Pandas      | Data manipulation             |

| Matplotlib  | Visualizations                |



---



\## 📁 Project Structure



```

Project5/

│

├── data/

│   └── WA\\\_Fn-UseC\\\_-HR-Employee-Attrition.csv

│

├── output/

│   └── output\\\_attrition\\\_report.xlsx

│

├── hr\\\_attrition\\\_app.py         # Main Streamlit app

├── attrition\\\_predictor.py      # Core ML model code

└── README.md                   # This file

```



---



\## ▶️ How to Run the App



\### 🔧 1. Install Dependencies



Install Python libraries (create a virtual environment first if needed):



```bash

pip install streamlit pandas xgboost shap matplotlib openpyxl

```



\### 🚀 2. Run the Streamlit App



```bash

streamlit run hr\\\_attrition\\\_app.py

```



The app will open in your browser.



---



\## 📤 Export Options



\- Export individual prediction results

\- Download full prediction results as Excel

\- SHAP-based visual explanation included



---



\## 📌 Future Enhancements (To-do ideas)



1\. Deploy on Streamlit Cloud or Hugging Face Spaces.

2\. Add login authentication.

3\. Use LightGBM or CatBoost for model comparison.

4\. Display ROC-AUC and confusion matrix.

5\. Save user feedback on predictions.

6\. Add file upload for batch predictions.

7\. Deploy an API for integration with other platforms.

8\. Add interactive filters and toggles for features.

9\. Use more advanced explanation techniques (e.g., LIME).

10\. Enable PDF report generation.



---



\## 💡 Use Cases



\- HR departments assessing attrition risk

\- Business analysts exploring workforce patterns

\- Projects involving interpretable machine learning



---



\## ❤️ Developed by



\*\*Krishanu Mahapatra\*\*



---


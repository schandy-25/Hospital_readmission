# Hospital Readmission Risk Prediction

An end-to-end machine learning pipeline to predict 30-day hospital readmission risk for diabetic patients, with cost-sensitive threshold optimization and SHAP explainability.

## Overview

Hospital readmissions are costly and often preventable. This project builds a predictive model on 101,766 patient records to identify high-risk patients before discharge, helping clinical teams prioritize interventions.

## Dataset

- **Source:** UCI Diabetes 130-US Hospitals Dataset (Kaggle)
- **Size:** 101,766 patient records, 50 features
- **Target:** 30-day readmission (binary classification)

## Approach

1. **Data Cleaning & Preprocessing** — handled missing values, encoded categorical features, removed irrelevant columns
2. **Exploratory Data Analysis (EDA)** — identified key readmission drivers including number of inpatient visits, diagnosis codes, and medication changes
3. **Feature Engineering** — one-hot encoding, numeric scaling, interaction features
4. **Modeling** — HistGradientBoostingClassifier with cost-sensitive threshold optimization
5. **Evaluation** — ROC AUC ~0.686, cost-model threshold search minimizing weighted FN/FP cost
6. **Explainability** — SHAP values to interpret feature importance for clinical stakeholders
7. **Deployment** — Interactive Streamlit dashboard for clinical decision support

## Key Results

- Validation ROC AUC: ~0.686
- Cost-sensitive threshold selection minimizes average misclassification cost per patient
- SHAP analysis reveals top predictors: number of inpatient visits, discharge disposition, diagnosis codes

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn
- XGBoost, HistGradientBoostingClassifier
- SHAP for explainability
- Streamlit for dashboard deployment

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `notebooks/` — EDA and modeling notebooks
- `app.py` — Streamlit dashboard
- `requirements.txt` — dependencies

---

*Built as part of MS Data Science coursework at Indiana University, Bloomington.*

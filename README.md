# Retail Customer Intelligence Platform

A retail analytics project that combines data cleaning, EDA, machine learning, clustering, anomaly detection, and a Streamlit dashboard for customer segmentation, churn risk, CLV forecasting, and campaign insights.

## What it does
- Customer segmentation using supervised classification
- Churn-risk prediction using a behavioural proxy label
- Customer lifetime value forecasting with regression
- Behavioural clustering with KMeans
- Anomaly detection with Isolation Forest
- Rule-based campaign recommendations and exportable reports
- Interactive dashboard in Streamlit

## Models used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Random Forest Regressor
- KMeans
- Isolation Forest

## Dataset
- `data/retail_customer_segmentation.csv`
- 50,000 customer records
- Features include demographics, spend, engagement, returns, payment method, and region

## Workflow
1. `python src/data_cleaning.py`
2. `python src/eda_visualization.py`
3. `python src/modeling.py`
4. `python src/advanced_analytics.py`
5. `streamlit run src/app.py`

## Outputs
- `outputs/models/` — trained models and feature metadata
- `outputs/figures/` — charts and evaluation visuals
- `outputs/reports/` — summary reports and metrics

## Tech stack
- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- Streamlit
- joblib

## Run
  pip install -r requirements.txt
  cd src
  streamlit run app.py
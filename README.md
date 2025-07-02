# 🌫️ Predicting Air Quality with Stacking Ensemble: A Machine Learning Approach for Indian Cities

This project leverages **supervised machine learning** to predict **Air Quality Index (AQI)** levels across 26 Indian cities using pollutant and meteorological data (2015–2020). A customized **stacking ensemble model** built using **Random Forest**, **XGBoost**, and **LightGBM** proved most effective in classifying days as **safe (AQI ≤ 100)** or **unsafe (AQI > 100)**.

---

## 📌 Project Overview

Air pollution is a major health hazard in India. Accurate AQI forecasting helps governments and citizens take proactive safety measures. This study develops a robust machine learning pipeline that includes:

- Data preprocessing
- Feature engineering
- Model training and evaluation
- Model interpretability using SHAP
- Visualization and reporting

---

## 📊 Dataset

- **Source**: [city_day.csv](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Cities Covered**: Delhi, Mumbai, Chennai, Bangalore, Hyderabad, etc.
- **Period**: 2015 – 2020
- **Features**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, etc.
- **Target**: Binary AQI label (Safe ≤100 / Unsafe >100)

---

## 🔧 Features Used

- Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene
- Engineered: `Month`, `PM_Ratio = PM2.5 / PM10`, `AQI_Lag1`
- Label: `0 = Safe`, `1 = Unsafe`

---

## 🧼 Data Preprocessing

- Handled missing values via median imputation
- Removed `Xylene` and `AQI_Bucket` columns
- Performed **SMOTE** to balance class distribution
- Applied **StandardScaler**
- Stratified train-test split (80/20)

---

## 🤖 Models Used

### Individual Models

- Logistic Regression  
- Support Vector Machine  
- K-Nearest Neighbors  
- Naive Bayes  
- Random Forest 🌲  
- XGBoost ⚡  
- LightGBM 🌟  
- Gradient Boosting  

### 🚀 Final Stacking Ensemble

- **Base Models**: Random Forest, XGBoost, LightGBM
- **Meta-model**: Gradient Boosting Classifier
- **Weighted voting** based on recall of base learners

---

## 📈 Results

| Model              | Accuracy | F1-Score |
|-------------------|----------|----------|
| Random Forest      | 92.74%   | 94.05%   |
| XGBoost            | 92.43%   | 93.71%   |
| LightGBM           | 92.90%   | 94.15%   |
| Gradient Boosting  | 92.90%   | 94.16%   |
| **Stacking Ensemble** | **93.22%** | **94.45%** |

- Ensemble model showed the **highest recall (93.88%)** for unsafe days
- SHAP analysis confirmed **PM2.5** and **AQI_Lag1** as the most influential features

---

## 📊 Visualizations

- Confusion matrix, ROC curve
- Monthly pollutant trends (PM2.5, NO2, SO2, CO, O3)
- Feature importance via SHAP plots
- Seasonal AQI fluctuation graphs

---

## 🛠️ How to Run

### 1. Clone this repository

```bash
git clone https://github.com/your-username/air-quality-prediction.git
cd air-quality-prediction


Install dependencies
pip install -r requirements.txt

📚 Libraries Used
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imblearn
shap



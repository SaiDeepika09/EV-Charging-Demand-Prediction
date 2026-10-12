# EV Charging Demand Prediction

## Project Overview

The **EV Charging Demand Prediction** project uses Machine Learning and Time Series Forecasting to analyze electric vehicle charging demand, predict demand levels, and understand how charging patterns vary over time.

The project includes regression for predicting numerical demand, classification for identifying high-demand sessions, and time series forecasting for analyzing daily demand trends and forecasting future demand.

An interactive Streamlit dashboard presents predictions, model comparisons, feature importance, SHAP explanations, and time series visualizations.

## Objectives

- Analyze EV charging records to identify demand patterns.
- Clean and preprocess charging session data.
- Perform Exploratory Data Analysis (EDA).
- Engineer time-based and categorical features.
- Build and compare multiple regression models.
- Classify charging sessions into High Demand and Normal Demand.
- Analyze trends, seasonality, and stationarity in daily demand.
- Compare time series forecasting methods.
- Explain model predictions using feature importance and SHAP.
- Deploy the selected models through an interactive Streamlit dashboard.

## Dataset Description

The project uses EV charging session records containing information about charging stations, charger types, locations, timestamps, session durations, and demand.

**Dataset details after cleaning:**

- Records: 72,826
- Features: 18
- Missing values: 0
- Duplicate rows: 0

The dataset includes fields such as:

- `UserID`
- `ChargerID`
- `ChargerCompany`
- `Location`
- `ChargerType`
- `StartDatetime`
- `EndDatetime`
- `Duration`
- `Demand`
- `Calculated_Duration`

The original timestamps were converted into datetime format. Invalid charging sessions were removed, and the cleaned dataset was saved for subsequent analysis.

> Dataset availability: Add the original dataset's source or download link here if redistribution is not permitted. Do not commit confidential or restricted data to GitHub.

## Project Workflow

### 1. Data Cleaning

- Converted timestamp columns to datetime format.
- Identified and removed records with invalid timestamps.
- Calculated charging duration from the start and end timestamps.
- Removed duplicate records.
- Checked missing values and data consistency.
- Saved the cleaned dataset as `ev_charging_cleaned.csv`.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of demand and its relationship with charging characteristics.

Key areas analyzed:

- Demand distribution and skewness.
- Relationship between charging duration and demand.
- Demand by charger type and charger company.
- Demand across different charging locations.
- Hourly and day-of-week demand patterns.
- Charging session duration groups.

**Key observation:** Demand generally increased with charging duration. The correlation between demand and calculated duration was approximately 0.604.

### 3. Feature Engineering

The following features were created or prepared for modeling:

- `Hour`
- `Day`
- `DayOfWeek`
- `Month`
- `Quarter`
- `IsWeekend`
- `TimeOfDay`
- `Location`
- `ChargerType`
- `ChargerCompany`
- `Calculated_Duration`

Categorical features were encoded, and preprocessing pipelines were used where appropriate.

### 4. Regression — Charging Demand Prediction

The regression task predicts the numerical demand value for a charging session.

The following models were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**Selected model: LightGBM Regressor**

| Metric | Result |
|---|---:|
| MAE | 3.49 |
| RMSE | 6.10 |
| R² Score | 0.794 |

LightGBM was selected because it achieved the best overall RMSE and R² results among the compared regression models. XGBoost achieved a slightly lower MAE.

### 5. Classification — Demand Category Prediction

The classification task identifies whether a charging session belongs to the High Demand or Normal Demand category.

The median demand value of **14.1** was used as the threshold:

- High Demand: Demand greater than or equal to 14.1
- Normal Demand: Demand below 14.1

The dataset had approximately balanced classes, so oversampling was not required.

Models compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (SVM)
- XGBoost Classifier
- LightGBM Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**Selected model: XGBoost Classifier**

| Metric | Result |
|---|---:|
| Accuracy | 89.65% |
| Precision | 86.96% |
| Recall | 93.32% |
| F1-score | 90.03% |
| ROC-AUC | 95.72% |

XGBoost was selected because it achieved the best overall accuracy, precision, F1-score, and ROC-AUC among the compared classification models. LightGBM achieved a slightly higher recall.

## Streamlit Dashboard

The project includes an interactive dashboard with three pages.

### Regression

- Predicts numerical charging demand.
- Displays regression evaluation metrics.
- Compares regression models.
- Visualizes feature importance.
- Provides SHAP-based explanations for individual predictions.

### Classification

- Predicts High Demand or Normal Demand.
- Displays the classification threshold.
- Shows model comparison metrics and class distribution.
- Visualizes feature importance.
- Provides SHAP-based explanations for individual predictions.


## Model Serialization

The selected Machine Learning models were saved using Python pickle so that the trained models could be reused by the Streamlit dashboard.

Saved models:

- `models/lightgbm_demand_model.pkl`
- `models/xgboost_classification_model.pkl`

The dashboard loads the saved models instead of retraining them every time the application starts.

Only load pickle files from trusted sources because loading untrusted pickle files can execute malicious code.

## Project Structure

```text
EV-Charging Demand prediction/
│
├── ChargingRecords.csv
├── Processed Data/
│   └── ev_charging_cleaned.csv
│
├── models/
│   ├── lightgbm_demand_model.pkl
│   └── xgboost_classification_model.pkl
│
├── notebook/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Classification.ipynb
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── Regression.py
│       ├── Classification.py
│
├── requirements.txt
└── README.md
```

*Note: Adjust the notebook filenames in this tree to match your actual filenames before publishing.*

## Installation and Setup

### Prerequisites

- Python 3.10 or another Python version compatible with the dependencies.
- pip
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "EV-Charging Demand prediction"
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

From the project root directory, execute:

```bash
streamlit run dashboard/app.py
```

The application should open in your browser.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- XGBoost
- Statsmodels
- SHAP
- Streamlit
- Pickle
- Jupyter Notebook
- Git and GitHub

## Key Findings

- Charging demand showed a moderate positive relationship with calculated charging duration.
- Demand varied across charger types, companies, locations, and times of day.
- LightGBM performed best overall for the regression task based on RMSE and R².
- XGBoost achieved the strongest overall classification performance across the selected evaluation metrics.
- The daily demand series was non-stationary according to the ADF test.
- The 7-Day Moving Average outperformed the tested Exponential Smoothing model on the held-out time series period.
- Feature importance and SHAP explanations helped interpret model predictions.

## Limitations

- Predictions depend on the patterns and coverage of the available dataset.
- Feature importance and SHAP values explain model behavior; they do not establish causation.
- The classification threshold is based on the dataset's median demand and may need adjustment for a real-world operational use case.
- Model results should be validated on additional data before making operational decisions.

## Future Improvements

- Incorporate additional charging stations and longer historical records.
- Evaluate more advanced time series forecasting approaches.
- Monitor model performance as new data becomes available.
- Improve demand classification thresholds based on operational requirements.
- Deploy the dashboard to a cloud hosting platform.

## Conclusion

This project combines exploratory analysis, feature engineering, regression, classification, and time series forecasting to study EV charging demand. LightGBM was selected for numerical demand prediction, XGBoost for demand category classification, and the 7-Day Moving Average for daily demand forecasting.

The Streamlit dashboard brings these results together in an interactive application, demonstrating an end-to-end Machine Learning workflow from data preparation to model evaluation and deployment.

## Author

**Deepika**

GUVI Machine Learning Project — EV Charging Demand Prediction

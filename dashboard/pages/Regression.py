import pickle

import pandas as pd
import streamlit as st
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "Processed Data" / "ev_charging_cleaned.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_demand_model.pkl"
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

DATA_PATH = (
    r"D:/Guvi/EV-Charging Demand prediction/"
    r"Processed Data/ev_charging_cleaned.csv"
)

MODEL_PATH = (
    r"D:/Guvi/EV-Charging Demand prediction/"
    r"models/lightgbm_demand_model.pkl"
)

df = pd.read_csv(DATA_PATH)

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])


# --------------------------------------------------
# Load Regression Model
# --------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    regression_model = pickle.load(file)


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("⚡ EV Charging Demand Prediction")

st.header("📈 Regression – Charging Demand Prediction")

st.write(
    "Use the inputs below to predict the charging demand "
    "for a charging session."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    location = st.selectbox(
        "Location",
        sorted(df["Location"].unique())
    )

    charger_type = st.selectbox(
        "Charger Type",
        sorted(df["ChargerType"].unique())
    )

    charger_company = st.selectbox(
        "Charger Company",
        sorted(df["ChargerCompany"].unique())
    )

    calculated_duration = st.number_input(
        "Charging Duration (minutes)",
        min_value=0.0,
        value=60.0
    )


with col2:

    hour = st.slider(
        "Charging Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day = st.slider(
        "Day of Month",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )


# --------------------------------------------------
# Derived Features
# --------------------------------------------------

quarter = (month - 1) // 3 + 1

is_weekend = int(day_of_week in [5, 6])


def get_time_of_day(hour_value):

    if 0 <= hour_value < 6:
        return "Night"

    elif 6 <= hour_value < 12:
        return "Morning"

    elif 12 <= hour_value < 17:
        return "Afternoon"

    elif 17 <= hour_value < 21:
        return "Evening"

    else:
        return "Night"


time_of_day = get_time_of_day(hour)

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------

input_data = pd.DataFrame({
    "Hour": [hour],
    "Day": [day],
    "DayOfWeek": [day_of_week],
    "Month": [month],
    "Quarter": [quarter],
    "IsWeekend": [is_weekend],
    "TimeOfDay": [time_of_day],
    "Location": [location],
    "ChargerType": [charger_type],
    "ChargerCompany": [charger_company],
    "Calculated_Duration": [calculated_duration]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Charging Demand"):

    input_data = pd.DataFrame({
        "Hour": [hour],
        "Day": [day],
        "DayOfWeek": [day_of_week],
        "Month": [month],
        "Quarter": [quarter],
        "IsWeekend": [is_weekend],
        "TimeOfDay": [time_of_day],
        "Location": [location],
        "ChargerType": [charger_type],
        "ChargerCompany": [charger_company],
        "Calculated_Duration": [calculated_duration]
    })

    prediction = regression_model.predict(input_data)

    st.success(
        f"Predicted Charging Demand: {prediction[0]:.2f}"
    )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("R² Score", "0.794")

metric2.metric("RMSE", "6.10")

metric3.metric("MAE", "3.49")

# --------------------------------------------------
# Regression Model Comparison
# --------------------------------------------------

st.subheader("Regression Model Comparison")

regression_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM"
    ],
    "MAE": [
        6.2605,
        4.4521,
        3.5082,
        3.7775,
        3.4533,
        3.4910
    ],
    "RMSE": [
        10.0437,
        8.5996,
        6.2343,
        6.4884,
        6.1104,
        6.1000
    ],
    "R²": [
        0.4402,
        0.5896,
        0.7843,
        0.7664,
        0.7928,
        0.7935
    ]
})

st.dataframe(
    regression_results.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# LightGBM Feature Importance
# --------------------------------------------------

st.subheader("LightGBM Feature Importance")

lightgbm_model = regression_model.named_steps["model"]
preprocessor = regression_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importance_values = lightgbm_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Clean feature names
feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("numeric__", "", regex=False)
)

# Select top 10 features
feature_importance = (
    feature_importance
    .sort_values(by="Importance", ascending=False)
    .head(10)
)

st.bar_chart(
    feature_importance.set_index("Feature")["Importance"],
    height=500
)

st.caption(
    "Feature importance shows how useful each feature was for the model's "
    "predictions. It does not imply causation."
)

# --------------------------------------------------
# SHAP Explanation
# --------------------------------------------------

st.subheader("SHAP Explanation")

st.write(
    "SHAP explains how each feature contributed to the "
    "individual charging-demand prediction."
)

if st.button("Explain Prediction with SHAP"):

    # Transform the user's input using the same preprocessing
    # used during model training
    transformed_input = preprocessor.transform(input_data)

    # Create SHAP explainer for LightGBM
    explainer = shap.TreeExplainer(lightgbm_model)

    shap_values = explainer.shap_values(transformed_input)

    # Handle SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values
})

    shap_df["Feature"] = (
    shap_df["Feature"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("numeric__", "", regex=False)
    )

    shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()

    shap_df = shap_df.sort_values(
        by="Absolute SHAP",
        ascending=False
    ).head(10)

    st.bar_chart(
        shap_df.set_index("Feature")["SHAP Value"],
        height=500
    )

    st.write(
        "Positive SHAP values push the prediction higher, "
        "while negative SHAP values push the prediction lower."
    )
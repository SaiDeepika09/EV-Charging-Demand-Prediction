import pickle
import pandas as pd
import streamlit as st
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "Processed Data" / "ev_charging_cleaned.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_classification_model.pkl"

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EV Charging - Classification",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# File Paths
# --------------------------------------------------

DATA_PATH = r"D:/Guvi/EV-Charging Demand prediction/Processed Data/ev_charging_cleaned.csv"

MODEL_PATH = r"D:/Guvi/EV-Charging Demand prediction/models/xgboost_classification_model.pkl"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])


# --------------------------------------------------
# Load Classification Model
# --------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    classification_model = pickle.load(file)


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🎯 EV Charging Demand Classification")

st.write(
    "Predict whether a charging session belongs to "
    "**High Demand** or **Normal Demand**."
)

# --------------------------------------------------
# User Input
# --------------------------------------------------

st.subheader("Enter Charging Session Details")

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
        value=60.0,
        step=1.0
    )


with col2:

    hour = st.slider(
        "Hour",
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

    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )


# --------------------------------------------------
# Derived Features
# --------------------------------------------------

quarter = (month - 1) // 3 + 1

is_weekend = int(day_of_week in [5, 6])


def get_time_of_day(hour):
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
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
# Classification Threshold
# --------------------------------------------------

st.subheader("Classification Threshold")

median_demand = 14.1

st.info(
    f"""
    **Demand Classification Rule**

    - Demand < {median_demand} → Normal Demand
    - Demand ≥ {median_demand} → High Demand

    The threshold of {median_demand} was selected using the median
    Demand value from the cleaned dataset.
    """
)

# --------------------------------------------------
# Class Balance
# --------------------------------------------------

st.subheader("Class Balance")

class_balance = pd.DataFrame({
    "Demand Category": [
        "Normal Demand",
        "High Demand"
    ],
    "Count": [
        36387,
        36439
    ],
    "Percentage": [
        49.96,
        50.04
    ]
})

st.dataframe(
    class_balance.style.format({
        "Percentage": "{:.2f}%"
    }),
    use_container_width=True
)

st.success(
    "The two classes are almost perfectly balanced, "
    "so oversampling such as SMOTE was not required."
)
# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Demand Category"):

    prediction = classification_model.predict(input_data)

    if prediction[0] == 1:
        st.success("🔴 High Demand")
    else:
        st.info("🟢 Normal Demand")

# --------------------------------------------------
# Model Performance Comparison
# --------------------------------------------------

st.subheader("Classification Model Performance")

classification_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Linear SVM",
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        0.8263,
        0.8477,
        0.8937,
        0.7992,
        0.8267,
        0.8965,
        0.8945
    ],
    "Precision": [
        0.8096,
        0.8478,
        0.8677,
        0.7913,
        0.7826,
        0.8696,
        0.8643
    ],
    "Recall": [
        0.8536,
        0.8478,
        0.9293,
        0.8131,
        0.9050,
        0.9332,
        0.9361
    ],
    "F1 Score": [
        0.8310,
        0.8478,
        0.8974,
        0.8021,
        0.8394,
        0.9003,
        0.8988
    ],
    "ROC-AUC": [
        0.8997,
        0.8479,
        0.9516,
        0.8764,
        0.8955,
        0.9572,
        0.9564
    ]
})

st.dataframe(
    classification_results.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "ROC-AUC": "{:.2%}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# XGBoost Feature Importance
# --------------------------------------------------

st.subheader("XGBoost Feature Importance")

xgb_model = classification_model.named_steps["model"]

preprocessor = classification_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance_values = xgb_model.feature_importances_

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

# Sort and select top 10
feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
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
    "individual demand classification prediction."
)

if st.button("Explain Prediction with SHAP"):

    # Transform the input using the same preprocessing
    transformed_input = preprocessor.transform(input_data)

    # Create SHAP explainer for XGBoost
    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(transformed_input)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = shap_values[0]

    # Create SHAP dataframe
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })

    # Clean feature names
    shap_df["Feature"] = (
        shap_df["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
    )

    # Calculate absolute SHAP value
    shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()

    # Show top 10 features
    shap_df = (
        shap_df
        .sort_values(
            by="Absolute SHAP",
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        shap_df.set_index("Feature")["SHAP Value"],
        height=500
    )

    st.write(
        "Positive SHAP values push the prediction toward "
        "High Demand, while negative SHAP values push the "
        "prediction toward Normal Demand."
    )

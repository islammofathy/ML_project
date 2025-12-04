import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ===============================
# Load Artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    feature_columns = joblib.load("columns.pkl")
    best_model = joblib.load("best_model.pkl")
    gbr_model = joblib.load("GradientBoosting.pkl")  # Gradient Boosting Regressor
    return label_encoders, scaler, feature_columns, best_model, gbr_model

try:
    label_encoders, scaler, feature_columns, best_model, gbr_model = load_artifacts()
except Exception as e:
    st.error("❌ Error loading model artifacts. Make sure all .pkl files are in the same folder as this app.")
    st.exception(e)
    st.stop()

categorical_cols = list(label_encoders.keys())
numeric_cols = [col for col in feature_columns if col not in categorical_cols]

# ===============================
# Header
# ===============================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #1f2933;
    }
    .sub-title {
        font-size: 16px;
        color: #4b5563;
    }
    .footer-text {
        font-size: 12px;
        color: #9ca3af;
        text-align: center;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚗 Car Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter the car features, and the app will predict the price using two models: Gradient Boosting and the overall best model. Then it averages them for a robust estimate.</div>',
    unsafe_allow_html=True
)

st.write("---")

# ===============================
# Sidebar Info
# ===============================
with st.sidebar:
    st.header("ℹ About")
    st.write(
        """
        This app uses:
        - *Label Encoders* for categorical features  
        - *MinMaxScaler* for numerical features  
        - *GradientBoostingRegressor*  
        - *Best model selected by R²*  
        
        The final prediction is the *average* of both models.
        """
    )
    st.write("📦 Models & preprocessors loaded from .pkl files.")

# ===============================
# Input Form
# ===============================
st.subheader("🔢 Input Features")

with st.form("prediction_form"):
    col_cat, col_num = st.columns(2)

    user_input = {}

    # Categorical inputs (based on label_encoders)
    with col_cat:
        st.markdown("### 🧩 Categorical Features")
        if len(categorical_cols) == 0:
            st.write("No categorical features detected.")
        for col in categorical_cols:
            classes = label_encoders[col].classes_
            default_value = classes[0] if len(classes) > 0 else ""
            val = st.selectbox(f"{col}", options=list(classes), index=0)
            user_input[col] = val

    # Numerical inputs
    with col_num:
        st.markdown("### 🔢 Numerical Features")
        if len(numeric_cols) == 0:
            st.write("No numerical features detected.")
        for col in numeric_cols:
            # Generic numeric input; customize min/max if you know them
            val = st.number_input(f"{col}", value=0.0)
            user_input[col] = val

    submitted = st.form_submit_button("🚀 Predict Price")

# ===============================
# Prediction Logic
# ===============================
def preprocess_and_predict(user_input_dict):
    # Build a DataFrame with one row
    df_input = pd.DataFrame([user_input_dict])

    # 1) Encode categorical columns using saved LabelEncoders
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))

    # 2) Scale numeric columns using MinMaxScaler
    #    scaler was fit only on numeric_cols during training
    df_input_numeric = df_input[numeric_cols].copy()
    df_input[numeric_cols] = scaler.transform(df_input_numeric)

    # 3) Reorder columns to match training feature order
    df_input = df_input[feature_columns]

    # 4) Predict with Gradient Boosting and best model
    pred_gbr = gbr_model.predict(df_input)[0]
    pred_best = best_model.predict(df_input)[0]
    pred_avg = (pred_gbr + pred_best) / 2.0

    return pred_gbr, pred_best, pred_avg

# ===============================
# Show Predictions
# ===============================
if submitted:
    try:
        pred_gbr, pred_best, pred_avg = preprocess_and_predict(user_input)

        st.write("---")
        st.subheader("📊 Predictions")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                label="Gradient Boosting Prediction",
                value=f"{pred_gbr:,.2f}"
            )
        with c2:
            st.metric(
                label="Best Model Prediction",
                value=f"{pred_best:,.2f}"
            )
        with c3:
            st.metric(
                label="Average of Both",
                value=f"{pred_avg:,.2f}"
            )

        st.write("---")
        with st.expander("📈 Details"):
            st.write("*Raw Input (after mapping to model features):*")
            st.json(user_input)

    except Exception as e:
        st.error("❌ Something went wrong during prediction.")
        st.exception(e)

# ===============================
# Footer
# ===============================
st.markdown(
    '<div class="footer-text">Made for your ML project • Streamlit + scikit-learn 🚀</div>',
    unsafe_allow_html=True
)
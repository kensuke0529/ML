import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Load model
model_path = 'models/xgbclassifier.json'
try:
    model = xgb.Booster()
    model.load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection App")

st.markdown(
    "Use the form below to simulate a credit card transaction and check if it might be **fraudulent**."
)

tab1, tab2 = st.tabs(["📈 Model", "🗃 Model Performance Metrics"])

with tab1:
    # Define features
    feature_names = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]

    input_values = []

    with st.expander("🧾 Transaction Feature Input", expanded=True):
        st.markdown(
            "🔧 Enter the transaction features (use actual or test values):")

        cols = st.columns(5)
        for i, name in enumerate(feature_names):
            col = cols[i % 5]
            val = col.number_input(name, value=0.0, step=0.01, format="%.3f")
            input_values.append(val)

    # Convert input to numpy array and reshape for model
    features = np.array(input_values).reshape(1, -1)

    # Prediction button
    if st.button("🚀 Predict Fraud"):
        dmat = xgb.DMatrix(features)
        probs = model.predict(dmat)
        pred = int(probs[0] > 0.5)
        prob = probs[0]

        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        if pred == 1:
            st.error(
                f"⚠️ Fraudulent Transaction Detected!\n\n**Probability:** {prob:.2f}")
        else:
            st.success(
                f"✅ Legitimate Transaction\n\n**Probability:** {prob:.2f}")

with tab2:
    st.write("Here are the performance metrics of the trained models:")

    data = {
        "Precision": ["0.99", "0.32", "0.99"],
        "Recall": ["0.77", "0.87", "0.84"],
        "F1 Score": ["0.86", "0.46", "0.91"],
        "AUC": ["0.96", "0.97", "0.98"]
    }

    index = ["RandomForest", "LogisticRegression", "XGBoostClassifier"]
    df = pd.DataFrame(data, index=index)

    st.table(df)

    st.write("XGBoost Classifier is the best model with high Recall and F1 Score!")

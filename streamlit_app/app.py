import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Load model
model = joblib.load("models/xgbclassifier.joblib")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection App")

st.markdown(
    "Use the form below to simulate a credit card transaction and check if it might be **fraudulent**.")

tab1, tab2, tab3 = st.tabs(
    ["📈 Model", "🗃 Model Performance Metrics", "Dataset"])

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

        # Group 5 columns
        cols = st.columns(5)
        for i, name in enumerate(feature_names):
            col = cols[i % 5]
            val = col.number_input(name, value=0.0, step=0.01, format="%.3f")
            input_values.append(val)

    # Convert to array
    features = np.array(input_values).reshape(1, -1)

    # Prediction button
    if st.button("🚀 Predict Fraud"):
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

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

with tab3:

    df = pd.read_csv(
        "data/creditcard.csv")

    st.write('#### Dataset Overview')
    st.write(f"Total Transactions: {len(df)}")

    st.write('Due to the privacy and security of the dataset, most of the columns are implemented PCA features. Therefore, we cannot interpret the meaning of each feature except Time and Amount.')
    st.write('The dataset contains 30 features, including PCA transformed features (V1 to V28) and two original features.')

    st.write("### Credit Card Fraud Detection Dataset")
    st.table(df.head(10))

    # Show dataset
    col1, col2, col3 = st.columns(3)

    class_count = df['Class'].value_counts()


# --- Plot 1: Class Distribution (Pie Chart)
    with col1:
        st.markdown("**Class Distribution**")
        fig1, ax1 = plt.subplots()
        ax1.pie(class_count, labels=class_count.index,
                autopct='%1.1f%%', startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        st.pyplot(fig1)

    # --- Plot 2: Transaction Amount Distribution
    with col2:
        st.markdown("**Transaction Amount**")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['Amount'], bins=50, color='skyblue', edgecolor='black')
        ax2.set_title("Amount")
        st.pyplot(fig2)

    # --- Plot 3: Time Feature Distribution
    with col3:
        st.markdown("**Time Feature**")
        fig3, ax3 = plt.subplots()
        ax3.hist(df['Time'], bins=100, color='orange', edgecolor='black')
        ax3.set_title("Time")
        st.pyplot(fig3)

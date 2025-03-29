import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.title('Linear Regression Web App')

uploaded_file = st.file_uploader(" Upload CSV", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())

    # Select Features and Target
    columns = data.columns.tolist()
    target = st.selectbox("Select Target Variable", columns)
    features = st.multiselect("🛠️ Select Feature Columns", [col for col in columns if col != target])

    if features:
        X = data[features]
        y = data[target]

        # Train Model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Show Predictions
        results = pd.DataFrame({
            "Actual": y,
            "Predicted": y_pred
        })
        st.subheader(" Predictions")
        st.write(results.head())

        # Show R2 Score
        r2 = r2_score(y, y_pred)
        st.subheader(f"R² Score: {r2:.2f}")

        # Plot Results
        st.subheader("Actual vs Predicted Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color='blue', alpha=0.6)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)

        st.success("Model Trained Successfully! ")
    else:
        st.info(" Please select at least one feature.")

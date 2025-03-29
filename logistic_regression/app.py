import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title('Logistic Regression Web App with Regularization')

# Upload Data
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Select Features and Target
    features = st.multiselect("Select Features", data.columns)
    target = st.selectbox("Select Target", data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Select Regularization Type
        regularization = st.selectbox("Select Regularization", ["L1", "L2", "L1 + L2 (ElasticNet)"])
        c_value = st.slider("Select Inverse of Regularization Strength (C)", 0.01, 10.0, 1.0)

        if regularization == "L1 + L2 (ElasticNet)":
            l1_ratio = st.slider("Select L1 Ratio (ElasticNet)", 0.0, 1.0, 0.5)
            model = LogisticRegression(penalty='elasticnet', solver='saga', C=c_value, l1_ratio=l1_ratio)
        else:
            penalty = 'l1' if regularization == "L1" else 'l2'
            model = LogisticRegression(penalty=penalty, solver='saga', C=c_value)

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)
        st.write("Classification Report:")
        st.text(report)

        st.success("Model training and evaluation completed!")
    else:
        st.error("Please select both features and target variables.")
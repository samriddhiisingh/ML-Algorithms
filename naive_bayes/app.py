import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title(' Naive Bayes Classifier Web App')

uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader(" Data Preview")
    st.write(data.head())

    columns = data.columns.tolist()
    target = st.selectbox("Select Target Column", columns)
    features = st.multiselect("🛠️ Select Feature Columns", [col for col in columns if col != target])

    if features:
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader(f" Accuracy: {accuracy:.2f}")

        st.subheader(" Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader(" Classification Report")
        st.text(report)

        st.success("Model training and evaluation completed successfully!")

    else:
        st.warning(" Please select at least one feature column.")

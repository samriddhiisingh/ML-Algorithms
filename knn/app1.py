import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 K-Nearest Neighbors (KNN) Classifier")
st.write("Upload your dataset and train a KNN model interactively.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning(" Please upload a CSV file to proceed.")


else:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Dataset Preview")
    st.write(df.head())

    columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column", columns)
    feature_columns = st.multiselect("🛠️ Select Feature Columns", [col for col in columns if col != target_column])

    if feature_columns and target_column:
        if df[feature_columns].select_dtypes(exclude=[np.number]).shape[1] > 0:
            st.error(" Only numeric features are allowed. Please remove categorical features or encode them.")
        else:
            X = df[feature_columns]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            k = st.slider("Select Number of Neighbors (K)", 1, 20, 5)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            st.subheader(f" Model Accuracy: {accuracy:.2f}")

            results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.subheader(" Predictions")
            st.write(results.head())

            st.subheader(" Prediction Distribution")
            st.bar_chart(results["Predicted"].value_counts())

            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader(" Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.success("KNN Model Trained and Evaluated Successfully! ")

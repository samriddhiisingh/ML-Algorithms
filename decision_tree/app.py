import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

st.title('Decision Tree Classifier Web App')

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

        # Select Hyperparameters
        max_depth = st.slider("Select Max Depth", 1, 20, 5)
        criterion = st.selectbox("Select Criterion", ["gini", "entropy"])

        # Train Model
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
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

        # Visualize Decision Tree
        st.write("### Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(10, 8))
        tree.plot_tree(model, feature_names=features, class_names=[str(c) for c in model.classes_], filled=True)
        st.pyplot(fig)

        st.success("Model training and evaluation completed!")
    else:
        st.error("Please select both features and target variables.")
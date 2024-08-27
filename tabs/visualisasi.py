import streamlit as st
import joblib
import pandas as pd
import numpy as np  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def visualisasi():
    scaler = joblib.load('models/scaler.pkl')
    dt_model = joblib.load('models/decision_tree_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    nb_model = joblib.load('models/naive_bayes_model.pkl')

    # Function to load data
    def load_data():
        data = pd.read_csv("diabetes1.csv")
        return data

    # Function to preprocess data
    def preprocess_data(data):
        data['smoking_history'] = data['smoking_history'].replace('No Info', np.NaN)
        mode_value = data['smoking_history'].mode()[0]
        data['smoking_history'].fillna(mode_value, inplace=True)
        data.drop(data[data['gender'] == 'Other'].index, inplace=True)
        smoking_history_mapping = {'never': 0, 'current': 1, 'former': 2, 'ever': 3, 'not current': 4}
        gender_mapping = {'Female': 0, 'Male': 1}
        data['smoking_history'] = data['smoking_history'].map(smoking_history_mapping)
        data['gender'] = data['gender'].map(gender_mapping)
        data['smoking_history'] = data['smoking_history'].astype('int64')
        data['gender'] = data['gender'].astype('int64')
        columns_to_normalize = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        # Ensure columns_to_normalize exactly match the expected column names for scaling
        expected_columns = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        if set(columns_to_normalize) != set(expected_columns):
            raise ValueError("Column names for scaling do not match the expected columns.")

        # Perform manual scaling
        for col in columns_to_normalize:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

        return data

    # Function to get accuracy and confusion matrix
    def get_model_evaluation(model, X_test, y_test):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm

    model_choice = st.selectbox("Choose a model", ["Decision Tree", "KNN", "Naive Bayes"])

    data = load_data()
    data = preprocess_data(data)
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Decision Tree":
        model = dt_model
    elif model_choice == "KNN":
        model = knn_model
    elif model_choice == "Naive Bayes":
        model = nb_model

    report, cm = get_model_evaluation(model, X_test, y_test)

    st.write("### Classification Report")
    st.json(report)

    st.write("### Confusion Matrix")
    st.write(cm)

import streamlit as st
from tabs import dashboard, visualisasi, multiple_predict, history, about
import joblib

def main():
    try:
        decision_tree_model = joblib.load('models/decision_tree_model.pkl')
        knn_model = joblib.load('models/knn_model.pkl')
        naive_bayes_model = joblib.load('models/naive_bayes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file 'decision_tree_model.pkl', 'knn_model.pkl', dan 'naive_bayes_model.pkl' berada di direktori 'models'.")
        return

    st.title('Diabpred')

    st.sidebar.title('Menu')
    menu = st.sidebar.radio('', ['Dashboard', 'Visualisasi', 'Multiple Predict', 'History', 'About'])

    # Tambahkan pemilihan model di sidebar
    st.sidebar.title('Pilih Model')
    model_name = st.sidebar.radio('', ['Decision Tree', 'KNN', 'Naive Bayes'])

    # Pemetaan nama model ke objek model
    models = {
        'Decision Tree': decision_tree_model,
        'KNN': knn_model,
        'Naive Bayes': naive_bayes_model
    }

    selected_model = models[model_name]

    if menu == 'Dashboard':
        dashboard.dashboard(selected_model, scaler, model_name)
    elif menu == 'Visualisasi':
        visualisasi.visualisasi()
    elif menu == 'Multiple Predict':
        multiple_predict.multiple_predict(selected_model, scaler,model_name)
    elif menu == 'History':
        history.history()
    elif menu == 'About':
        about.about()

if __name__ == '__main__':
    main()
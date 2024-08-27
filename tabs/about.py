import streamlit as st

def about():
    col1, col2 = st.columns(2)

    with col1:
        st.write("""Why use this application?
                    This application can predict whether someone has diabetes or not
                    Help medical personnel to diagnose diabetes
                    Help people to know whether they have diabetes or not
                    Help people to monitor their health"""
                )

    with col2:
        st.write("""Why use this dataset?
                    Linkage of Diabetes Risk Factors
                    Prediction and Diagnosis of Diabetes
                    Important Information
                    Real World Application"""
                )

    col3, col4 = st.columns(2)

    with col3:
        st.write("""About Dataset Diabetes Prediction
                    The Dataset that using for Train Model
                    Link for Diabetes Prediction Dataset: [Click Here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"""
                )

    with col4:
        st.write("""Why use this dataset?
                    Linkage of Diabetes Risk Factors
                    Prediction and Diagnosis of Diabetes
                    Important Information
                    Real World Application"""
                )
    
    st.write("")

    col1 = st.columns(1)


    st.image("assets/deny.jpg", caption="Yohanes Deny Novandian", use_column_width=True)


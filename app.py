import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
model = load_model('regression_model.h5')

# Define the input fields for the user
def user_input_features():
    age = st.number_input("Age", min_value=0, max_value=100, value=23)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    gender_flag = 1 if Gender == "Male" else 0
    BMI = st.number_input("BMI", min_value=0, max_value=50, value=15)
    Children = st.number_input("Number of Children", min_value=0, max_value=10, value=2)
    Smoker = st.selectbox("Smoker or Not?", ["Smoker", "Non-Smoker"])
    smoker_flag = 1 if Smoker == "Smoker" else 0
    AnnualIncome = st.number_input("Annual Family Income", min_value=0, value=70000)
    FamilyHistory = st.selectbox("Any Family Medical History", ["Yes", "No"])
    family_history_flag = 1 if FamilyHistory == "Yes" else 0
    CoverageAmount = st.number_input("Insurance Coverage Amount", min_value=0, value=10000)
    Dependent = st.number_input("Number of Dependent", min_value=0, max_value=10, value=1)
    MarritalStauts = st.selectbox("Marrital Status", ["Married", "Unmarried"])
    marital_status_flag = 1 if MarritalStauts == "Married" else 0
    Profession = st.selectbox("Profession", ['Self Employed', 'Other', 'Realtor', 'Teacher', 'Business Owner',
                                             'Doctor', 'Field Worker', 'Engineer'])

    # Set profession-related flags
    profession_flags = {
        'Self Employed': 0,
        'Other': 0,
        'Realtor': 0,
        'Teacher': 0,
        'Business Owner': 0,
        'Doctor': 0,
        'Field Worker': 0,
        'Engineer': 0
    }
    profession_flags[Profession] = 1

    # Create a DataFrame with the user inputs
    data = {
        'Age': [age],
        'Gender': [gender_flag],
        'BMI': [BMI],
        'Children': [Children],
        'Smoker': [smoker_flag],
        'AnnualIncome': [AnnualIncome],
        'FamilyHistory': [family_history_flag],
        'CoverageAmount': [CoverageAmount],
        'Dependent': [Dependent],
        'MarritalStatus': [marital_status_flag],
        'Profession_Doctor': [profession_flags['Doctor']],
        'Profession_Engineer': [profession_flags['Engineer']],
        'Profession_Field Worker': [profession_flags['Field Worker']],
        'Profession_Other': [profession_flags['Other']],
        'Profession_Realtor': [profession_flags['Realtor']],
        'Profession_Self Employed': [profession_flags['Self Employed']],
        'Profession_Teacher': [profession_flags['Teacher']]
    }
    
    features = pd.DataFrame(data)
    return features

st.title("Health Insurance Cost Predictor")
st.subheader('Presented By: Group 04')

# Get user input
input_df = user_input_features()

# Add a button to trigger prediction
if st.button("Predict"):
    # Validate that all fields have valid data
    if not input_df.isnull().values.any() and (input_df != '').all().all():
        try:
            # Predict using the model
            predictions = model.predict(input_df)

            # Display the results
            st.subheader('Predicted Insurance Costs')
            st.write(f"Manulife: ${predictions[0][0]:.2f}")
            st.write(f"Sun Life: ${predictions[0][1]:.2f}")
            st.write(f"Canada Life: ${predictions[0][2]:.2f}")
            st.write(f"Ontario Blue Cross: ${predictions[0][3]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please ensure all fields are filled out correctly.")

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:55:01 2025

@author: DEVADOS
"""
"""
"""
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import sqlite3

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (name TEXT, age TEXT, phone TEXT)''')
conn.commit()

# Load the saved models
diabetes_model = pickle.load(open('C:/Users/DEVADOS/Downloads/Multiple Disease Prediction System/saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/DEVADOS/Downloads/Multiple Disease Prediction System/saved models/heartdisease.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/DEVADOS/Downloads/Multiple Disease Prediction System/saved models/parkinsons.sav', 'rb'))
bmi_model = pickle.load(open('C:/Users/DEVADOS/Downloads/Multiple Disease Prediction System/saved models/bmi_classifier.pkl', 'rb'))
bmi_scaler = pickle.load(open('C:/Users/DEVADOS/Downloads/Multiple Disease Prediction System/saved models/scaler.pkl', 'rb'))


# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #4CAF50; text-align: center; }
    .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Login', 'About', 'BMI Tracker', 'Disease Prediction'],
                           icons=['person', 'info-circle', 'calculator', 'stethoscope'],
                           menu_icon='stethoscope',
                           default_index=0)

# Login Page
if selected == "Login":
    st.title("User Login")
    name = st.text_input("Enter Your Name")
    age = st.text_input("Enter Your Age")
    phone = st.text_input("Enter Your Phone Number")
    if st.button("Proceed"):
        if name and age and phone:
            st.session_state["logged_in"] = True
            st.session_state["name"] = name
            st.session_state["age"] = age
            st.session_state["phone"] = phone
            c.execute("INSERT INTO users (name, age, phone) VALUES (?, ?, ?)", (name, age, phone))
            conn.commit()
            st.success("Login successful! Navigate to the other pages using the sidebar.")
        else:
            st.error("Please fill in all details")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("Please log in to access other sections.")
    st.stop()

# About Page with Bootstrap Icons
elif selected == "About":
    st.title("📖 About This Application")
    st.write("""
    This application predicts diseases like Diabetes, Heart Disease, and Parkinson’s using machine learning models.
    """)

    # Diabetes Section
    st.markdown("""
    <h3><i class="bi bi-droplet-fill" style="color: #e63946;"></i> Diabetes</h3>
    """, unsafe_allow_html=True)
    st.write("""
    Diabetes is a chronic condition that affects how the body processes blood sugar (glucose). If left untreated, it can lead to severe complications like kidney failure, blindness, and heart disease.
    """)
    st.write("**Symptoms:** Frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision.")
    st.write("**Causes:** Insulin resistance, genetics, obesity, physical inactivity.")
    st.write("**Remedies:** Healthy diet, regular exercise, insulin therapy, blood sugar monitoring.")

    # Heart Disease Section
    st.markdown("""
    <h3><i class="bi bi-heart-pulse-fill" style="color: #d62828;"></i> Heart Disease</h3>
    """, unsafe_allow_html=True)
    st.write("""
    Heart disease refers to a range of conditions affecting the heart, including coronary artery disease, arrhythmias, and heart failure. It remains one of the leading causes of death worldwide.
    """)
    st.write("**Symptoms:** Chest pain, shortness of breath, dizziness, irregular heartbeat.")
    st.write("**Causes:** High cholesterol, high blood pressure, smoking, obesity, diabetes.")
    st.write("**Remedies:** Lifestyle changes, medication, stress management, surgery if needed.")

    # Parkinson’s Disease Section
    st.markdown("""
    <h3><i class="bi bi-brain" style="color: #6a4c93;"></i> Parkinson's Disease</h3>
    """, unsafe_allow_html=True)
    st.write("""
    Parkinson’s disease is a progressive nervous system disorder that affects movement, often causing tremors and muscle stiffness. It primarily results from the loss of dopamine-producing brain cells.
    """)
    st.write("**Symptoms:** Tremors, muscle stiffness, slow movements, balance problems.")
    st.write("**Causes:** Loss of dopamine-producing brain cells, genetics, environmental factors.")
    st.write("**Remedies:** Medications (Levodopa), physical therapy, deep brain stimulation (DBS).")




# BMI Tracker
elif selected == "BMI Tracker":
    st.title("📊 BMI Prediction")
    weight = st.text_input("Enter your weight in kg")
    height = st.text_input("Enter your height in meters")
    gender = st.selectbox("Select your gender", ["Male", "Female", "Prefer not to say"])
    if st.button("Predict BMI Category"):
        try:
            weight = float(weight)
            height = float(height)
            input_data = np.array([[height, weight]])
            input_scaled = bmi_scaler.transform(input_data)
            bmi_prediction = bmi_model.predict(input_scaled)
            categories = ["Extremely Underweight", "Underweight", "Normal", "Overweight", "Obese", "Extremely Obese"]
            st.success(f"Predicted BMI Category: {categories[bmi_prediction[0]]}")
        except ValueError:
            st.error("Please enter valid numerical values for height and weight.")

# Disease Prediction
elif selected == "Disease Prediction":
    disease_selected = option_menu("Disease Prediction", ['Diabetes', 'Heart Disease', "Parkinson's"],
                                   icons=['capsule', 'heart', 'person-circle'],
                                   default_index=0, orientation='horizontal')
    
    if disease_selected == 'Diabetes':
        st.title('Diabetes Prediction')
        inputs = [st.text_input(label) for label in ['Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 'Insulin Level', 'BMI', 'Diabetes Pedigree Function', 'Age']]
        if st.button('Predict Diabetes'):
            input_data = np.array([inputs]).astype(float)
            prediction = diabetes_model.predict(input_data)
            st.success('The person is Diabetic!' if prediction[0] else 'The person is NOT Diabetic!')
    
    elif disease_selected == 'Heart Disease':
        st.title('Heart Disease Prediction')
        inputs = [st.text_input(label) for label in ['Age', 'Sex', 'Chest Pain types', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'ST Depression', 'Slope of ST Segment', 'Major Vessels', 'Thal']] 
        if st.button('Predict Heart Disease'):
            input_data = np.array([inputs]).astype(float)
            prediction = heart_disease_model.predict(input_data)
            st.success('The person HAS Heart Disease!' if prediction[0] else 'The person does NOT have Heart Disease!')
    
    elif disease_selected == "Parkinson's":
        st.title("Parkinson's Prediction")
        inputs = [st.text_input(label) for label in ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']]
        if st.button("Parkinson's Test Result"):
            input_data = np.array([inputs]).astype(float)
            prediction = parkinsons_model.predict(input_data)
            st.success("The person has Parkinson's disease" if prediction[0] else "The person does not have Parkinson's disease")

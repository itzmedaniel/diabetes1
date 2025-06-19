import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("full_diabetes_model_rf.pkl")

st.set_page_config(page_title="Advanced Diabetes Risk App", layout="centered")
st.title("🩺 Advanced Diabetes Risk Predictor")
st.markdown("""
<style>
    .stSlider > div > div {
        padding-top: 5px;
    }
    .stRadio > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("Enter your health and lifestyle information to calculate your risk of diabetes.")

# Sidebar: Personal Info
st.sidebar.header("👤 Personal Info")
age = st.sidebar.slider("Age", 18, 80, 30)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
height = st.sidebar.number_input("Height (cm)", 130, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
bmi = round(weight / ((height / 100) ** 2), 1)
st.sidebar.markdown(f"**Calculated BMI:** `{bmi}` *(Normal: 18.5 - 24.9)*")

pregnancies = st.sidebar.slider("Pregnancies", 0, 10, 2) if sex == "Female" else 0

# Vitals
st.header("💓 Vitals")
systolic = st.slider("Systolic BP (mmHg)", 90, 200, 120, help="Low: <90 | Normal: 90-119 | High: >120")
diastolic = st.slider("Diastolic BP (mmHg)", 50, 130, 80, help="Low: <60 | Normal: 60-79 | High: >80")
pulse = st.slider("Pulse (bpm)", 40, 130, 75, help="Low: <60 | Normal: 60-100 | High: >100")
skin_thickness = st.slider("Skin Thickness (mm)", 5, 50, 20, help="Normal range: 10-40 mm")

# Lab Results
st.header("🧪 Lab Results")
glucose = st.slider("Glucose (mg/dL)", 70, 300, 100, help="Low: <70 | Normal: 70-99 | High: >100")
insulin = st.slider("Insulin (μU/mL)", 10, 300, 100, help="Low: <2 | Normal: 2–25 | High: >25")
a1c = st.slider("A1C (%)", 3.0, 12.0, 5.5, help="Low: <4 | Normal: 4–5.6 | High: >5.7")
chol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200, help="Desirable: <200 | High: >240")
hdl = st.slider("HDL (mg/dL)", 20, 100, 50, help="Low: <40 | Good: >60")
ldl = st.slider("LDL (mg/dL)", 50, 250, 100, help="Optimal: <100 | High: >160")

# Lifestyle
st.header("🏃 Lifestyle & History")
smoker = st.radio("Do you smoke?", ["No", "Yes"], help="Smoking increases diabetes risk")
alcohol = st.radio("Do you drink alcohol?", ["No", "Yes"], help="Alcohol in excess increases risk")
exercise = st.radio("Do you exercise regularly?", ["Yes", "No"], help="Improves insulin sensitivity")
eats_fruits = st.radio("Eat fruits daily?", ["Yes", "No"])
eats_veggies = st.radio("Eat vegetables daily?", ["Yes", "No"])
family_history = st.radio("Family history of diabetes?", ["Yes", "No"])
activity_level = st.radio("Activity level", ["Sedentary", "Moderate", "Active"])

# Predict
if st.button("🔍 Predict Diabetes Risk"):
    input_data = pd.DataFrame([[
        age,
        0 if sex == 'Male' else 1,
        height,
        weight,
        pregnancies,
        systolic,
        diastolic,
        pulse,
        skin_thickness,
        glucose,
        insulin,
        a1c,
        chol,
        hdl,
        ldl,
        1 if smoker == 'Yes' else 0,
        1 if alcohol == 'Yes' else 0,
        1 if exercise == 'Yes' else 0,
        1 if eats_fruits == 'Yes' else 0,
        1 if eats_veggies == 'Yes' else 0,
        1 if family_history == 'Yes' else 0,
        {"Sedentary": 0, "Moderate": 1, "Active": 2}[activity_level],
        bmi
    ]], columns=[
        'Age', 'Sex', 'Height_cm', 'Weight_kg', 'Pregnancies',
        'SystolicBP', 'DiastolicBP', 'Pulse', 'SkinThickness', 'Glucose',
        'Insulin', 'A1C', 'Cholesterol_Total', 'HDL', 'LDL',
        'Smoker', 'Alcohol', 'Exercise', 'EatsFruits', 'EatsVeggies',
        'FamilyHistory', 'ActivityLevel', 'BMI'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    fig, ax = plt.subplots()
    sns.barplot(x=['Risk %', 'Safe %'], y=[probability*100, (1-probability)*100], ax=ax)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of diabetes ({probability*100:.2f}%)")

    # Save prediction history
    record = input_data.copy()
    record['Prediction'] = "High Risk" if prediction == 1 else "Low Risk"
    record['RiskProbability'] = round(probability * 100, 2)

    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        history = pd.concat([history, record], ignore_index=True)
    else:
        history = record

    history.to_csv(history_file, index=False)

    # Show history
    with st.expander("📈 Prediction History"):
        st.dataframe(history.tail(10))
        st.line_chart(history['RiskProbability'].tail(10))

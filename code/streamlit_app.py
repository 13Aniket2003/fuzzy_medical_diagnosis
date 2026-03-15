import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

st.set_page_config(page_title="Fuzzy Diabetes Diagnosis", page_icon="🩺", layout="centered")
st.title("🩺 Fuzzy Logic–Based Diabetes Diagnosis System")

# --- How to Use Section ---
st.markdown("""
### 🧭 How to Use:
1. Adjust the sliders for **Glucose**, **Blood Pressure**, **BMI**, and **Age**.
2. Click **🩺 Compute Diabetes Risk** to calculate the risk score.
3. View the **risk level**, **health advice**, and top diagnosis result.
4. Click **💾 Save Patient Report** to store the result in a CSV file.
""")
st.markdown("---")

# --- Define fuzzy variables ---
glucose = ctrl.Antecedent(np.arange(50, 201, 1), 'glucose')
bp = ctrl.Antecedent(np.arange(60, 181, 1), 'bp')
bmi = ctrl.Antecedent(np.arange(10, 51, 1), 'bmi')
age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# --- Membership functions ---
glucose['low'] = fuzz.trimf(glucose.universe, [50, 70, 100])
glucose['normal'] = fuzz.trimf(glucose.universe, [90, 110, 140])
glucose['high'] = fuzz.trimf(glucose.universe, [130, 160, 200])

bp['low'] = fuzz.trimf(bp.universe, [60, 70, 90])
bp['normal'] = fuzz.trimf(bp.universe, [80, 100, 120])
bp['high'] = fuzz.trimf(bp.universe, [110, 140, 180])

bmi['low'] = fuzz.trimf(bmi.universe, [10, 15, 20])
bmi['normal'] = fuzz.trimf(bmi.universe, [18, 23, 28])
bmi['high'] = fuzz.trimf(bmi.universe, [26, 35, 50])

age['young'] = fuzz.trimf(age.universe, [20, 25, 35])
age['middle'] = fuzz.trimf(age.universe, [30, 45, 60])
age['old'] = fuzz.trimf(age.universe, [55, 70, 80])

risk['low'] = fuzz.trimf(risk.universe, [0, 15, 40])
risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
risk['high'] = fuzz.trimf(risk.universe, [60, 80, 100])

# --- Define rules ---
rules = [
    ctrl.Rule(glucose['high'] & bmi['high'], risk['high']),
    ctrl.Rule(glucose['high'] & bp['high'], risk['high']),
    ctrl.Rule(glucose['high'] & age['old'], risk['high']),
    ctrl.Rule(glucose['normal'] & bmi['normal'] & bp['normal'], risk['medium']),
    ctrl.Rule(glucose['low'] & bmi['low'], risk['low']),
    ctrl.Rule(bp['low'] & age['young'], risk['low']),
    ctrl.Rule(glucose['normal'] & age['middle'], risk['medium']),
]

# --- Control system ---
risk_ctrl = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(risk_ctrl)

# --- Input sliders ---
st.subheader("Enter Patient Details")

glucose_val = st.slider("Fasting Glucose (mg/dL)", 50, 200, 120)
bp_val = st.slider("Blood Pressure (mm Hg)", 60, 180, 100)
bmi_val = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 24.0)
age_val = st.slider("Age (years)", 20, 80, 35)

# --- Predict button ---
if st.button("🩺 Compute Diabetes Risk"):
    sim.input['glucose'] = glucose_val
    sim.input['bp'] = bp_val
    sim.input['bmi'] = bmi_val
    sim.input['age'] = age_val
    sim.compute()

    risk_score = sim.output['risk']
    st.subheader(f"🎯 Diabetes Risk Score: **{risk_score:.2f}%**")

    # --- Risk message based on score ---
    if risk_score < 40:
        st.success("✅ Low Risk — Maintain a healthy lifestyle.")
        advice = "Maintain a balanced diet, stay active, and go for yearly checkups."
    elif 40 <= risk_score < 70:
        st.warning("⚠️ Medium Risk — Regular checkups recommended.")
        advice = "Monitor your sugar, control your weight, and visit a doctor if symptoms appear."
    else:
        st.error("🚨 High Risk — Immediate medical attention advised!")
        advice = "Immediate medical consultation required! Follow doctor’s advice strictly."

    st.info(f"🩺 **Health Advice:** {advice}")

    # --- Store patient result in variable for optional saving ---
    patient_data = {
        "Glucose": [glucose_val],
        "BP": [bp_val],
        "BMI": [bmi_val],
        "Age": [age_val],
        "Risk_Score": [round(risk_score, 2)],
        "Risk_Level": ["High" if risk_score >= 70 else "Medium" if risk_score >= 40 else "Low"],
        "Advice": [advice]
    }

    df_new = pd.DataFrame(patient_data)

    # --- Optional Save Button ---
    if st.button("💾 Save Patient Report"):
        csv_file = "patients.csv"
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_csv(csv_file, index=False)
        st.success("✅ Patient data saved successfully to 'patients.csv'.")

st.markdown("---")
st.caption("Project: Fuzzy Logic for Medical Diagnosis — Diabetes (PIMA Dataset)")

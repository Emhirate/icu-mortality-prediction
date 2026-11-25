import pickle
import pandas as pd
import streamlit as st
from sklearn import set_config
from transformers import BiomedicalAttrAdder
set_config(transform_output="pandas")

st.set_page_config(page_title="ICU Mortality Risk Calculator", page_icon="🏥", layout="wide")

@st.cache_resource
def load_model():
    with open("icu_mortality_model.bin", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🏥 ICU Patient Mortality Risk Calculator")
st.markdown(
    "Enter first-24-hour bedside vitals, metabolic labs, and neurological scores to estimate "
    "in-hospital mortality risk using the tuned **LightGBM** clinical pipeline (`ROC-AUC = 0.878`)."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Demographics & Admission")
    age = st.slider("Age (years)", 18, 95, 62)
    bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=65.0, value=27.5, step=0.1)
    gender = st.selectbox("Sex", ["M", "F"])
    pre_icu_los = st.number_input("Pre-ICU Hospital Stay (days)", min_value=0.0, max_value=60.0, value=0.5, step=0.1)
    admit_source = st.selectbox(
        "ICU Admission Source",
        ["Accident & Emergency", "Operating Room / Recovery", "Floor", "Other Hospital", "Other ICU"]
    )
    body_system = st.selectbox(
        "Primary Organ System (APACHE)",
        ["Cardiovascular", "Sepsis", "Respiratory", "Neurological", "Gastrointestinal", "Metabolic", "Trauma", "Genitourinary"]
    )

with col2:
    st.subheader("2. Hemodynamics & Vitals")
    hr_max = st.slider("Max Heart Rate (bpm)", 40, 200, 105)
    sbp_min = st.slider("Min Systolic BP (mmHg)", 40, 180, 95)
    diasbp_min = st.slider("Min Diastolic BP (mmHg)", 20, 120, 52)
    resprate = st.slider("Respiratory Rate (breaths/min)", 6, 55, 22)
    temp = st.number_input("Core Temperature (°C)", min_value=32.0, max_value=42.0, value=36.8, step=0.1)
    spo2_min = st.slider("Min SpO₂ (%)", 50, 100, 94)

with col3:
    st.subheader("3. Labs, Neurology & Support")
    glucose_max = st.number_input("Max Glucose (mg/dL)", min_value=40, max_value=700, value=165)
    glucose_min = st.number_input("Min Glucose (mg/dL)", min_value=20, max_value=400, value=105)
    potassium_max = st.number_input("Max Potassium (mmol/L)", min_value=2.0, max_value=8.5, value=4.3, step=0.1)
    potassium_min = st.number_input("Min Potassium (mmol/L)", min_value=1.5, max_value=7.0, value=3.8, step=0.1)
    gcs_eyes = st.selectbox("GCS Eyes (1–4)", [4, 3, 2, 1], index=0)
    gcs_motor = st.selectbox("GCS Motor (1–6)", [6, 5, 4, 3, 2, 1], index=0)
    gcs_verbal = st.selectbox("GCS Verbal (1–5)", [5, 4, 3, 2, 1], index=0)
    ventilated = st.checkbox("Mechanically Ventilated (First 24h)", value=False)
    intubated = st.checkbox("Endotracheally Intubated", value=False)

input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "gender": gender,
    "icu_admit_source": admit_source,
    "pre_icu_los_days": pre_icu_los,
    "gcs_eyes_apache": gcs_eyes,
    "gcs_motor_apache": gcs_motor,
    "gcs_verbal_apache": gcs_verbal,
    "intubated_apache": int(intubated),
    "resprate_apache": resprate,
    "temp_apache": temp,
    "ventilated_apache": int(ventilated),
    "d1_diasbp_min": diasbp_min,
    "d1_heartrate_max": hr_max,
    "d1_spo2_min": spo2_min,
    "d1_sysbp_min": sbp_min,
    "d1_glucose_max": glucose_max,
    "d1_glucose_min": glucose_min,
    "d1_potassium_max": potassium_max,
    "d1_potassium_min": potassium_min,
    "apache_3j_bodysystem": body_system,
}])

st.divider()

if st.button("Calculate Mortality Risk", type="primary", use_container_width=True):
    risk_prob = model.predict_proba(input_df)[0, 1]
    shock_idx = hr_max / (sbp_min + 1e-5)
    gcs_total = gcs_eyes + gcs_motor + gcs_verbal

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Mortality Risk", f"{risk_prob * 100:.1f}%")
    r2.metric("Modified Shock Index", f"{shock_idx:.2f}", delta="High (>0.9)" if shock_idx > 0.9 else "Normal", delta_color="inverse")
    r3.metric("Total GCS Score", f"{gcs_total} / 15")

    if risk_prob >= 0.50:
        st.error("⚠️ **High Risk Stratification:** Elevated predicted probability of in-hospital mortality.")
    elif risk_prob >= 0.20:
        st.warning("🟠 **Moderate Risk Stratification:** Close hemodynamic and metabolic monitoring advised.")
    else:
        st.success("✅ **Lower Risk Stratification:** Vitals and metabolic markers within lower-risk bounds.")

st.caption("Disclaimer: Educational Biomedical Engineering portfolio prototype trained on historical ICU data. Not intended for clinical diagnosis.")

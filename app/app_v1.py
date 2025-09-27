import streamlit as st
import pandas as pd
import joblib
import os

# --- Paths ---
MODEL_DIR = os.path.join("outputs", "metrics", "part14_final_models")

# Load models
model_highbp = joblib.load(os.path.join(MODEL_DIR, "model_highbp.pkl"))
model_diabetes = joblib.load(os.path.join(MODEL_DIR, "model_diabetes.pkl"))
model_cardio = joblib.load(os.path.join(MODEL_DIR, "model_cardio.pkl"))

# Thresholds (can later be loaded from metrics JSON/CSV if needed)
thresholds = {"highbp": 0.23, "diabetes": 0.24, "cardio": 0.24}

# Thresholds
thresholds = {
    "highbp": 0.23,
    "diabetes": 0.24,
    "cardio": 0.24
}

# Feature options
feature_options = {
    "Age Group": ['18 to 34 years', '35 to 49 years', '50 to 64 years', '65 and older'],
    "Sex at Birth": ['Male', 'Female'],
    "Marital Status": [
        'Married/Common-law',
        'Widowed/Divorced/Separated/Single, never married'
    ],
    "Perceived health ": ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'],
    "Perceived mental health ": ['Poor', 'Fair', 'Good', 'Very good', 'Excellent', 'Unknown'],
    "Satisfaction with life in general ": [
        'Very Satisfied', 'Satisfied', 'Neither satisfied nor dissatisfied', 'Dissatisfied', 'Unknown'
    ],
    "Smoking status": [
        'Non-smoker (abstainer or experimental)',
        'Former daily smoker (non-smoker now)',
        'Current daily smoker',
        'Former occasional smoker (non-smoker now)',
        'Current occasional smoker',
        'Unknown'
    ],
    "Used cannabis - 12 months": ['No', 'Yes', 'Unknown'],
    "Severity of Canabis Dependence": [
        'No cannabis use',
        'Takes cannabis but no dependence',
        'Takes cannabis & dependent on it',
        'Unknown'
    ],
    "Type of drinker": [
        'Regular drinker', 'Occasional drinker', 'Did not drink in the last 12 months'
    ],
    "Drank 5+ / 4+ drinks one occasion - frequency - 12 months": [
        'Never', 'Less than once a month', 'Once a month', '2-3 times a month', 'Once a week', 'More than once a week', 'Valid skip'
    ],
    "Total Household Income - All Sources": [
        '$80,000 or more', '$60,000 to $79,999', '$40,000 to $59,999', '$20,000 to $39,999', 'No income or less than $20,000', 'Unknown'
    ],
    "BMI classification for adults aged 18 and over (adjusted) - international": [
        'Underweight/ Normal weight', 'Overweight / Obese - Class I, II, III', 'Unknown'
    ],
    "Pain health status": ['Has usual pain or discomfort', 'No usual pain or discomfort'],
    "Has sleep apnea": ['No', 'Yes'],
    "Has high blood cholesterol / lipids": ['No', 'Yes', 'Unknown'],
    "High blood cholesterol / lipids - took medication - 1 month": ['No', 'Yes'],
    "Has chronic fatigue syndrome": ['No', 'Yes'],
    "Has a mood disorder (depression, bipolar, mania, dysthymia)": ['No', 'Yes'],
    "Has an anxiety disorder (phobia, OCD, panic)": ['No', 'Yes'],
    "Has respiratory chronic condition (asthma or COPD)": ['No', 'Yes', 'Unknown'],
    "Musculoskeletal condition (Arthritis, fibromyalgia, osteoporosis)": ['No', 'Yes', 'Unknown'],
    "Had a seasonal flu shot (excluding H1N1) - lifetime": ['No', 'Yes', 'Unknown'],
    "Seasonal flu shot - last time": [
        'Less than 1 year ago', '1 year to less than 2 years ago', '2 years ago or more', 'Valid skip', 'Unknown'
    ],
    "Usual place for immediate care for minor problem": ['Yes', 'No'],
    "Considered suicide - lifetime": ['No', 'Yes', 'Unknown'],
    "Considered suicide - last 12 months": ['No', 'Yes', 'Unknown'],
    "High blood pressure - took medication - 1 month": ['No', 'Yes'],
}

st.title(" Progressive Health Risk Predictor")
st.markdown("This tool predicts your health risks for high blood pressure, diabetes, and cardiovascular conditions.")

# --- Input Form ---
with st.form("user_input_form"):
    input_dict = {}
    st.subheader(" Lifestyle & Health Information")

    for feat, options in feature_options.items():
        friendly_label = feat.strip()
        choice = st.selectbox(friendly_label, options, key=feat)
        input_dict[feat] = choice

    bp_known = st.radio("Do you already know if you have high blood pressure?", ["Yes", "No"])
    if bp_known == "Yes":
        bp_val = st.selectbox("What is your blood pressure status?", ["Yes", "No"])
        input_dict['Has a high blood pressure'] = bp_val

    diab_known = st.radio("Do you already know if you have diabetes?", ["Yes", "No"])
    if diab_known == "Yes":
        diab_val = st.selectbox("What is your diabetes status?", ["Yes", "No"])
        input_dict['Has diabetes'] = diab_val

    submit = st.form_submit_button("🔍 Predict My Risk")

if submit:
    user_df = pd.DataFrame([input_dict])

    # --- Progressive Logic ---
    if 'Has a high blood pressure' not in user_df.columns:
        proba = model_highbp.predict_proba(user_df)[0][1]
        user_df['Has a high blood pressure'] = "Yes" if proba >= thresholds['highbp'] else "No"
        st.markdown(f" **Predicted High Blood Pressure**: {user_df['Has a high blood pressure'].iloc[0]} (prob: {proba:.2f})")

    if 'Has diabetes' not in user_df.columns:
        proba = model_diabetes.predict_proba(user_df)[0][1]
        user_df['Has diabetes'] = "Yes" if proba >= thresholds['diabetes'] else "No"
        st.markdown(f" **Predicted Diabetes**: {user_df['Has diabetes'].iloc[0]} (prob: {proba:.2f})")

    # --- Cardio Prediction ---
    cardio_proba = model_cardio.predict_proba(user_df)[0][1]
    cardio_pred = "Yes" if cardio_proba >= thresholds['cardio'] else "No"

    st.markdown("---")
    st.subheader(" Final Prediction")
    st.write(f"**Risk of Cardiovascular Condition (Heart disease or stroke)**: **{cardio_pred}**")
    st.write(f"**Predicted probability**: {cardio_proba:.2f}")

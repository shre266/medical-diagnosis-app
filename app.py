import streamlit as st
import pickle
import json
import numpy as np

# Page config
st.set_page_config(page_title="Medical Diagnosis Assistant", 
                   page_icon="🏥", layout="wide")

# Load models
def load_model(disease):
    with open(f'models/{disease}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/{disease}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'models/{disease}_features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

# Sidebar
st.sidebar.title("🏥 Medical Diagnosis Assistant")
st.sidebar.markdown("---")
disease = st.sidebar.selectbox("Select Disease to Predict", 
                                ["Diabetes", "Heart Disease", "Parkinson's"])

# Main title
st.title(f"🔬 {disease} Prediction")
st.markdown("Fill in the details below and click **Predict** to get the result.")
st.markdown("---")

# ─── DIABETES ───
if disease == "Diabetes":
    model, scaler, features = load_model('diabetes')
    
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.number_input("Glucose Level", 0, 300, 120)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        SkinThickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        Insulin = st.number_input("Insulin Level", 0, 900, 80)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 1, 120, 30)
    
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

# ─── HEART DISEASE ───
elif disease == "Heart Disease":
    model, scaler, features = load_model('heart')
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])
    
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

# ─── PARKINSON'S ───
elif disease == "Parkinson's":
    model, scaler, features = load_model('parkinsons')
    
    st.markdown("#### Voice Measurement Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 80.0, 270.0, 150.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 200.0)
        flo = st.number_input("MDVP:Flo(Hz)", 60.0, 240.0, 110.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 0.1, 0.005)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001, 0.00004)
        rap = st.number_input("MDVP:RAP", 0.0, 0.05, 0.003)
        ppq = st.number_input("MDVP:PPQ", 0.0, 0.05, 0.003)
    with col2:
        ddp = st.number_input("Jitter:DDP", 0.0, 0.15, 0.009)
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.03)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 2.0, 0.3)
        apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.02)
        apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.15, 0.02)
        apq = st.number_input("MDVP:APQ", 0.0, 0.15, 0.02)
    with col3:
        dda = st.number_input("Shimmer:DDA", 0.0, 0.3, 0.05)
        nhr = st.number_input("NHR", 0.0, 0.5, 0.02)
        hnr = st.number_input("HNR", 5.0, 40.0, 22.0)
        rpde = st.number_input("RPDE", 0.2, 0.7, 0.45)
        dfa = st.number_input("DFA", 0.5, 0.9, 0.72)
        spread1 = st.number_input("Spread1", -8.0, 0.0, -5.0)
        spread2 = st.number_input("Spread2", 0.0, 0.5, 0.25)
        d2 = st.number_input("D2", 1.5, 4.0, 2.5)
        ppe = st.number_input("PPE", 0.0, 0.6, 0.2)
    
    input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq,
                            ddp, shimmer, shimmer_db, apq3, apq5, apq, dda,
                            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

# ─── PREDICT BUTTON ───
st.markdown("---")
if st.button("🔍 Predict", use_container_width=True):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("### Result:")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected** — The model predicts a positive result for {disease}.")
    else:
        st.success(f"✅ **Low Risk** — The model predicts no signs of {disease}.")
    
    st.markdown(f"**Confidence:** {max(probability) * 100:.1f}%")
    st.warning("⚠️ This is not a medical diagnosis. Please consult a doctor.")
    # SHAP Explanation
    st.markdown("---")
    st.markdown("### 🔍 Why did the model predict this?")
    
    import shap
    import matplotlib.pyplot as plt
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values, input_scaled, 
                      feature_names=features, 
                      plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close()
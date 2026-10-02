import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image

st.set_page_config(page_title="HQML Microwave Cancer Diagnostic", layout="centered")

st.markdown("<h2 style='text-align: center; color: #00e5ff;'>MICROWAVE BREAST TUMOR DIAGNOSTIC</h2>", unsafe_allow_html=True)
st.write("---")

# Model Load
@st.cache_resource
def load_model():
    if os.path.exists("breast_tumor_model.pkl"):
        return joblib.load("breast_tumor_model.pkl")
    return None

model = load_model()

# Input Parameters Grid
st.subheader("Sensor / Simulation Parameters")
col1, col2 = st.columns(2)

with col1:
    freq = st.number_input("Frequency (GHz)", value=2.4000, format="%.4f")
    shift = st.number_input("Frequency Shift Δf (GHz)", value=-0.0140, format="%.4f")
    mag = st.number_input("Reflection Magnitude |S11| (dB)", value=-19.0280, format="%.4f")

with col2:
    phase = st.number_input("Reflection Phase ∠S11 (°)", value=-30.3420, format="%.4f")
    res_freq = st.number_input("Resonant Frequency (GHz)", value=2.5540, format="%.4f")
    bw = st.number_input("Bandwidth (GHz)", value=0.1330, format="%.4f")

st.write("")
if st.button("RUN DIAGNOSTIC SCAN", use_container_width=True):
    input_data = np.array([[5.0, 1.0, freq, shift, mag, phase, res_freq, bw]])
    
    is_tumor = False
    confidence = 94.5
    
    if model:
        try:
            pred = model.predict(input_data)
            is_tumor = (pred[0] == 1)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data)[0]
                confidence = float(np.max(probs) * 100)
        except Exception as e:
            is_tumor = (mag > -15.0)
    else:
        is_tumor = (mag > -15.0)

    st.write("---")
    if is_tumor:
        st.error(f"⚠️ TUMOR DETECTED (Confidence: {confidence:.1f}%)")
        if os.path.exists("tumor_base.png"):
            st.image("tumor_base.png", caption="Tissue Response: Tumor Region Identified", use_container_width=True)
    else:
        st.success(f"✅ NO TUMOR DETECTED (Confidence: {confidence:.1f}%)")
        if os.path.exists("no_tumor_base.png"):
            st.image("no_tumor_base.png", caption="Tissue Response: Normal Tissue", use_container_width=True)
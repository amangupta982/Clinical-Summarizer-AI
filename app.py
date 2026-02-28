import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

## --- BACKEND LOGIC --- ##

def process_clinical_text(raw_text):
    """Uses NLTK to clean and extract key sentences."""
    if not raw_text: 
        return ""
    sentences = sent_tokenize(raw_text)
    # Filter for clinical keywords to reduce noise
    keywords = ['patient', 'history', 'diagnosis', 'pain', 'treatment', 'stable', 'denies']
    important = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return " ".join(important[:5]) 

def analyze_vitals(df):
    """Identifies trends and returns raw metrics for display."""
    avg_hr = df['HeartRate'].mean()
    avg_spo2 = df['SpO2'].mean()
    
    status = "Normal"
    if avg_hr > 100: status = "Tachycardic"
    elif avg_hr < 60: status = "Bradycardic"
    
    return avg_hr, avg_spo2, status

def get_actionable_insights(abnormal_labs, vital_status):
    """Heuristic-based clinical logic for 'Next Steps'."""
    actions = []
    if "Hemoglobin" in abnormal_labs:
        actions.append("Consider CBC re-check and iron studies.")
    if "WBC" in abnormal_labs:
        actions.append("Monitor for signs of infection/sepsis.")
    if vital_status != "Normal":
        actions.append("ECHO/ECG recommended for cardiac evaluation.")
    if not actions:
        actions.append("Continue routine observation.")
    return actions

## --- STREAMLIT UI --- ##

st.set_page_config(page_title="Pro Clinical Summarizer", layout="wide")
st.title("🏥 Clinical Multi-Modal AI Summarizer")
st.markdown("Synthesizing **Unstructured Notes**, **Laboratory Data**, and **Time-Series Vitals**.")

# Layout for Inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Physician Notes")
    user_notes = st.text_area("Input clinical narrative...", height=250, 
                             placeholder="Paste doctor's notes here...")

    st.subheader("2. Vitals Log (Live Feed)")
    # Generate data
    chart_data = pd.DataFrame(
        np.random.randint(60, 115, size=(12, 2)),
        columns=['HeartRate', 'SpO2']
    )
    st.line_chart(chart_data)

with col2:
    st.subheader("3. Laboratory Results")
    uploaded_file = st.file_uploader("Upload Lab CSV", type=["csv"])
    if uploaded_file:
        lab_df = pd.read_csv(uploaded_file)
    else:
        # Default testing data
        lab_df = pd.DataFrame({
            'Test': ['Hemoglobin', 'WBC', 'Creatinine', 'Glucose'],
            'Result': [10.5, 14.2, 0.9, 115.0],
            'Unit': ['g/dL', '10^3/uL', 'mg/dL', 'mg/dL'],
            'Status': ['Low', 'High', 'Normal', 'High']
        })
    st.dataframe(lab_df, use_container_width=True)

st.divider()

# --- SUMMARY GENERATION --- #

if st.button("Generate Comprehensive Clinical Report", type="primary"):
    if not user_notes:
        st.warning("Please provide clinical notes to generate a summary.")
    else:
        with st.spinner("Processing Multi-Modal Data..."):
            # Process Modalities
            cleaned_text = process_clinical_text(user_notes)
            avg_hr, avg_spo2, hr_status = analyze_vitals(chart_data)
            abnormal_labs = lab_df[lab_df['Status'] != 'Normal']['Test'].tolist()
            recommendations = get_actionable_insights(abnormal_labs, hr_status)

            # 1. High-Level Metrics Row
            st.subheader("Patient Snapshot")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Heart Rate", f"{avg_hr:.1f} bpm", delta=hr_status, delta_color="inverse")
            m2.metric("Avg SpO2", f"{avg_spo2:.1f}%", delta="Normal" if avg_spo2 > 94 else "Low")
            m3.metric("Abnormal Labs", len(abnormal_labs), delta="Attention Required" if abnormal_labs else None)

            # 2. Color-coded Integrated Summary
            severity_color = "error" if (len(abnormal_labs) > 2 or hr_status != "Normal") else "info"
            
            with st.container():
                if severity_color == "error":
                    st.error("### 🚨 Clinical Summary: High Priority")
                else:
                    st.info("### 📋 Clinical Summary: Stable")
                
                # Final Combined Output
                st.markdown(f"**Patient History & Presenting Illness:** {cleaned_text}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Significant Lab Findings:**")
                    if abnormal_labs:
                        for lab in abnormal_labs:
                            st.write(f"⚠️ {lab} is Abnormal")
                    else:
                        st.write("✅ No abnormal lab values detected.")
                
                with c2:
                    st.markdown("**Actionable Plan:**")
                    for rec in recommendations:
                        st.write(f"- {rec}")

            # 3. Download Feature
            final_report = f"Summary: {cleaned_text}\nLabs: {abnormal_labs}\nVitals: {hr_status}"
            st.download_button("Download Report as TXT", final_report, file_name="clinical_summary.txt")
import pandas as pd
import nltk
import numpy as np
import streamlit as st

# Updated NLTK download logic for Streamlit Cloud
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

## --- BACKEND LOGIC --- ##

def process_clinical_text(raw_text):
    if not raw_text: 
        return ""
    sentences = sent_tokenize(raw_text)
    keywords = ['patient', 'history', 'diagnosis', 'pain', 'treatment', 'stable', 'denies', 'presents']
    important = [s for s in sentences if any(k in s.lower() for k in keywords)]
    
    # --- NEW FALLBACK LOGIC ---
    if not important:
        return "❌ No relevant clinical information detected in the notes. Please ensure notes contain clinical context."
        
    return " ".join(important[:5])

def analyze_sentiment(text):
    if not text:
        return "Neutral"
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    if score['neg'] > 0.15 or any(word in text.lower() for word in ["acute", "distressed", "severe", "emergency"]):
        return "Urgent/Critical"
    return "Routine/Stable"

def generate_scenario_vitals(scenario):
    if scenario == "Sepsis Risk":
        hr = np.random.randint(105, 130, size=(12))   
        spo2 = np.random.randint(89, 94, size=(12))   
    elif scenario == "Cardiac Distress":
        hr = np.random.randint(45, 58, size=(12))     
        spo2 = np.random.randint(85, 92, size=(12))   
    else: 
        hr = np.random.randint(65, 85, size=(12))
        spo2 = np.random.randint(96, 100, size=(12))
    return pd.DataFrame({'HeartRate': hr, 'SpO2': spo2})

def analyze_vitals(df):
    avg_hr = df['HeartRate'].mean()
    avg_spo2 = df['SpO2'].mean()
    
    status = "Normal"
    if avg_hr > 100: status = "Tachycardic"
    elif avg_hr < 60: status = "Bradycardic"
    
    return avg_hr, avg_spo2, status

def get_actionable_insights(abnormal_labs, vital_status, text_sentiment):
    actions = []
    if "WBC" in abnormal_labs and vital_status == "Tachycardic":
        actions.append("🚨 HIGH ALERT: Possible Sepsis detected (Elevated WBC + Tachycardia). Initiate immediate protocol.")
    
    if text_sentiment == "Urgent/Critical" and vital_status != "Normal":
        actions.append("⚠️ Patient narrative indicates acute distress aligning with abnormal vitals. Immediate physician review required.")

    if "Hemoglobin" in abnormal_labs:
        actions.append("Consider CBC re-check and iron studies for low Hemoglobin.")
    if "WBC" in abnormal_labs and not any("Sepsis" in a for a in actions):
        actions.append("Monitor closely for signs of localized infection.")
    if vital_status != "Normal" and not any("HIGH ALERT" in a for a in actions):
        actions.append(f"ECHO/ECG recommended to evaluate {vital_status} state.")
        
    if not actions:
        actions.append("Continue routine observation and standard care protocols.")
        
    return actions

def style_lab_status(val):
    """Colors the dataframe cells based on clinical status."""
    if val in ['High', 'Low']:
        return 'color: #ff4b4b; font-weight: bold;' # Red
    elif val == 'Normal':
        return 'color: #09ab3b; font-weight: bold;' # Green
    return ''

## --- STREAMLIT UI --- ##

st.set_page_config(page_title="Pro Clinical Summarizer", layout="wide")
st.title("🏥 Clinical Multi-Modal AI Summarizer")
st.markdown("Synthesizing **Unstructured Notes**, **Laboratory Data**, and **Time-Series Vitals** into dynamic, context-aware insights.")

with st.sidebar:
    st.header("⚙️ Simulation Controls")
    st.info("Select a scenario to dynamically alter the patient's vital signs and test the model's cross-modality logic.")
    selected_scenario = st.selectbox("Patient Scenario", ["Baseline / Normal", "Sepsis Risk", "Cardiac Distress"])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Physician Notes")
    user_notes = st.text_area("Input clinical narrative...", height=250, 
                             placeholder="Paste doctor's notes here...")

    st.subheader("2. Vitals Log (Live Feed)")
    chart_data = generate_scenario_vitals(selected_scenario)
    st.line_chart(chart_data)

with col2:
    st.subheader("3. Laboratory Results")
    uploaded_file = st.file_uploader("Upload Lab CSV", type=["csv"])
    if uploaded_file:
        lab_df = pd.read_csv(uploaded_file)
    else:
        lab_df = pd.DataFrame({
            'Test': ['Hemoglobin', 'WBC', 'Creatinine', 'Glucose'],
            'Result': [10.5, 14.2, 0.9, 115.0],
            'Unit': ['g/dL', '10^3/uL', 'mg/dL', 'mg/dL'],
            'Status': ['Low', 'High', 'Normal', 'High']
        })
    # Apply the color styling to the dataframe before displaying it
    st.dataframe(lab_df.style.map(style_lab_status, subset=['Status']), use_container_width=True)

st.divider()

# --- SUMMARY GENERATION --- #

if st.button("Generate Comprehensive Clinical Report", type="primary"):
    if not user_notes:
        st.warning("Please provide clinical notes to generate a summary.")
    else:
        with st.spinner("Processing Multi-Modal Data..."):
            cleaned_text = process_clinical_text(user_notes)
            text_sentiment = analyze_sentiment(user_notes)
            avg_hr, avg_spo2, hr_status = analyze_vitals(chart_data)
            abnormal_labs = lab_df[lab_df['Status'] != 'Normal']['Test'].tolist()
            recommendations = get_actionable_insights(abnormal_labs, hr_status, text_sentiment)

            st.subheader("Patient Snapshot")
            m1, m2, m3, m4 = st.columns(4)
            
            # Dynamic Metric Color Logic
            hr_color = "normal" if hr_status == "Normal" else "inverse"
            spo2_color = "normal" if avg_spo2 >= 94 else "inverse"
            lab_color = "inverse" if abnormal_labs else "normal"
            tone_color = "inverse" if text_sentiment == "Urgent/Critical" else "normal"

            m1.metric("Avg Heart Rate", f"{avg_hr:.1f} bpm", delta=hr_status, delta_color=hr_color)
            m2.metric("Avg SpO2", f"{avg_spo2:.1f}%", delta="Normal" if avg_spo2 >= 94 else "Low", delta_color=spo2_color)
            m3.metric("Abnormal Labs", len(abnormal_labs), delta="Attention Required" if abnormal_labs else "All Clear", delta_color=lab_color)
            m4.metric("Clinical Tone", text_sentiment, delta="Warning" if text_sentiment == "Urgent/Critical" else "Stable", delta_color=tone_color)

            severity_color = "error" if (len(abnormal_labs) > 2 or hr_status != "Normal" or text_sentiment == "Urgent/Critical") else "info"
            
            with st.container():
                if severity_color == "error":
                    st.error("### 🚨 Clinical Summary: High Priority")
                else:
                    st.info("### 📋 Clinical Summary: Stable")
                
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

            final_report = f"Summary: {cleaned_text}\nClinical Tone: {text_sentiment}\nLabs: {abnormal_labs}\nVitals: {hr_status}"
            st.download_button("Download Report as TXT", final_report, file_name="clinical_summary.txt")
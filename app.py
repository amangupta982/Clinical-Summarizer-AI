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

## --- STREAMLIT UI (FINAL PREMIUM VERSION) --- ##

st.set_page_config(page_title="Clinical AI Dashboard", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #111827);
}
section[data-testid="stSidebar"] {
    background-color: #0b1220;
}
h1, h2, h3 {
    font-weight: 600;
}
.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff784b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    border: none;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🏥 Clinical Multi-Modal AI Summarizer")
st.markdown(
    "Synthesizing **Unstructured Notes**, **Laboratory Data**, and **Vitals** into unified AI-powered clinical intelligence."
)

st.markdown("### 🧭 Workflow")
st.markdown("""
**Step 1:** Enter Physician Notes  
**Step 2:** Review Vitals  
**Step 3:** Upload Lab Report  
**Step 4:** Generate AI Summary  
""")

st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🧠 About This AI System")
    st.markdown("""
- Clinical Notes (NLP Processing)  
- Lab Abnormality Detection  
- Vitals Trend Analysis  
- Heuristic Fusion Engine  
""")

    st.divider()

    st.markdown("## ⚙ Simulation Controls")
    selected_scenario = st.selectbox(
        "Patient Scenario",
        ["Baseline / Normal", "Sepsis Risk", "Cardiac Distress"]
    )

# ---------- MAIN LAYOUT ----------
left_col, right_col = st.columns([1.4, 1])

# ======================================================
# LEFT PANEL (INPUTS)
# ======================================================
with left_col:

    # Step 1
    st.markdown("## 📝 Step 1: Enter Physician Notes")
    user_notes = st.text_area(
        "Input clinical narrative",
        height=220,
        placeholder="Paste doctor's notes here..."
    )

    st.divider()

    # Step 2
    st.markdown("## ❤️ Step 2: Review Vitals")
    chart_data = generate_scenario_vitals(selected_scenario)
    st.line_chart(chart_data)

    st.divider()

    # Step 3
    st.markdown("## 📊 Step 3: Upload Laboratory Report")

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

    st.dataframe(
        lab_df.style.map(style_lab_status, subset=['Status']),
        use_container_width=True
    )

# ======================================================
# RIGHT PANEL (SNAPSHOT + GENERATE + SUMMARY)
# ======================================================
with right_col:

    st.markdown("## 📊 Patient Snapshot")

    avg_hr, avg_spo2, hr_status = analyze_vitals(chart_data)
    abnormal_labs = lab_df[lab_df['Status'] != 'Normal']['Test'].tolist()
    text_sentiment = analyze_sentiment(user_notes) if user_notes else "Neutral"

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    m1.metric("Avg HR", f"{avg_hr:.1f} bpm", hr_status)
    m2.metric("Avg SpO2", f"{avg_spo2:.1f}%")
    m3.metric("Abnormal Labs", len(abnormal_labs))
    m4.metric("Clinical Tone", text_sentiment)

    st.divider()

    st.markdown("## 🚀 Generate Clinical Report")

    generate = st.button("Generate Comprehensive Clinical Report")

    # ---------- RESULT AREA ----------
    result_container = st.container()

    if generate:

        if not user_notes:
            result_container.warning(
                "⚠ Please provide clinical notes before generating report."
            )

        else:
            with result_container:
                with st.spinner("Processing Multi-Modal Data..."):

                    cleaned_text = process_clinical_text(user_notes)
                    text_sentiment = analyze_sentiment(user_notes)
                    avg_hr, avg_spo2, hr_status = analyze_vitals(chart_data)
                    abnormal_labs = lab_df[lab_df['Status'] != 'Normal']['Test'].tolist()
                    recommendations = get_actionable_insights(
                        abnormal_labs, hr_status, text_sentiment
                    )

                    st.divider()
                    st.markdown("## 📋 Clinical Summary")

                    severity = (
                        len(abnormal_labs) > 2 or
                        hr_status != "Normal" or
                        text_sentiment == "Urgent/Critical"
                    )

                    if severity:
                        st.error("🚨 High Priority Case")
                    else:
                        st.success("✅ Stable Condition")

                    st.markdown(f"**Patient Summary:** {cleaned_text}")

                    colA, colB = st.columns(2)

                    with colA:
                        st.markdown("### ⚠ Significant Lab Findings")
                        if abnormal_labs:
                            for lab in abnormal_labs:
                                st.write(f"- {lab} Abnormal")
                        else:
                            st.write("No abnormal lab values.")

                    with colB:
                        st.markdown("### 💡 Actionable Plan")
                        for rec in recommendations:
                            st.write(f"- {rec}")

                    final_report = f"""
Summary: {cleaned_text}
Clinical Tone: {text_sentiment}
Labs: {abnormal_labs}
Vitals: {hr_status}
"""

                    st.download_button(
                        "Download Clinical Report",
                        final_report,
                        file_name="clinical_summary.txt"
                    )
import re
import pandas as pd
import nltk
import numpy as np
import streamlit as st
import datetime
import io

# ── NLTK bootstrap ─────────────────────────────────────────────────────────────
for resource, path in [
    ("punkt",        "tokenizers/punkt_tab"),
    ("punkt_tab",    "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import sent_tokenize

# ── Import new engine modules ──────────────────────────────────────────────────
from ai_engine import (
    CLINICAL_KEYWORDS, URGENCY_KEYWORDS, SAMPLE_NOTES,
    process_clinical_text, analyze_sentiment,
    HybridReasoningEngine, ContextCorrelator,
)
from lab_analyzer import (
    LAB_REFERENCE_RANGES, DEFAULT_LABS,
    auto_flag_labs, compute_lab_severity,
    get_lab_interpretation, get_detailed_lab_analysis,
)
from vitals_engine import (
    generate_scenario_vitals, analyze_vitals,
    TemporalTracker,
)
from insight_generator import (
    get_actionable_insights,
    CompositeRiskScorer, EarlyWarningSystem,
    ClinicalDecisionAssistant, generate_structured_report,
)

# ── PDF export (optional) ─────────────────────────────────────────────────────
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ── MTSamples-derived sample notes ────────────────────────────────────────────
# Re-exported from ai_engine for backward compatibility — also kept here inline.
SAMPLE_NOTES = {
    "Sepsis (ICU)": (
        "Patient is a 67-year-old male presenting with acute fever of 39.8°C, rigors, and confusion. "
        "History of type 2 diabetes mellitus and chronic kidney disease stage 3. "
        "Patient complains of severe dysuria and flank pain for the past 3 days. "
        "Diagnosis: urosepsis secondary to complicated UTI. "
        "Treatment initiated: IV piperacillin-tazobactam, aggressive fluid resuscitation, "
        "blood cultures drawn. Patient denies recent travel or sick contacts. "
        "Examination reveals hypotension (BP 88/54), tachycardia (HR 124), temperature 39.6°C."
    ),
    "Cardiac Distress": (
        "Patient is a 58-year-old female presenting with acute onset chest pain radiating to the left arm, "
        "onset 2 hours prior to admission. History of hypertension, hyperlipidemia, and smoking (30 pack-years). "
        "Patient denies shortness of breath at rest but complains of exertional dyspnea. "
        "Diagnosis: suspected NSTEMI, rule out unstable angina. "
        "Treatment: aspirin 325mg, nitroglycerin sublingual, heparin infusion initiated. "
        "ECG shows ST depression in leads V4-V6. Troponin I elevated at 2.4 ng/mL."
    ),
    "Routine Post-Op": (
        "Patient is a 45-year-old male, post-operative day 2 following elective laparoscopic cholecystectomy. "
        "Patient denies fever, nausea, or vomiting. History of mild hypertension, well controlled on lisinopril. "
        "Patient complains of mild incisional pain, rated 3/10, managed with oral acetaminophen. "
        "Examination: wound sites clean and dry, no erythema or discharge. Bowel sounds present. "
        "Assessment: stable post-operative course. Treatment: continue current analgesia, advance diet, "
        "ambulation encouraged. Discharge planned for tomorrow if remains afebrile and tolerates diet."
    ),
}


def style_lab_status(val):
    if val == "High":
        return "color: #ff4b4b; font-weight: bold;"
    if val == "Low":
        return "color: #ff8c00; font-weight: bold;"
    if val == "Normal":
        return "color: #09ab3b; font-weight: bold;"
    return ""


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Clinical Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import premium font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global font (exclude Material Icon elements) ── */
    html, body,
    h1, h2, h3, h4, h5, h6, p, span, div, li, td, th, label, input, textarea, button {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header gradient banner ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(78,140,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Metric cards — glassmorphism ── */
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.07);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* ── Alert boxes ── */
    .alert-critical {
        border-left: 4px solid #ff4b4b;
        background: rgba(255, 75, 75, 0.08);
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .alert-warning {
        border-left: 4px solid #ff8c00;
        background: rgba(255, 140, 0, 0.08);
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .alert-info {
        border-left: 4px solid #4e8cff;
        background: rgba(78, 140, 255, 0.08);
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .alert-success {
        border-left: 4px solid #09ab3b;
        background: rgba(9, 171, 59, 0.08);
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* ── Risk gauge ── */
    .risk-gauge-container {
        text-align: center;
        padding: 1.5rem;
    }
    .risk-gauge {
        position: relative;
        width: 200px;
        height: 110px;
        margin: 0 auto;
        overflow: hidden;
    }
    .risk-gauge-bg {
        width: 200px;
        height: 100px;
        border-radius: 100px 100px 0 0;
        background: conic-gradient(
            from 0.75turn at 50% 100%,
            #09ab3b 0deg,
            #ffcc00 90deg,
            #ff8c00 135deg,
            #ff4b4b 180deg
        );
        position: absolute;
        top: 0;
        left: 0;
    }
    .risk-gauge-mask {
        width: 160px;
        height: 80px;
        border-radius: 80px 80px 0 0;
        background: var(--background-color, #0e1117);
        position: absolute;
        top: 20px;
        left: 20px;
    }
    .risk-gauge-needle {
        width: 3px;
        height: 70px;
        background: #fff;
        position: absolute;
        bottom: 0;
        left: 50%;
        transform-origin: bottom center;
        border-radius: 2px;
        transition: transform 0.6s ease;
    }
    .risk-gauge-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .risk-gauge-label {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* ── Severity bar ── */
    .severity-bar-track {
        height: 8px;
        background: rgba(128,128,128,0.15);
        border-radius: 4px;
        overflow: hidden;
        margin: 2px 0;
    }
    .severity-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* ── Section cards ── */
    .section-card {
        background: rgba(128,128,128,0.04);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(128,128,128,0.1);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(128,128,128,0.6);
        margin-bottom: 0.6rem;
    }

    /* ── Early warning pulse animation ── */
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .ew-alert {
        animation: pulse-alert 2s ease-in-out infinite;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }
    .ew-critical {
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    .ew-warning {
        background: rgba(255, 140, 0, 0.12);
        border: 1px solid rgba(255, 140, 0, 0.3);
    }
    .ew-info {
        background: rgba(78, 140, 255, 0.12);
        border: 1px solid rgba(78, 140, 255, 0.3);
    }

    /* ── Correlation badge ── */
    .corr-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .corr-strong { background: rgba(255,75,75,0.15); color: #ff4b4b; }
    .corr-moderate { background: rgba(255,140,0,0.15); color: #ff8c00; }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────────────────────────
if "temporal_tracker" not in st.session_state:
    st.session_state.temporal_tracker = TemporalTracker()

# ── Initialize engine instances ───────────────────────────────────────────────
reasoning_engine  = HybridReasoningEngine()
context_correlator = ContextCorrelator()
risk_scorer       = CompositeRiskScorer()
warning_system    = EarlyWarningSystem()
decision_assistant = ClinicalDecisionAssistant()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    st.info(
        "Select a scenario to dynamically alter the patient's vital signs "
        "and test the model's cross-modality logic."
    )
    selected_scenario = st.selectbox(
        "Patient Scenario",
        ["Baseline / Normal", "Sepsis Risk", "Cardiac Distress"],
    )

    st.divider()
    st.subheader("📋 Sample Clinical Notes")
    st.caption("Load a real de-identified note pattern from MTSamples:")
    selected_sample = st.selectbox(
        "Load sample note",
        ["— select —"] + list(SAMPLE_NOTES.keys()),
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("🔬 Lab Reference Ranges")
    with st.expander("View WHO standard ranges"):
        for test, ref in LAB_REFERENCE_RANGES.items():
            st.caption(f"**{test}**: {ref['low']}–{ref['high']} {ref['unit']}")

    st.divider()
    st.subheader("📊 Temporal Data")
    tracker = st.session_state.temporal_tracker
    st.caption(f"Readings tracked: **{len(tracker.history)}**")
    if st.button("🗑️ Reset Temporal History", use_container_width=True):
        st.session_state.temporal_tracker = TemporalTracker()
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ **Research prototype only.**  \n"
        "Not for clinical use. Always consult qualified healthcare professionals."
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Clinical Intelligence Dashboard</h1>
    <p>Next-generation clinical decision support — synthesizing unstructured notes,
    laboratory data, and time-series vitals into actionable, context-aware intelligence.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PATIENT INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📝 PATIENT INPUT</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Physician Notes")
    prefill = SAMPLE_NOTES.get(selected_sample, "")
    user_notes = st.text_area(
        "Input clinical narrative...",
        value=prefill,
        height=250,
        placeholder=(
            "Paste doctor's notes here...\n\n"
            "Or load a sample note from the sidebar ←"
        ),
    )

    st.subheader("2. Vitals Log (Live Feed)")
    chart_data = generate_scenario_vitals(selected_scenario)
    st.line_chart(chart_data, height=200, use_container_width=True)
    st.caption(f"Simulated 12-point time series · Scenario: **{selected_scenario}**")

with col2:
    st.subheader("3. Laboratory Results")
    uploaded_file = st.file_uploader(
        "Upload Lab CSV",
        type=["csv"],
        help="CSV with columns: Test, Result, Unit. Status auto-computed from WHO ranges.",
    )
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        lab_df = auto_flag_labs(raw_df)
        st.caption("✅ Status auto-computed from WHO reference ranges.")
    else:
        lab_df = DEFAULT_LABS.copy()
        st.caption("Using default demo labs. Upload your own CSV to analyse real data.")

    # Show critical flag visually
    st.dataframe(
        lab_df[["Test", "Result", "Unit", "Status"]].style.map(style_lab_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    # Download sample CSV
    sample_csv = pd.DataFrame({
        "Test":   list(LAB_REFERENCE_RANGES.keys()),
        "Result": [14.0, 8.5, 1.0, 90.0, 250, 140, 4.0, 30],
        "Unit":   [r["unit"] for r in LAB_REFERENCE_RANGES.values()],
    }).to_csv(index=False)
    st.download_button(
        "⬇ Download sample lab CSV",
        sample_csv,
        file_name="sample_labs.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ── Generate button ────────────────────────────────────────────────────────────
if st.button("🔍 Generate Comprehensive Clinical Intelligence Report", type="primary", use_container_width=True):
    if not user_notes.strip():
        st.warning("Please provide clinical notes to generate a summary.")
    else:
        with st.spinner("Processing Multi-Modal Data... Analyzing across all modalities..."):
            # ── Core analysis (existing logic, now from modules) ──────────────
            cleaned_text    = process_clinical_text(user_notes)
            text_sentiment  = analyze_sentiment(user_notes)
            avg_hr, avg_spo2, hr_status = analyze_vitals(chart_data)
            lab_df_flagged  = auto_flag_labs(lab_df)
            abnormal_labs   = lab_df_flagged[lab_df_flagged["Status"] != "Normal"]["Test"].tolist()
            recommendations = get_actionable_insights(abnormal_labs, hr_status, text_sentiment)

            # ── NEW: Advanced intelligence ────────────────────────────────────
            lab_analyses     = get_detailed_lab_analysis(lab_df)
            lab_severities   = [la["severity"] for la in lab_analyses if la["severity"] > 0]
            critical_labs    = [la for la in lab_analyses if la["critical"]]

            # Risk scoring
            risk_score, risk_label, risk_triggers = risk_scorer.compute(
                text_sentiment, hr_status, abnormal_labs, avg_spo2, lab_severities
            )

            # Hybrid reasoning
            clinical_summary = reasoning_engine.generate_clinical_summary(
                cleaned_text, hr_status, avg_hr, avg_spo2,
                abnormal_labs, text_sentiment, lab_analyses
            )
            confidence = reasoning_engine.compute_confidence(
                text_sentiment, hr_status, abnormal_labs, critical_labs
            )

            # Early warnings
            early_warnings = warning_system.evaluate(
                text_sentiment, hr_status, abnormal_labs, avg_spo2
            )

            # Context correlations
            correlations = context_correlator.detect_correlations(
                user_notes, hr_status, abnormal_labs
            )

            # Clinical decision support
            suggestions = decision_assistant.generate_suggestions(
                text_sentiment, hr_status, abnormal_labs,
                avg_spo2, avg_hr, lab_analyses
            )

            # Temporal tracking
            tracker = st.session_state.temporal_tracker
            tracker.record(avg_hr, avg_spo2, hr_status)
            trend_summary = tracker.get_trend_summary()
            trend_data    = tracker.detect_worsening()

            # Risk explanation
            risk_explanation = reasoning_engine.explain_risk(
                risk_label, risk_score, risk_triggers
            )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2: AI INSIGHTS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">🧠 AI INSIGHTS</div>', unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        st.subheader("Patient Snapshot")
        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Avg Heart Rate", f"{avg_hr:.0f} bpm", delta=hr_status,
            delta_color="normal" if hr_status == "Normal" else "inverse",
        )
        m2.metric(
            "Avg SpO₂", f"{avg_spo2:.1f}%",
            delta="Normal" if avg_spo2 >= 94 else "Low ⚠",
            delta_color="normal" if avg_spo2 >= 94 else "inverse",
        )
        m3.metric(
            "Abnormal Labs", len(abnormal_labs),
            delta="Attention Required" if abnormal_labs else "All Clear",
            delta_color="inverse" if abnormal_labs else "normal",
        )
        m4.metric(
            "Clinical Tone", text_sentiment,
            delta="Warning" if text_sentiment == "Urgent/Critical" else "Stable",
            delta_color="inverse" if text_sentiment == "Urgent/Critical" else "normal",
        )
        m5.metric(
            "Confidence", f"{confidence}%",
            delta="High" if confidence >= 70 else "Moderate",
            delta_color="normal" if confidence >= 70 else "off",
        )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3: RISK MONITOR
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown('<div class="section-title">⚠️ RISK MONITOR</div>', unsafe_allow_html=True)

        risk_col1, risk_col2 = st.columns([1, 2])

        with risk_col1:
            # Risk gauge visualization
            needle_angle = -90 + (risk_score / 100) * 180
            if risk_score >= 70:
                gauge_color = "#ff4b4b"
            elif risk_score >= 40:
                gauge_color = "#ff8c00"
            else:
                gauge_color = "#09ab3b"

            st.markdown(f"""
            <div class="risk-gauge-container">
                <div class="risk-gauge">
                    <div class="risk-gauge-bg"></div>
                    <div class="risk-gauge-mask"></div>
                    <div class="risk-gauge-needle" style="transform: rotate({needle_angle}deg);"></div>
                </div>
                <div class="risk-gauge-value" style="color: {gauge_color};">{risk_score}</div>
                <div class="risk-gauge-label" style="color: {gauge_color};">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with risk_col2:
            # Early warning alerts
            if early_warnings:
                st.markdown("**🚨 Active Alerts:**")
                for ew in early_warnings:
                    sev_class = f"ew-{ew['severity']}"
                    st.markdown(
                        f'<div class="ew-alert {sev_class}">'
                        f'<strong>{ew["alert"]}</strong><br/>'
                        f'<span style="font-size:0.82rem;opacity:0.85;">{ew["detail"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No early warning alerts triggered.")

            # Risk explanation
            st.markdown("**Risk Analysis:**")
            st.markdown(risk_explanation)

        # ── Temporal Trend ────────────────────────────────────────────────────
        st.divider()
        trend_col1, trend_col2 = st.columns([1, 1])

        with trend_col1:
            st.markdown("**📈 Temporal Vitals Trend**")
            history_df = tracker.get_history_df()
            if len(history_df) >= 2:
                chart_df = history_df.set_index("Reading #")
                st.line_chart(chart_df, height=220)
            else:
                st.info("Generate multiple reports to see temporal trends.")

        with trend_col2:
            st.markdown("**📊 Trend Analysis**")
            st.markdown(trend_summary)

            if trend_data["overall"] == "deteriorating":
                st.error("⚠️ Patient condition showing signs of deterioration. "
                         "Consider increasing monitoring frequency.")
            elif trend_data["overall"] == "improving":
                st.success("✅ Patient condition showing improvement trends.")

        # ── Context Correlations ──────────────────────────────────────────────
        if correlations:
            st.divider()
            st.markdown("**🔗 Cross-Modal Correlations Detected:**")
            for corr in correlations:
                badge_class = "corr-strong" if corr["strength"] == "Strong" else "corr-moderate"
                st.markdown(
                    f'<div class="section-card">'
                    f'<span class="corr-badge {badge_class}">{corr["strength"]}</span> '
                    f'<strong>{corr["condition"]}</strong><br/>'
                    f'<span style="font-size:0.88rem;">{corr["alert"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Triage banner ─────────────────────────────────────────────────────
        is_critical = (
            len(abnormal_labs) > 1
            or hr_status != "Normal"
            or text_sentiment == "Urgent/Critical"
            or avg_spo2 < 94
        )
        st.divider()
        if is_critical:
            st.error("### 🚨 Clinical Summary: High Priority")
        else:
            st.info("### 📋 Clinical Summary: Stable")

        # ── AI Clinical Summary ───────────────────────────────────────────────
        st.markdown("**🧠 AI-Generated Clinical Summary:**")
        st.markdown(clinical_summary)

        # ── Summary + Labs + Actions ──────────────────────────────────────────
        c1, c2 = st.columns([3, 2])

        with c1:
            st.markdown("**Patient History & Presenting Illness:**")
            st.markdown(f"> {cleaned_text}")

            # ── Smart Lab Findings with Severity Bars ─────────────────────────
            st.markdown("**📊 Detailed Lab Analysis:**")
            if abnormal_labs:
                for la in lab_analyses:
                    if la["status"] == "Normal":
                        continue
                    color = "#ff4b4b" if la["status"] == "High" else "#ff8c00"
                    sev = la["severity"]
                    sev_pct = sev * 10
                    if sev >= 7:
                        sev_color = "#ff4b4b"
                    elif sev >= 4:
                        sev_color = "#ff8c00"
                    else:
                        sev_color = "#ffcc00"

                    crit_tag = " 🔴 **CRITICAL**" if la["critical"] else ""

                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">⚠ {la["test"]}</span>'
                        f' — {la["result"]} {la["unit"]} ({la["status"]})'
                        f' · Ref: {la["reference_range"]}{crit_tag}',
                        unsafe_allow_html=True,
                    )

                    # Severity bar
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0 8px 0;">'
                        f'<span style="font-size:0.75rem;color:rgba(128,128,128,0.6);width:60px;">Severity</span>'
                        f'<div class="severity-bar-track" style="flex:1;">'
                        f'<div class="severity-bar-fill" style="width:{sev_pct}%;background:{sev_color};"></div>'
                        f'</div>'
                        f'<span style="font-size:0.8rem;font-weight:600;color:{sev_color};">{sev}/10</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Interpretation
                    if la["interpretation"]:
                        st.markdown(
                            f'<div style="font-size:0.84rem;color:rgba(128,128,128,0.7);'
                            f'padding:0 0 0.5rem 1rem;border-left:2px solid rgba(128,128,128,0.15);">'
                            f'💡 {la["interpretation"]}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.write("✅ No abnormal lab values detected.")

        with c2:
            st.markdown("**Actionable Plan:**")
            css_map = {
                "critical": "alert-critical",
                "warning":  "alert-warning",
                "info":     "alert-info",
                "success":  "alert-success",
            }
            for tier, action in recommendations:
                st.markdown(
                    f'<div class="{css_map.get(tier,"alert-info")}">{action}</div>',
                    unsafe_allow_html=True,
                )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4: CLINICAL DECISION ASSISTANT
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown('<div class="section-title">🧑‍⚕️ AI CLINICAL ASSISTANT</div>',
                    unsafe_allow_html=True)

        assist_col1, assist_col2, assist_col3 = st.columns(3)

        with assist_col1:
            st.markdown("**🔬 Suggested Next Steps:**")
            for step in suggestions["next_steps"]:
                st.markdown(f"• {step}")

        with assist_col2:
            st.markdown("**🩺 Possible Conditions:**")
            for cond in suggestions["possible_conditions"]:
                st.markdown(f"• {cond}")

        with assist_col3:
            st.markdown("**📋 Monitoring Plan:**")
            for mon in suggestions["monitoring"]:
                st.markdown(f"• {mon}")

        st.caption("⚠️ *This is not a diagnosis. All suggestions use safe language "
                   "and require physician review.*")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5: CLINICAL REPORT
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown('<div class="section-title">📄 CLINICAL REPORT</div>',
                    unsafe_allow_html=True)

        # Generate structured report
        report_text = generate_structured_report(
            cleaned_text=cleaned_text,
            text_sentiment=text_sentiment,
            avg_hr=avg_hr,
            avg_spo2=avg_spo2,
            vital_status=hr_status,
            abnormal_labs=abnormal_labs,
            lab_analyses=lab_analyses,
            recommendations=recommendations,
            risk_score=risk_score,
            risk_label=risk_label,
            risk_triggers=risk_triggers,
            early_warnings=early_warnings,
            suggestions=suggestions,
            correlations=correlations,
            trend_summary=trend_summary,
            confidence=confidence,
        )

        with st.expander("📄 View Full Report", expanded=False):
            st.code(report_text, language=None)

        # Download buttons
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                "⬇ Download Report as TXT",
                report_text,
                file_name="clinical_intelligence_report.txt",
                use_container_width=True,
            )

        with dl_col2:
            if HAS_FPDF:
                # Sanitize report for PDF (replace Unicode with ASCII equivalents)
                pdf_text = report_text
                pdf_replacements = {
                    "═": "=", "─": "-", "•": "-", "→": "->",
                    "⚠": "[!]", "✅": "[OK]", "🚨": "[!!]",
                    "🔴": "[!!]", "🟠": "[!]", "🟡": "[~]",
                    "🔗": "[LINK]", "📊": "", "📈": "[UP]",
                    "📉": "[DOWN]", "➡️": "[->]", "💡": "[i]",
                    "🩸": "", "🦠": "", "⚡": "", "🍬": "",
                    "🫘": "", "💧": "", "💓": "", "🧠": "",
                    "🔬": "", "🩺": "", "📋": "", "📄": "",
                    "🧑\u200d⚕️": "", "❌": "[X]",
                    "**": "", "SpO₂": "SpO2",
                    "10³/µL": "10^3/uL",
                }
                for old, new in pdf_replacements.items():
                    pdf_text = pdf_text.replace(old, new)
                # Remove any remaining non-latin-1 characters
                pdf_text = pdf_text.encode('latin-1', 'ignore').decode('latin-1')

                # Generate PDF with proper wrapping
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Courier", size=8)
                effective_w = pdf.w - pdf.l_margin - pdf.r_margin
                for line in pdf_text.split("\n"):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(effective_w, 4, line)
                pdf_bytes = pdf.output()
                st.download_button(
                    "⬇ Download Report as PDF",
                    data=bytes(pdf_bytes),
                    file_name="clinical_intelligence_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("Install `fpdf2` for PDF export: `pip install fpdf2`")
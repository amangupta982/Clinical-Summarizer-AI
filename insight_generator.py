"""
insight_generator.py — Clinical Decision Support & Risk Scoring Engine.

Migrated from app.py insight logic + new decision support features.
Provides:
  - Actionable insight generation (existing, preserved)
  - Composite Risk Scorer (0–100 risk score combining all modalities)
  - Early Warning System (sepsis, cardiac, respiratory alerts)
  - Clinical Decision Assistant (suggested next steps + possible conditions)
  - Structured report generation
"""

import datetime


# ── Actionable insights (migrated from app.py) ────────────────────────────────

def get_actionable_insights(abnormal_labs, vital_status, text_sentiment):
    """Priority-tiered cross-modal clinical recommendations."""
    actions = []

    # Tier 1 — Critical cross-modal alerts
    if "WBC" in abnormal_labs and vital_status == "Tachycardic":
        actions.append(("critical",
            "🚨 Possible Sepsis: Elevated WBC + Tachycardia detected. "
            "Initiate SIRS/Sepsis protocol immediately. Draw blood cultures."))
    if text_sentiment == "Urgent/Critical" and vital_status != "Normal":
        actions.append(("critical",
            "⚠️ Acute distress narrative aligns with abnormal vitals. "
            "Immediate physician review required."))

    # Tier 2 — Lab-driven warnings
    if "Hemoglobin" in abnormal_labs:
        actions.append(("warning",
            "🩸 Low Hemoglobin: Order CBC re-check and iron panel. "
            "Review transfusion threshold (consider if Hgb < 7 g/dL)."))
    if "WBC" in abnormal_labs and not any("Sepsis" in a[1] for a in actions):
        actions.append(("warning",
            "🦠 Elevated WBC: Monitor for localized infection. "
            "Blood culture and differential recommended."))
    if "Potassium" in abnormal_labs:
        actions.append(("warning",
            "⚡ Abnormal Potassium: Arrhythmia risk. "
            "12-lead ECG recommended. Begin electrolyte replacement protocol."))
    if "Glucose" in abnormal_labs:
        actions.append(("info",
            "🍬 Abnormal Glucose: Check HbA1c. "
            "Review current diabetic medication regimen with endocrinology."))
    if "Creatinine" in abnormal_labs:
        actions.append(("warning",
            "🫘 Elevated Creatinine: Monitor renal function (BUN, eGFR). "
            "Hold nephrotoxic agents. Urology or nephrology consult if persists."))
    if "Sodium" in abnormal_labs:
        actions.append(("warning",
            "💧 Abnormal Sodium: Assess fluid balance and hydration status. "
            "Gradual correction protocol — avoid rapid shifts."))

    # Tier 3 — Vital-sign driven
    if vital_status != "Normal" and not any(a[0] == "critical" for a in actions):
        actions.append(("warning",
            f"💓 {vital_status} detected: Continuous cardiac monitoring advised. "
            "ECHO/ECG recommended."))

    # Default
    if not actions:
        actions.append(("success",
            "✅ All parameters within acceptable range. "
            "Continue routine observation and standard care protocols."))
    return actions


# ── Composite Risk Scorer ─────────────────────────────────────────────────────

class CompositeRiskScorer:
    """
    Combines text sentiment, vital status, lab results, and SpO2 into
    a single 0–100 risk score with categorical labeling.

    Scoring methodology:
      - Text urgency:        0–25 points
      - Vital abnormality:   0–25 points
      - Lab abnormalities:   0–30 points (scaled by count + severity)
      - SpO2 status:         0–20 points
    """

    def compute(self, text_sentiment, vital_status, abnormal_labs,
                avg_spo2, lab_severities=None):
        """
        Returns: (score: int, label: str, triggers: list[str])
        """
        score = 0
        triggers = []

        # Text urgency
        if text_sentiment == "Urgent/Critical":
            score += 25
            triggers.append("Clinical narrative indicates urgent/critical condition")
        elif text_sentiment == "Routine/Stable":
            score += 5  # Minimal baseline

        # Vital status
        if vital_status == "Tachycardic":
            score += 22
            triggers.append("Tachycardia detected (HR > 100 bpm)")
        elif vital_status == "Bradycardic":
            score += 18
            triggers.append("Bradycardia detected (HR < 60 bpm)")

        # SpO2
        if avg_spo2 < 90:
            score += 20
            triggers.append(f"Critical hypoxemia (SpO₂ {avg_spo2:.1f}%)")
        elif avg_spo2 < 94:
            score += 14
            triggers.append(f"Hypoxemia (SpO₂ {avg_spo2:.1f}%)")
        elif avg_spo2 < 96:
            score += 6
            triggers.append(f"Borderline SpO₂ ({avg_spo2:.1f}%)")

        # Lab abnormalities
        n_abnormal = len(abnormal_labs)
        if n_abnormal >= 3:
            score += 25
            triggers.append(f"Multiple lab abnormalities ({n_abnormal} tests)")
        elif n_abnormal == 2:
            score += 18
            triggers.append(f"Lab abnormalities in: {', '.join(abnormal_labs)}")
        elif n_abnormal == 1:
            score += 10
            triggers.append(f"Lab abnormality: {abnormal_labs[0]}")

        # Severity bonus from lab_severities
        if lab_severities:
            max_sev = max(lab_severities) if lab_severities else 0
            if max_sev >= 7:
                score += 5
                triggers.append("At least one lab result at severe deviation")

        score = min(score, 100)

        # Categorize
        if score >= 70:
            label = "HIGH RISK"
        elif score >= 40:
            label = "MODERATE RISK"
        else:
            label = "LOW RISK"

        return score, label, triggers


# ── Early Warning System ──────────────────────────────────────────────────────

class EarlyWarningSystem:
    """
    Generates specific clinical early warning alerts by pattern-matching
    across all modalities.
    """

    ALERT_PATTERNS = {
        "sepsis_risk": {
            "conditions": lambda s, v, labs, spo2: (
                "WBC" in labs and v == "Tachycardic"
            ),
            "alert": "🔴 Sepsis Risk Increasing",
            "detail": ("Elevated WBC combined with tachycardia suggests possible "
                       "systemic infection. Consider initiating sepsis protocol (qSOFA)."),
            "severity": "critical",
        },
        "cardiac_instability": {
            "conditions": lambda s, v, labs, spo2: (
                v in ("Tachycardic", "Bradycardic") and spo2 < 94
            ),
            "alert": "🔴 Cardiac Instability Detected",
            "detail": ("Abnormal heart rate with low oxygen saturation may indicate "
                       "cardiac decompensation. Continuous telemetry recommended."),
            "severity": "critical",
        },
        "respiratory_decline": {
            "conditions": lambda s, v, labs, spo2: spo2 < 92,
            "alert": "🟠 Respiratory Decline Warning",
            "detail": ("SpO₂ below 92% indicates significant hypoxemia. "
                       "ABG recommended. Prepare for supplemental O₂ or ventilatory support."),
            "severity": "warning",
        },
        "renal_alert": {
            "conditions": lambda s, v, labs, spo2: (
                "Creatinine" in labs and "Potassium" in labs
            ),
            "alert": "🟠 Renal Function Alert",
            "detail": ("Both creatinine and potassium abnormal — possible acute kidney "
                       "injury. Nephrology consult and renal panel recommended."),
            "severity": "warning",
        },
        "metabolic_alert": {
            "conditions": lambda s, v, labs, spo2: (
                "Glucose" in labs and "Sodium" in labs
            ),
            "alert": "🟡 Metabolic Imbalance",
            "detail": ("Concurrent glucose and sodium abnormalities suggest metabolic "
                       "dysregulation. Endocrine evaluation recommended."),
            "severity": "info",
        },
        "anemia_risk": {
            "conditions": lambda s, v, labs, spo2: (
                "Hemoglobin" in labs and v == "Tachycardic"
            ),
            "alert": "🟠 Anemia with Cardiac Compensation",
            "detail": ("Low hemoglobin with tachycardia — the heart is compensating for "
                       "reduced oxygen-carrying capacity. Transfusion threshold review needed."),
            "severity": "warning",
        },
    }

    def evaluate(self, text_sentiment, vital_status, abnormal_labs, avg_spo2):
        """
        Returns list of triggered alerts:
        [{"name": str, "alert": str, "detail": str, "severity": str}]
        """
        alerts = []
        for name, pattern in self.ALERT_PATTERNS.items():
            try:
                if pattern["conditions"](text_sentiment, vital_status,
                                         abnormal_labs, avg_spo2):
                    alerts.append({
                        "name":     name,
                        "alert":    pattern["alert"],
                        "detail":   pattern["detail"],
                        "severity": pattern["severity"],
                    })
            except Exception:
                continue
        return alerts


# ── Clinical Decision Assistant ───────────────────────────────────────────────

class ClinicalDecisionAssistant:
    """
    Generates suggested next steps and possible conditions using
    safe, non-diagnostic language.
    """

    def generate_suggestions(self, text_sentiment, vital_status, abnormal_labs,
                             avg_spo2, avg_hr, lab_analyses=None):
        """
        Returns:
          {
            "next_steps": [str, ...],
            "possible_conditions": [str, ...],
            "monitoring": [str, ...],
          }
        """
        next_steps = []
        conditions = []
        monitoring = []

        # Text-based suggestions
        if text_sentiment == "Urgent/Critical":
            next_steps.append("Consider immediate physician bedside assessment")
            next_steps.append("Prepare for potential escalation to higher level of care")
            monitoring.append("Continuous vital sign monitoring (q15 min)")

        # Vital-based suggestions
        if vital_status == "Tachycardic":
            next_steps.append("Consider 12-lead ECG to rule out arrhythmia")
            next_steps.append("Evaluate fluid status and consider IV hydration")
            conditions.append("May indicate: Dehydration, Infection, Anxiety, "
                              "Cardiac Arrhythmia, or Pain Response")
            monitoring.append("Continuous cardiac telemetry")
        elif vital_status == "Bradycardic":
            next_steps.append("Consider 12-lead ECG and cardiac enzyme panel")
            next_steps.append("Review current medications for beta-blockers "
                              "or calcium channel blockers")
            conditions.append("May indicate: Medication effect, Heart block, "
                              "Hypothyroidism, or Vagal response")
            monitoring.append("Continuous cardiac monitoring with atropine available")

        # SpO2-based suggestions
        if avg_spo2 < 90:
            next_steps.append("Order arterial blood gas (ABG) immediately")
            next_steps.append("Initiate supplemental oxygen therapy")
            conditions.append("May indicate: Pneumonia, Pulmonary embolism, "
                              "ARDS, or Cardiac failure")
        elif avg_spo2 < 94:
            next_steps.append("Consider chest X-ray and ABG")
            next_steps.append("Evaluate for supplemental oxygen need")

        # Lab-based suggestions
        if "WBC" in abnormal_labs:
            next_steps.append("Order blood cultures and complete differential")
            conditions.append("May indicate: Bacterial infection, Viral illness, "
                              "or Inflammatory process")

        if "Creatinine" in abnormal_labs:
            next_steps.append("Order comprehensive metabolic panel with BUN and eGFR")
            next_steps.append("Review and hold nephrotoxic medications")
            conditions.append("May indicate: Acute kidney injury, "
                              "Chronic kidney disease, or Dehydration")

        if "Hemoglobin" in abnormal_labs:
            next_steps.append("Order reticulocyte count and iron studies")
            conditions.append("May indicate: Iron-deficiency anemia, "
                              "Chronic disease anemia, or Active bleeding")

        if "Glucose" in abnormal_labs:
            next_steps.append("Order HbA1c and consider endocrinology consult")
            monitoring.append("Blood glucose monitoring q4h")

        if "Potassium" in abnormal_labs:
            next_steps.append("Order repeat potassium with magnesium level")
            next_steps.append("Obtain 12-lead ECG to assess for cardiac effects")
            monitoring.append("Continuous telemetry for arrhythmia detection")

        # Cross-modal: WBC + Tachycardia
        if "WBC" in abnormal_labs and vital_status == "Tachycardic":
            conditions.append("Consider evaluating for: Sepsis / SIRS criteria")
            next_steps.append("Calculate qSOFA score and lactate level")

        # Default monitoring
        if not monitoring:
            monitoring.append("Standard nursing monitoring per facility protocol")
        if not next_steps:
            next_steps.append("Continue current management and reassess as needed")
        if not conditions:
            conditions.append("No specific conditions identified from current data")

        return {
            "next_steps":          next_steps,
            "possible_conditions": conditions,
            "monitoring":          monitoring,
        }


# ── Structured Report Generation ─────────────────────────────────────────────

def generate_structured_report(
    cleaned_text, text_sentiment, avg_hr, avg_spo2, vital_status,
    abnormal_labs, lab_analyses, recommendations, risk_score, risk_label,
    risk_triggers, early_warnings, suggestions, correlations=None,
    trend_summary=None, confidence=None,
):
    """
    Assemble a full structured clinical report as a list of text lines.
    Suitable for TXT export or PDF rendering.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "═" * 60,
        "   AI CLINICAL INTELLIGENCE DASHBOARD — REPORT",
        "═" * 60,
        f"Generated: {now}",
        f"Disclaimer: Research prototype only. Not for clinical use.",
        "",

        "─" * 60,
        "  PATIENT SUMMARY",
        "─" * 60,
        f"  Triage Level  : {risk_label}",
        f"  Risk Score    : {risk_score}/100",
        f"  Clinical Tone : {text_sentiment}",
        f"  Confidence    : {confidence}%" if confidence else "",
        "",

        "─" * 60,
        "  VITAL SIGNS",
        "─" * 60,
        f"  Heart Rate : {avg_hr:.0f} bpm ({vital_status})",
        f"  SpO₂       : {avg_spo2:.1f}%",
        "",
    ]

    # Temporal trend
    if trend_summary:
        lines.extend([
            "─" * 60,
            "  TEMPORAL TREND",
            "─" * 60,
            f"  {trend_summary.replace('**', '').replace('*', '')}",
            "",
        ])

    # Lab findings
    lines.extend([
        "─" * 60,
        "  LABORATORY FINDINGS",
        "─" * 60,
    ])
    if lab_analyses:
        for la in lab_analyses:
            flag = " *** CRITICAL ***" if la["critical"] else ""
            lines.append(
                f"  {la['test']}: {la['result']} {la['unit']} "
                f"[{la['status']}] (Severity: {la['severity']}/10){flag}"
            )
            if la["interpretation"]:
                lines.append(f"    → {la['interpretation']}")
    else:
        lines.append("  All values within reference ranges.")
    lines.append("")

    # Key clinical extract
    lines.extend([
        "─" * 60,
        "  KEY CLINICAL EXTRACT",
        "─" * 60,
        f"  {cleaned_text[:500]}",
        "",
    ])

    # Risk alerts
    if early_warnings:
        lines.extend([
            "─" * 60,
            "  ⚠ EARLY WARNING ALERTS",
            "─" * 60,
        ])
        for ew in early_warnings:
            lines.append(f"  {ew['alert']}")
            lines.append(f"    {ew['detail']}")
        lines.append("")

    # Risk explanation
    if risk_triggers:
        lines.extend([
            "─" * 60,
            "  RISK ANALYSIS",
            "─" * 60,
        ])
        for trigger in risk_triggers:
            lines.append(f"  • {trigger}")
        lines.append("")

    # Cross-modal correlations
    if correlations:
        lines.extend([
            "─" * 60,
            "  CROSS-MODAL CORRELATIONS",
            "─" * 60,
        ])
        for corr in correlations:
            lines.append(f"  [{corr['strength']}] {corr['condition']}")
            lines.append(f"    {corr['alert']}")
        lines.append("")

    # Clinical decision support
    if suggestions:
        lines.extend([
            "─" * 60,
            "  SUGGESTED NEXT STEPS",
            "─" * 60,
        ])
        for step in suggestions.get("next_steps", []):
            lines.append(f"  • {step}")
        lines.append("")
        lines.extend(["  POSSIBLE CONDITIONS:"])
        for cond in suggestions.get("possible_conditions", []):
            lines.append(f"  • {cond}")
        lines.append("")
        lines.extend(["  MONITORING PLAN:"])
        for mon in suggestions.get("monitoring", []):
            lines.append(f"  • {mon}")
        lines.append("")

    # Actionable plan
    lines.extend([
        "─" * 60,
        "  ACTIONABLE PLAN",
        "─" * 60,
    ])
    for tier, action in recommendations:
        lines.append(f"  [{tier.upper()}] {action}")

    lines.extend([
        "",
        "═" * 60,
        "  ⚠ Research prototype only. Not for clinical use.",
        "  Always consult qualified healthcare professionals.",
        "═" * 60,
    ])

    return "\n".join(lines)

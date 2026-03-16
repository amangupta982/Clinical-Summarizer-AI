"""
Unit tests for Clinical-Summarizer-AI core logic.
Run with: pytest tests/test_app.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    process_clinical_text,
    analyze_sentiment,
    analyze_vitals,
    get_actionable_insights,
    generate_scenario_vitals,
)


# ── process_clinical_text ──────────────────────────────────────────────────

class TestProcessClinicalText:
    def test_extracts_sentences_with_clinical_keywords(self):
        text = "Patient presents with fever. The weather is nice today. Diagnosis is influenza."
        result = process_clinical_text(text)
        assert "Patient presents with fever" in result
        assert "Diagnosis is influenza" in result

    def test_returns_fallback_for_nonclinical_text(self):
        text = "The sun is shining and birds are singing."
        result = process_clinical_text(text)
        assert "No relevant clinical information" in result

    def test_returns_empty_string_for_empty_input(self):
        assert process_clinical_text("") == ""
        assert process_clinical_text(None) == ""

    def test_limits_output_to_five_sentences(self):
        text = " ".join([
            "Patient has a history of hypertension.",
            "Patient presents with chest pain.",
            "Diagnosis is angina.",
            "Treatment with aspirin was initiated.",
            "Patient is stable and denies shortness of breath.",
            "Patient also has a history of diabetes.",
            "Pain was described as severe and radiating.",
        ])
        result = process_clinical_text(text)
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) <= 5


# ── analyze_sentiment ─────────────────────────────────────────────────────

class TestAnalyzeSentiment:
    def test_urgent_text_returns_urgent(self):
        text = "Patient is in severe distress with acute chest pain emergency."
        assert analyze_sentiment(text) == "Urgent/Critical"

    def test_routine_text_returns_stable(self):
        text = "Patient is recovering well. No complaints reported today."
        assert analyze_sentiment(text) == "Routine/Stable"

    def test_empty_text_returns_neutral(self):
        assert analyze_sentiment("") == "Neutral"
        assert analyze_sentiment(None) == "Neutral"

    def test_negative_vader_score_triggers_urgent(self):
        # Highly negative clinical language
        text = "Patient deteriorating rapidly, not responding to treatment, critical condition worsening."
        result = analyze_sentiment(text)
        assert result == "Urgent/Critical"


# ── analyze_vitals ────────────────────────────────────────────────────────

class TestAnalyzeVitals:
    def _make_df(self, hr, spo2):
        return pd.DataFrame({
            "HeartRate": [hr] * 12,
            "SpO2": [spo2] * 12
        })

    def test_normal_vitals(self):
        df = self._make_df(72, 98)
        avg_hr, avg_spo2, status = analyze_vitals(df)
        assert abs(avg_hr - 72) < 0.1
        assert status == "Normal"

    def test_tachycardia_detection(self):
        df = self._make_df(115, 97)
        _, _, status = analyze_vitals(df)
        assert status == "Tachycardic"

    def test_bradycardia_detection(self):
        df = self._make_df(50, 97)
        _, _, status = analyze_vitals(df)
        assert status == "Bradycardic"

    def test_returns_correct_averages(self):
        hr_vals = list(range(60, 72))   # mean = 65.5
        spo2_vals = list(range(96, 108))  # mean = 101.5
        df = pd.DataFrame({"HeartRate": hr_vals, "SpO2": spo2_vals})
        avg_hr, avg_spo2, _ = analyze_vitals(df)
        assert abs(avg_hr - 65.5) < 0.1
        assert abs(avg_spo2 - 101.5) < 0.1


# ── get_actionable_insights ───────────────────────────────────────────────

class TestGetActionableInsights:
    def test_sepsis_alert_triggered(self):
        actions = get_actionable_insights(["WBC"], "Tachycardic", "Routine/Stable")
        assert any("Sepsis" in a for a in actions)

    def test_urgent_critical_with_abnormal_vitals(self):
        actions = get_actionable_insights([], "Tachycardic", "Urgent/Critical")
        assert any("physician" in a.lower() or "review" in a.lower() for a in actions)

    def test_routine_stable_returns_standard_care(self):
        actions = get_actionable_insights([], "Normal", "Routine/Stable")
        assert any("routine" in a.lower() or "standard" in a.lower() for a in actions)

    def test_hemoglobin_flag(self):
        actions = get_actionable_insights(["Hemoglobin"], "Normal", "Routine/Stable")
        assert any("Hemoglobin" in a or "iron" in a.lower() for a in actions)


# ── generate_scenario_vitals ──────────────────────────────────────────────

class TestGenerateScenarioVitals:
    def test_returns_dataframe_with_correct_columns(self):
        df = generate_scenario_vitals("Baseline / Normal")
        assert "HeartRate" in df.columns
        assert "SpO2" in df.columns

    def test_returns_12_rows(self):
        for scenario in ["Baseline / Normal", "Sepsis Risk", "Cardiac Distress"]:
            df = generate_scenario_vitals(scenario)
            assert len(df) == 12, f"Expected 12 rows for {scenario}"

    def test_sepsis_produces_elevated_hr(self):
        df = generate_scenario_vitals("Sepsis Risk")
        assert df["HeartRate"].mean() > 100

    def test_cardiac_distress_produces_low_hr(self):
        df = generate_scenario_vitals("Cardiac Distress")
        assert df["HeartRate"].mean() < 65

    def test_normal_baseline_produces_normal_hr(self):
        df = generate_scenario_vitals("Baseline / Normal")
        assert 60 < df["HeartRate"].mean() < 90

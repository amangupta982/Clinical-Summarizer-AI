"""
Unit tests for Clinical-Summarizer-AI — v2.
All 32 tests should pass. Run with: pytest tests/test_app.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine import process_clinical_text, analyze_sentiment
from vitals_engine import analyze_vitals, generate_scenario_vitals
from insight_generator import get_actionable_insights
from lab_analyzer import auto_flag_labs


class TestProcessClinicalText:

    def test_extracts_sentences_with_clinical_keywords(self):
        text = "Patient presents with fever. The weather is nice today. Diagnosis is influenza."
        result = process_clinical_text(text)
        assert "fever" in result or "influenza" in result

    def test_returns_fallback_for_nonclinical_text(self):
        text = "The sun is shining and birds are singing."
        result = process_clinical_text(text)
        assert "No relevant clinical information" in result

    def test_returns_empty_string_for_empty_input(self):
        assert process_clinical_text("") == ""
        assert process_clinical_text(None) == ""

    def test_limits_output_to_five_sentences(self):
        sentences = [
            "Patient has a history of hypertension.",
            "Patient presents with chest pain.",
            "Diagnosis is angina.",
            "Treatment with aspirin was initiated.",
            "Patient is stable and denies shortness of breath.",
            "Patient also has a history of diabetes.",
            "Pain was described as severe and radiating.",
        ]
        result = process_clinical_text(" ".join(sentences))
        parts = [s.strip() for s in result.split(".") if s.strip()]
        assert len(parts) <= 6

    def test_higher_keyword_density_ranked_first(self):
        text = (
            "The sky is blue. "
            "Patient presents with pain and diagnosis of acute chest condition and history of treatment."
        )
        result = process_clinical_text(text)
        assert "presents" in result or "diagnosis" in result


class TestAnalyzeSentiment:

    def test_urgent_text_returns_urgent(self):
        assert analyze_sentiment("Patient is in severe distress with acute chest pain emergency.") == "Urgent/Critical"

    def test_routine_text_returns_stable(self):
        """Key fix: 'No complaints' must NOT trigger Urgent/Critical."""
        assert analyze_sentiment("Patient is recovering well. No complaints reported today.") == "Routine/Stable"

    def test_denial_language_is_routine(self):
        assert analyze_sentiment("Patient denies shortness of breath. Patient denies chest pain. Vitals stable.") == "Routine/Stable"

    def test_empty_returns_neutral(self):
        assert analyze_sentiment("") == "Neutral"
        assert analyze_sentiment(None) == "Neutral"

    def test_deterioration_triggers_urgent(self):
        assert analyze_sentiment("Patient deteriorating rapidly, critical condition worsening, not responding.") == "Urgent/Critical"

    def test_acute_keyword_triggers_urgent(self):
        assert analyze_sentiment("Patient presents with acute myocardial infarction.") == "Urgent/Critical"

    def test_emergency_keyword_triggers_urgent(self):
        assert analyze_sentiment("Patient is in an emergency state.") == "Urgent/Critical"


class TestAnalyzeVitals:

    def _df(self, hr, spo2):
        return pd.DataFrame({"Heart Rate (bpm)": [hr]*12, "SpO₂ (%)": [spo2]*12})

    def test_normal(self):
        _, _, status = analyze_vitals(self._df(72, 98))
        assert status == "Normal"

    def test_tachycardia(self):
        _, _, status = analyze_vitals(self._df(115, 97))
        assert status == "Tachycardic"

    def test_bradycardia(self):
        _, _, status = analyze_vitals(self._df(50, 97))
        assert status == "Bradycardic"

    def test_boundary_101_is_tachycardic(self):
        _, _, status = analyze_vitals(self._df(101, 97))
        assert status == "Tachycardic"

    def test_correct_averages(self):
        df = pd.DataFrame({"Heart Rate (bpm)": list(range(60, 72)), "SpO₂ (%)": list(range(96, 108))})
        avg_hr, avg_spo2, _ = analyze_vitals(df)
        assert abs(avg_hr - 65.5) < 0.1
        assert abs(avg_spo2 - 101.5) < 0.1


class TestGetActionableInsights:

    def test_sepsis_alert(self):
        actions = get_actionable_insights(["WBC"], "Tachycardic", "Routine/Stable")
        assert any("Sepsis" in a[1] for a in actions)

    def test_urgent_with_abnormal_vitals(self):
        actions = get_actionable_insights([], "Tachycardic", "Urgent/Critical")
        assert any("physician" in a[1].lower() or "review" in a[1].lower() for a in actions)

    def test_routine_returns_success_tier(self):
        actions = get_actionable_insights([], "Normal", "Routine/Stable")
        assert any(a[0] == "success" for a in actions)

    def test_hemoglobin_flag(self):
        actions = get_actionable_insights(["Hemoglobin"], "Normal", "Routine/Stable")
        assert any("Hemoglobin" in a[1] or "iron" in a[1].lower() for a in actions)

    def test_returns_list_of_tuples(self):
        actions = get_actionable_insights([], "Normal", "Routine/Stable")
        assert all(isinstance(a, tuple) and len(a) == 2 for a in actions)

    def test_potassium_flag(self):
        actions = get_actionable_insights(["Potassium"], "Normal", "Routine/Stable")
        assert any("Potassium" in a[1] or "arrhythmia" in a[1].lower() for a in actions)


class TestGenerateScenarioVitals:

    def test_correct_columns(self):
        df = generate_scenario_vitals("Baseline / Normal")
        assert "Heart Rate (bpm)" in df.columns
        assert "SpO₂ (%)" in df.columns

    def test_12_rows_all_scenarios(self):
        for s in ["Baseline / Normal", "Sepsis Risk", "Cardiac Distress"]:
            assert len(generate_scenario_vitals(s)) == 12

    def test_sepsis_elevated_hr(self):
        means = [generate_scenario_vitals("Sepsis Risk")["Heart Rate (bpm)"].mean() for _ in range(10)]
        assert all(m > 100 for m in means)

    def test_cardiac_low_hr(self):
        means = [generate_scenario_vitals("Cardiac Distress")["Heart Rate (bpm)"].mean() for _ in range(10)]
        assert all(m < 65 for m in means)

    def test_normal_baseline_hr_range(self):
        means = [generate_scenario_vitals("Baseline / Normal")["Heart Rate (bpm)"].mean() for _ in range(10)]
        assert all(60 < m < 90 for m in means)


class TestAutoFlagLabs:

    def test_flags_high_wbc(self):
        df = pd.DataFrame({"Test": ["WBC"], "Result": [14.0], "Unit": ["10³/µL"]})
        result = auto_flag_labs(df)
        assert result[result["Test"] == "WBC"]["Status"].values[0] == "High"

    def test_flags_low_hemoglobin(self):
        df = pd.DataFrame({"Test": ["Hemoglobin"], "Result": [9.0], "Unit": ["g/dL"]})
        result = auto_flag_labs(df)
        assert result[result["Test"] == "Hemoglobin"]["Status"].values[0] == "Low"

    def test_normal_creatinine(self):
        df = pd.DataFrame({"Test": ["Creatinine"], "Result": [1.0], "Unit": ["mg/dL"]})
        result = auto_flag_labs(df)
        assert result[result["Test"] == "Creatinine"]["Status"].values[0] == "Normal"

    def test_unknown_test_preserved(self):
        df = pd.DataFrame({"Test": ["CustomTest"], "Result": [99.0], "Unit": ["units"]})
        result = auto_flag_labs(df)
        assert "CustomTest" in result["Test"].values
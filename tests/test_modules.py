"""
Unit tests for new Clinical Intelligence modules.
Run with: pytest tests/test_modules.py -v
"""
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine import HybridReasoningEngine, ContextCorrelator
from lab_analyzer import (
    compute_lab_severity, get_lab_interpretation,
    get_detailed_lab_analysis, auto_flag_labs, LAB_REFERENCE_RANGES,
)
from vitals_engine import TemporalTracker
from insight_generator import (
    CompositeRiskScorer, EarlyWarningSystem,
    ClinicalDecisionAssistant,
)


# ── Lab Analyzer Tests ────────────────────────────────────────────────────────

class TestLabSeverity:

    def test_normal_value_scores_zero(self):
        assert compute_lab_severity("WBC", 7.0) == 0

    def test_mildly_abnormal_scores_low(self):
        sev = compute_lab_severity("WBC", 12.0)
        assert 1 <= sev <= 4

    def test_very_abnormal_scores_high(self):
        sev = compute_lab_severity("WBC", 28.0)
        assert sev >= 7

    def test_unknown_test_scores_zero(self):
        assert compute_lab_severity("UnknownTest", 99.0) == 0

    def test_critical_low_scores_ten(self):
        sev = compute_lab_severity("Hemoglobin", 5.0)
        assert sev >= 9


class TestLabInterpretation:

    def test_high_wbc_explains_infection(self):
        interp = get_lab_interpretation("WBC", "High")
        assert "infection" in interp.lower()

    def test_normal_returns_empty(self):
        assert get_lab_interpretation("WBC", "Normal") == ""

    def test_unknown_test_returns_empty(self):
        assert get_lab_interpretation("FakeTest", "High") == ""


class TestDetailedLabAnalysis:

    def test_returns_list_of_dicts(self):
        df = pd.DataFrame({"Test": ["WBC"], "Result": [14.0], "Unit": ["10³/µL"]})
        result = get_detailed_lab_analysis(df)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "severity" in result[0]
        assert "interpretation" in result[0]

    def test_analysis_includes_reference_range(self):
        df = pd.DataFrame({"Test": ["Hemoglobin"], "Result": [10.0], "Unit": ["g/dL"]})
        result = get_detailed_lab_analysis(df)
        assert result[0]["reference_range"] != ""


# ── Temporal Tracker Tests ────────────────────────────────────────────────────

class TestTemporalTracker:

    def test_record_and_history(self):
        t = TemporalTracker()
        t.record(80.0, 98.0, "Normal")
        assert len(t.history) == 1

    def test_trend_insufficient_data(self):
        t = TemporalTracker()
        t.record(80.0, 98.0, "Normal")
        result = t.detect_worsening()
        assert result["overall"] == "insufficient_data"

    def test_trend_worsening_hr(self):
        t = TemporalTracker()
        t.record(80.0, 98.0, "Normal")
        t.record(82.0, 98.0, "Normal")
        t.record(85.0, 98.0, "Normal")
        t.record(95.0, 98.0, "Normal")
        t.record(100.0, 97.0, "Tachycardic")
        t.record(110.0, 96.0, "Tachycardic")
        result = t.detect_worsening()
        assert result["hr_trend"] == "worsening"

    def test_get_trend_summary_returns_string(self):
        t = TemporalTracker()
        t.record(80.0, 98.0, "Normal")
        t.record(82.0, 97.0, "Normal")
        summary = t.get_trend_summary()
        assert isinstance(summary, str)

    def test_max_history_limit(self):
        t = TemporalTracker()
        for i in range(25):
            t.record(float(70 + i), 98.0, "Normal")
        assert len(t.history) == 20  # max_history


# ── Risk Scorer Tests ─────────────────────────────────────────────────────────

class TestCompositeRiskScorer:

    def test_low_risk_baseline(self):
        scorer = CompositeRiskScorer()
        score, label, triggers = scorer.compute("Routine/Stable", "Normal", [], 98.0)
        assert label == "LOW RISK"
        assert score < 40

    def test_high_risk_multiple_abnormals(self):
        scorer = CompositeRiskScorer()
        score, label, triggers = scorer.compute(
            "Urgent/Critical", "Tachycardic",
            ["WBC", "Hemoglobin", "Creatinine"], 90.0
        )
        assert label == "HIGH RISK"
        assert score >= 70

    def test_score_capped_at_100(self):
        scorer = CompositeRiskScorer()
        score, _, _ = scorer.compute(
            "Urgent/Critical", "Tachycardic",
            ["WBC", "Hemoglobin", "Creatinine", "Glucose"], 85.0, [10, 9, 8]
        )
        assert score <= 100


# ── Early Warning System Tests ────────────────────────────────────────────────

class TestEarlyWarningSystem:

    def test_sepsis_alert_triggered(self):
        ews = EarlyWarningSystem()
        alerts = ews.evaluate("Routine/Stable", "Tachycardic", ["WBC"], 96.0)
        assert any("Sepsis" in a["alert"] for a in alerts)

    def test_no_alerts_on_normal(self):
        ews = EarlyWarningSystem()
        alerts = ews.evaluate("Routine/Stable", "Normal", [], 98.0)
        assert len(alerts) == 0


# ── Clinical Decision Assistant Tests ─────────────────────────────────────────

class TestClinicalDecisionAssistant:

    def test_returns_required_keys(self):
        cda = ClinicalDecisionAssistant()
        result = cda.generate_suggestions(
            "Routine/Stable", "Normal", [], 98.0, 75.0
        )
        assert "next_steps" in result
        assert "possible_conditions" in result
        assert "monitoring" in result

    def test_tachycardia_suggests_ecg(self):
        cda = ClinicalDecisionAssistant()
        result = cda.generate_suggestions(
            "Routine/Stable", "Tachycardic", [], 97.0, 110.0
        )
        ecg_mentioned = any("ECG" in s or "ecg" in s.lower() for s in result["next_steps"])
        assert ecg_mentioned


# ── Hybrid Reasoning Engine Tests ─────────────────────────────────────────────

class TestHybridReasoningEngine:

    def test_generates_summary_string(self):
        engine = HybridReasoningEngine()
        summary = engine.generate_clinical_summary(
            "Patient presents with fever.", "Tachycardic", 110.0, 93.0,
            ["WBC"], "Urgent/Critical"
        )
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_confidence_increases_with_data(self):
        engine = HybridReasoningEngine()
        low = engine.compute_confidence("Neutral", "Normal", [])
        high = engine.compute_confidence("Urgent/Critical", "Tachycardic",
                                         ["WBC", "Hemoglobin"], ["WBC"])
        assert high > low


# ── Context Correlator Tests ──────────────────────────────────────────────────

class TestContextCorrelator:

    def test_detects_sepsis_correlation(self):
        cc = ContextCorrelator()
        corrs = cc.detect_correlations(
            "Patient has fever and infection signs.", "Tachycardic", ["WBC"]
        )
        assert len(corrs) >= 1

    def test_no_correlation_on_empty_text(self):
        cc = ContextCorrelator()
        corrs = cc.detect_correlations("", "Normal", [])
        assert corrs == []

# ── ML Text Classifier Tests ──────────────────────────────────────────────────

class TestMLTextClassifier:

    def test_predict_without_model(self):
        from ai_engine import MLTextClassifier
        # Test fallback behavior when model is not trained yet
        clf = MLTextClassifier(model_dir="non_existent_dummy_dir")
        assert not clf.loaded
        label, conf = clf.predict("Patient has acute chest pain.")
        assert label is None
        assert conf == 0.0

class TestHybridSentiment:

    def test_hybrid_sentiment_fallback(self):
        from ai_engine import get_hybrid_sentiment, ml_classifier
        ml_classifier.loaded = False  # Ensure fallback
        rule_sent, ml_bonus, ml_conf, ml_pred = get_hybrid_sentiment("Patient has acute chest pain.")
        assert rule_sent == "Urgent/Critical"
        assert ml_bonus == 0.0
        assert ml_conf == 0.0
        assert ml_pred is None

    def test_hybrid_sentiment_agreement_mock(self, monkeypatch):
        from ai_engine import ml_classifier, get_hybrid_sentiment
        # Mock predict to simulate agreement
        def mock_predict(text):
            return "Urgent/Critical", 80.0
            
        monkeypatch.setattr(ml_classifier, "predict", mock_predict)
        rule_sent, ml_bonus, ml_conf, ml_pred = get_hybrid_sentiment("Patient has acute chest pain.")
        
        assert rule_sent == "Urgent/Critical"
        assert ml_pred == "Urgent/Critical"
        assert ml_conf == 80.0
        assert ml_bonus > 0.0  # Bonus should be applied for agreement

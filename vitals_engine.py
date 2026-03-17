"""
vitals_engine.py — Vital Signs Engine & Temporal Intelligence.

Migrated from app.py vitals logic + new temporal trending system.
Provides:
  - Scenario-based vital sign generation (MIMIC-III calibrated)
  - Vitals analysis with status classification
  - Temporal trend tracking across sessions
  - Worsening/improvement detection
"""

import numpy as np
import pandas as pd


# ── Core vitals functions (migrated from app.py) ─────────────────────────────

def generate_scenario_vitals(scenario: str) -> pd.DataFrame:
    """
    Generate 12-point vitals time series.
    Ranges calibrated from MIMIC-III ICU mean vitals per condition.
    """
    rng = np.random.default_rng()
    if scenario == "Sepsis Risk":
        hr   = rng.integers(105, 131, size=12)
        spo2 = rng.integers(89,  95,  size=12)
    elif scenario == "Cardiac Distress":
        hr   = rng.integers(45,  58,  size=12)
        spo2 = rng.integers(85,  93,  size=12)
    else:
        hr   = rng.integers(65,  86,  size=12)
        spo2 = rng.integers(96, 101,  size=12)
    return pd.DataFrame({"Heart Rate (bpm)": hr, "SpO₂ (%)": spo2})


def analyze_vitals(df: pd.DataFrame) -> tuple:
    """Compute average HR, SpO2, and classify cardiac status."""
    avg_hr   = df["Heart Rate (bpm)"].mean()
    avg_spo2 = df["SpO₂ (%)"].mean()
    if avg_hr > 100:
        status = "Tachycardic"
    elif avg_hr < 60:
        status = "Bradycardic"
    else:
        status = "Normal"
    return avg_hr, avg_spo2, status


# ── Temporal Intelligence ─────────────────────────────────────────────────────

class TemporalTracker:
    """
    Tracks vital signs over time using a simple in-memory history list.
    Designed to be stored in st.session_state for persistence across
    Streamlit reruns.

    Usage:
        if "temporal_tracker" not in st.session_state:
            st.session_state.temporal_tracker = TemporalTracker()

        tracker = st.session_state.temporal_tracker
        tracker.record(avg_hr, avg_spo2, vital_status)
    """

    def __init__(self):
        self.history = []  # List of {"hr": float, "spo2": float, "status": str}
        self.max_history = 20  # Keep last 20 readings

    def record(self, avg_hr: float, avg_spo2: float, status: str):
        """Record a new vital sign reading."""
        self.history.append({
            "hr":     avg_hr,
            "spo2":   avg_spo2,
            "status": status,
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history_df(self) -> pd.DataFrame:
        """Return history as a DataFrame for charting."""
        if not self.history:
            return pd.DataFrame(columns=["Reading #", "Heart Rate (bpm)", "SpO₂ (%)"])
        df = pd.DataFrame(self.history)
        df = df.rename(columns={"hr": "Heart Rate (bpm)", "spo2": "SpO₂ (%)"})
        df["Reading #"] = range(1, len(df) + 1)
        return df[["Reading #", "Heart Rate (bpm)", "SpO₂ (%)"]]

    def detect_worsening(self) -> dict:
        """
        Compare the most recent readings against earlier ones to detect trends.
        Returns: {"hr_trend": str, "spo2_trend": str, "overall": str}

        Trends: "worsening", "improving", "stable"
        """
        if len(self.history) < 2:
            return {
                "hr_trend":   "insufficient_data",
                "spo2_trend": "insufficient_data",
                "overall":    "insufficient_data",
            }

        # Compare last 3 (or fewer) readings vs. the 3 before them
        recent_count = min(3, len(self.history))
        recent = self.history[-recent_count:]
        older  = self.history[:-recent_count] if len(self.history) > recent_count else recent

        if not older:
            older = recent

        avg_recent_hr   = sum(r["hr"] for r in recent) / len(recent)
        avg_older_hr    = sum(r["hr"] for r in older) / len(older)
        avg_recent_spo2 = sum(r["spo2"] for r in recent) / len(recent)
        avg_older_spo2  = sum(r["spo2"] for r in older) / len(older)

        # HR trend: rising HR = worsening (tachycardia concern)
        hr_delta = avg_recent_hr - avg_older_hr
        if hr_delta > 5:
            hr_trend = "worsening"
        elif hr_delta < -5:
            hr_trend = "improving"
        else:
            hr_trend = "stable"

        # SpO2 trend: dropping SpO2 = worsening
        spo2_delta = avg_recent_spo2 - avg_older_spo2
        if spo2_delta < -2:
            spo2_trend = "worsening"
        elif spo2_delta > 2:
            spo2_trend = "improving"
        else:
            spo2_trend = "stable"

        # Overall assessment
        if hr_trend == "worsening" or spo2_trend == "worsening":
            overall = "deteriorating"
        elif hr_trend == "improving" and spo2_trend != "worsening":
            overall = "improving"
        elif spo2_trend == "improving" and hr_trend != "worsening":
            overall = "improving"
        else:
            overall = "stable"

        return {
            "hr_trend":   hr_trend,
            "spo2_trend": spo2_trend,
            "overall":    overall,
        }

    def get_trend_summary(self) -> str:
        """Return a human-readable trend summary."""
        trends = self.detect_worsening()

        if trends["overall"] == "insufficient_data":
            return ("📊 *Temporal tracking active.* Generate multiple reports "
                    "to see trend analysis.")

        trend_icon = {
            "deteriorating": "📉",
            "improving":     "📈",
            "stable":        "➡️",
        }

        icon = trend_icon.get(trends["overall"], "➡️")
        overall = trends["overall"].upper()

        parts = [f"{icon} **Patient Condition: {overall}**"]

        if trends["hr_trend"] != "stable":
            parts.append(f"  • Heart Rate: {trends['hr_trend']}")
        if trends["spo2_trend"] != "stable":
            parts.append(f"  • SpO₂: {trends['spo2_trend']}")
        if trends["hr_trend"] == "stable" and trends["spo2_trend"] == "stable":
            parts.append("  • All vital parameters stable across readings")

        parts.append(f"  • Data points: {len(self.history)} readings tracked")

        return "\n".join(parts)

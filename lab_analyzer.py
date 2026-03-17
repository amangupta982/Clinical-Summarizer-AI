"""
lab_analyzer.py — Smart Laboratory Analysis Engine.

Migrated from app.py lab logic + new smart interpretation layer.
Provides:
  - WHO/ARUP reference ranges with auto-flagging
  - Medical explanation engine per abnormal test
  - Severity scoring (0–10) based on deviation from normal
  - Detailed lab interpretation for clinical decision support
"""

import pandas as pd

# ── Lab reference ranges (based on WHO / ARUP standard adult ranges) ──────────
LAB_REFERENCE_RANGES = {
    "Hemoglobin":   {"unit": "g/dL",   "low": 13.5, "high": 17.5,
                     "critical_low": 7.0,  "critical_high": 20.0},
    "WBC":          {"unit": "10³/µL", "low": 4.5,  "high": 11.0,
                     "critical_low": 2.0,  "critical_high": 30.0},
    "Creatinine":   {"unit": "mg/dL",  "low": 0.7,  "high": 1.3,
                     "critical_low": 0.0,  "critical_high": 10.0},
    "Glucose":      {"unit": "mg/dL",  "low": 70.0, "high": 100.0,
                     "critical_low": 40.0, "critical_high": 500.0},
    "Platelets":    {"unit": "10³/µL", "low": 150,  "high": 400,
                     "critical_low": 50,   "critical_high": 1000},
    "Sodium":       {"unit": "mEq/L",  "low": 136,  "high": 145,
                     "critical_low": 120,  "critical_high": 160},
    "Potassium":    {"unit": "mEq/L",  "low": 3.5,  "high": 5.1,
                     "critical_low": 2.5,  "critical_high": 6.5},
    "ALT":          {"unit": "U/L",    "low": 7,    "high": 56,
                     "critical_low": 0,    "critical_high": 1000},
}

DEFAULT_LABS = pd.DataFrame({
    "Test":   ["Hemoglobin", "WBC",   "Creatinine", "Glucose", "Platelets", "Sodium"],
    "Result": [10.5,          14.2,    0.9,           115.0,     220,         138],
    "Unit":   ["g/dL",       "10³/µL","mg/dL",       "mg/dL",  "10³/µL",   "mEq/L"],
    "Status": ["Low",         "High",  "Normal",      "High",   "Normal",   "Normal"],
})

# ── Medical explanation knowledge base ────────────────────────────────────────
# Sourced from standard clinical pathology references.

LAB_EXPLANATIONS = {
    "Hemoglobin": {
        "High": "Elevated hemoglobin may indicate polycythemia, dehydration, "
                "chronic hypoxia, or high-altitude adaptation.",
        "Low":  "Low hemoglobin suggests anemia — may be due to iron deficiency, "
                "chronic disease, blood loss, or bone marrow disorders.",
    },
    "WBC": {
        "High": "Elevated WBC (leukocytosis) may indicate bacterial infection, "
                "inflammation, stress response, or leukemia.",
        "Low":  "Low WBC (leukopenia) may suggest viral infection, bone marrow "
                "suppression, autoimmune conditions, or chemotherapy effects.",
    },
    "Creatinine": {
        "High": "Elevated creatinine suggests impaired kidney function — may indicate "
                "acute kidney injury, chronic kidney disease, or dehydration.",
        "Low":  "Low creatinine may indicate reduced muscle mass, advanced liver "
                "disease, or excessive fluid intake.",
    },
    "Glucose": {
        "High": "Hyperglycemia may indicate diabetes mellitus, stress response, "
                "corticosteroid use, or pancreatic disorders.",
        "Low":  "Hypoglycemia may indicate insulin excess, liver failure, "
                "adrenal insufficiency, or prolonged fasting.",
    },
    "Platelets": {
        "High": "Thrombocytosis may indicate infection, inflammation, iron deficiency, "
                "or myeloproliferative disorders.",
        "Low":  "Thrombocytopenia may indicate bone marrow failure, DIC, ITP, "
                "viral infections, or medication effects.",
    },
    "Sodium": {
        "High": "Hypernatremia may indicate dehydration, diabetes insipidus, "
                "or excessive sodium intake.",
        "Low":  "Hyponatremia may indicate fluid overload, SIADH, diuretic use, "
                "heart failure, or adrenal insufficiency.",
    },
    "Potassium": {
        "High": "Hyperkalemia may indicate renal failure, acidosis, tissue breakdown, "
                "or medication effects (ACE inhibitors, spironolactone).",
        "Low":  "Hypokalemia may indicate diuretic use, vomiting/diarrhea, "
                "alkalosis, or inadequate dietary intake.",
    },
    "ALT": {
        "High": "Elevated ALT suggests hepatocellular injury — may indicate "
                "hepatitis, drug toxicity, fatty liver, or ischemic injury.",
        "Low":  "Low ALT is generally not clinically significant.",
    },
}


# ── Core lab functions (migrated from app.py) ─────────────────────────────────

def auto_flag_labs(lab_df: pd.DataFrame) -> pd.DataFrame:
    """Auto-compute Status + Critical flag from WHO reference ranges."""
    lab_df = lab_df.copy()
    if "Status" not in lab_df.columns:
        lab_df["Status"] = "Normal"
    if "Critical" not in lab_df.columns:
        lab_df["Critical"] = False

    for idx, row in lab_df.iterrows():
        test = row.get("Test", "")
        if test in LAB_REFERENCE_RANGES:
            ref    = LAB_REFERENCE_RANGES[test]
            result = float(row.get("Result", 0))
            if result < ref["low"]:
                lab_df.at[idx, "Status"] = "Low"
            elif result > ref["high"]:
                lab_df.at[idx, "Status"] = "High"
            else:
                lab_df.at[idx, "Status"] = "Normal"
            lab_df.at[idx, "Critical"] = (
                result <= ref["critical_low"] or result >= ref["critical_high"]
            )
    return lab_df


# ── Smart Lab Interpretation ──────────────────────────────────────────────────

def compute_lab_severity(test_name: str, result: float) -> int:
    """
    Compute severity score (0–10) based on how far the result deviates
    from the normal range, scaled relative to the critical thresholds.

    0     = within normal range
    1–3   = mild deviation
    4–6   = moderate deviation
    7–9   = severe deviation
    10    = critical / extreme
    """
    if test_name not in LAB_REFERENCE_RANGES:
        return 0

    ref = LAB_REFERENCE_RANGES[test_name]
    low, high = ref["low"], ref["high"]
    crit_low, crit_high = ref["critical_low"], ref["critical_high"]

    # Within normal range
    if low <= result <= high:
        return 0

    if result < low:
        # How far below normal, scaled to critical
        range_span = low - crit_low if low != crit_low else 1
        deviation = (low - result) / range_span
    else:
        # How far above normal, scaled to critical
        range_span = crit_high - high if crit_high != high else 1
        deviation = (result - high) / range_span

    # Clamp and scale to 1–10
    deviation = max(0, min(deviation, 1.0))
    severity = int(1 + deviation * 9)
    return min(severity, 10)


def get_lab_interpretation(test_name: str, status: str) -> str:
    """
    Return a medical explanation for an abnormal lab value.
    Uses the LAB_EXPLANATIONS knowledge base.
    """
    if status == "Normal" or test_name not in LAB_EXPLANATIONS:
        return ""
    explanations = LAB_EXPLANATIONS.get(test_name, {})
    return explanations.get(status, "")


def get_detailed_lab_analysis(lab_df: pd.DataFrame) -> list:
    """
    For each lab test, return a detailed analysis dict:
    {
        "test": str, "result": float, "unit": str, "status": str,
        "critical": bool, "severity": int, "interpretation": str,
        "reference_range": str
    }
    """
    flagged = auto_flag_labs(lab_df)
    analyses = []

    for _, row in flagged.iterrows():
        test_name = row["Test"]
        result = float(row["Result"])
        status = row["Status"]
        critical = bool(row.get("Critical", False))

        ref_range = ""
        if test_name in LAB_REFERENCE_RANGES:
            ref = LAB_REFERENCE_RANGES[test_name]
            ref_range = f"{ref['low']}–{ref['high']} {ref['unit']}"

        analyses.append({
            "test":            test_name,
            "result":          result,
            "unit":            row.get("Unit", ""),
            "status":          status,
            "critical":        critical,
            "severity":        compute_lab_severity(test_name, result),
            "interpretation":  get_lab_interpretation(test_name, status),
            "reference_range": ref_range,
        })

    return analyses

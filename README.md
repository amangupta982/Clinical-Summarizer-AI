# 🧠 AI Clinical Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48-red?logo=streamlit)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-3.9-green)](https://nltk.org)
[![Tests](https://img.shields.io/badge/Tests-58%20passed-brightgreen)](#-running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-ff4b4b?logo=streamlit)](https://clinical-summarizer-ai.streamlit.app)

> **Next-generation clinical decision support — synthesizing unstructured notes, laboratory data, and time-series vitals into actionable, context-aware intelligence.**

A hospital-grade, multimodal AI clinical intelligence system featuring hybrid reasoning, temporal trend tracking, composite risk scoring, early warning alerts, smart lab interpretation, clinical decision assistance, and PDF report generation — built on NLP text analysis, anomaly detection, and time-series vital sign monitoring.

---

## 📸 Demo

<!-- Add a GIF here: record your screen using QuickTime (Mac) or OBS, then upload to the repo -->
<!-- ![Demo GIF](assets/demo.gif) -->

> 🔴 **[Try the Live App →](https://lnkd.in/gpwkQrQD)**  

---

## 🧠 How It Works

The system fuses three independent data modalities through a modular engine architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│              AI Clinical Intelligence Dashboard                  │
│                                                                  │
│  📝 Physician Notes  ──► ai_engine.py                            │
│     │                    ├─ NLP Extraction + Sentiment            │
│     │                    ├─ Hybrid Reasoning Engine               │
│     │                    └─ Context-Aware Correlator              │
│     │                                                            │
│  🧪 Lab Results CSV  ──► lab_analyzer.py                         │
│     │                    ├─ WHO Reference Range Flagging          │
│     │                    ├─ Severity Scoring (0–10)               │
│     │                    └─ Medical Explanation Engine            │
│     │                                                            │
│  💓 Vitals Stream    ──► vitals_engine.py                        │
│     │                    ├─ Time-Series Analysis                  │
│     │                    └─ Temporal Trend Tracker                │
│     │                                                            │
│     └──────────────────► insight_generator.py                     │
│                          ├─ Composite Risk Scorer (0–100)        │
│                          ├─ Early Warning System (6 patterns)    │
│                          ├─ Clinical Decision Assistant          │
│                          └─ Structured Report Generator (PDF)    │
│                                    │                             │
│                                    ▼                             │
│          🚨 AI Intelligence Report + Risk Monitor + Alerts       │
└──────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| AI Engine | `ai_engine.py` | Clinical text NLP, sentiment analysis, hybrid reasoning, cross-modal correlation detection |
| Lab Analyzer | `lab_analyzer.py` | WHO reference range flagging, severity scoring (0–10), medical explanations per test |
| Vitals Engine | `vitals_engine.py` | Scenario-based vital generation, temporal trend tracking, worsening/improvement detection |
| Insight Generator | `insight_generator.py` | Composite risk scoring, early warning system, clinical decision assistant, report generation |
| Dashboard | `app.py` | Streamlit UI with premium glassmorphism design, risk gauge, trend charts, PDF export |

### 🔥 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 Hybrid Reasoning | Doctor-friendly summaries with confidence scoring |
| 📈 Temporal Intelligence | Track vitals across sessions, detect deterioration trends |
| 🧬 Smart Lab Interpretation | Severity bars, medical explanations per abnormal test |
| ⚠️ Early Warning System | 6 pattern-based alerts (sepsis, cardiac, respiratory, renal, metabolic, anemia) |
| 📊 Risk Score Gauge | Composite 0–100 risk score combining all modalities |
| 🧑‍⚕️ Clinical Decision Assistant | Suggested next steps, possible conditions, monitoring plans |
| 🔗 Context-Aware Correlations | Cross-modal pattern detection (e.g., fever text + high HR + elevated WBC) |
| 📄 PDF Report Export | Full structured clinical intelligence report with PDF download |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/amangupta982/Clinical-Summarizer-AI.git
cd Clinical-Summarizer-AI
pip install -r requirements.txt
```

### 2. Run Locally

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

### 3. Try a Sample

Paste this into the Physician Notes box and click **Generate Comprehensive Clinical Intelligence Report**:

```
Patient presents with acute chest pain radiating to the left arm. History of hypertension 
and type 2 diabetes. Patient denies shortness of breath. Diagnosis: rule out ACS. 
Treatment initiated with aspirin and nitrates. Vitals currently stable.
```

Then select **Cardiac Distress** in the sidebar to simulate deteriorating vitals.

> 💡 **Tip:** Click "Generate" multiple times to see temporal trend tracking in action — the system tracks vitals across sessions and detects worsening/improvement patterns.

---

## 📊 Sample Output

**Patient Scenario: Sepsis Risk**

| Metric | Value | Status |
|--------|-------|--------|
| Avg Heart Rate | 118 bpm | 🔴 Tachycardic |
| Avg SpO2 | 91.2% | 🔴 Low |
| Abnormal Labs | 2 (WBC, Hemoglobin) | ⚠️ Attention Required |
| Clinical Tone | Urgent/Critical | 🚨 Warning |
| Risk Score | 87/100 | 🔴 HIGH RISK |
| Confidence | 85% | ✅ High |

**Generated Actions:**
- 🚨 HIGH ALERT: Possible Sepsis detected (Elevated WBC + Tachycardia). Initiate immediate protocol.
- ⚠️ Patient narrative indicates acute distress aligning with abnormal vitals. Immediate physician review required.

**Early Warning Alerts:**
- 🔴 Sepsis Risk Increasing — Elevated WBC + tachycardia suggests systemic infection
- 🔴 Cardiac Instability Detected — Abnormal HR with low SpO₂

**AI Clinical Assistant:**
- Consider 12-lead ECG to rule out arrhythmia
- Order blood cultures and complete differential
- May indicate: Sepsis / SIRS criteria

---

## 🗂️ Project Structure

```
Clinical-Summarizer-AI/
├── app.py                  # Main Streamlit dashboard (UI + orchestration)
├── ai_engine.py            # Clinical text NLP, hybrid reasoning, context correlator
├── lab_analyzer.py         # Smart lab analysis, severity scoring, medical explanations
├── vitals_engine.py        # Vitals generation, temporal trend tracking
├── insight_generator.py    # Risk scoring, early warnings, decision support, reports
├── requirements.txt        # Python dependencies
├── workflow.jpeg           # System architecture diagram
├── .devcontainer/          # VS Code dev container config
├── assets/                 # Screenshots and demo GIFs
│   └── demo.gif            # (add this after recording demo)
├── sample_data/
│   └── sample_labs.csv     # Example lab CSV for testing
├── tests/
│   ├── test_app.py         # Unit tests for core logic (32 tests)
│   └── test_modules.py     # Unit tests for new modules (26 tests)
└── README.md
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

> **58 tests** across 2 test files — 32 original + 26 new module tests.

---

## ☁️ Deployment

### Streamlit Cloud (Live)

This app is deployed on **Streamlit Cloud**:

> 🚀 **[Live App → clinical-summarizer-ai.streamlit.app](https://clinical-summarizer-ai.streamlit.app)**

To deploy your own fork:
1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your forked repo, branch `main`, file `app.py`
4. Click **Deploy** — your app is live!

### HuggingFace Spaces (Alternative)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Select **Streamlit** as the SDK
3. Push this repo to the Space:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/clinical-summarizer
git push space main
```

4. Your app is live at `https://huggingface.co/spaces/YOUR_USERNAME/clinical-summarizer`

---

## 🔭 Roadmap

- [x] Modular engine architecture (ai_engine, lab_analyzer, vitals_engine, insight_generator)
- [x] Hybrid reasoning engine with confidence scoring
- [x] Temporal vitals tracking with trend detection
- [x] Smart lab interpretation with severity scoring & medical explanations
- [x] Composite risk scoring (0–100) with visual gauge
- [x] Early warning system (sepsis, cardiac, respiratory, renal, metabolic, anemia)
- [x] Clinical decision assistant with safe-language suggestions
- [x] Cross-modal correlation detection
- [x] Export reports as structured PDF
- [ ] Replace keyword extraction with fine-tuned BioBERT summarization model
- [ ] Add support for FHIR-format patient records (HL7 standard)
- [ ] Integrate with real-time vitals streaming via WebSockets
- [ ] Add SHAP-based explainability for lab anomaly flags
- [ ] Multi-patient dashboard view
- [ ] LLM integration (OpenAI / HuggingFace) for enhanced reasoning

> 💡 See [open issues](https://github.com/amangupta982/Clinical-Summarizer-AI/issues) — contributions welcome!  
> Issues labeled [`good first issue`](https://github.com/amangupta982/Clinical-Summarizer-AI/issues?q=label%3A%22good+first+issue%22) are beginner-friendly.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feat/add-bioBERT-summarizer`
3. Commit your changes: `git commit -m 'feat: add BioBERT-based clinical summarization'`
4. Push to the branch: `git push origin feat/add-bioBERT-summarizer`
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## ⚠️ Disclaimer

This tool is a **research prototype and educational demonstration only**. It is **not intended for clinical use** and should not be used to make real medical decisions. Always consult qualified healthcare professionals.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aman Gupta** · [@amangupta982](https://github.com/amangupta982)  
[LinkedIn](https://linkedin.com/in/aman-gupta-b617772a6) · [Email](mailto:amanmacair98@gmail.com)

---

*If this project helped you, please consider giving it a ⭐ — it helps others find it!*

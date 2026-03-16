# 🏥 Clinical Multi-Modal AI Summarizer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces)

> **Synthesizing unstructured physician notes, laboratory data, and time-series vitals into dynamic, context-aware clinical insights — in real time.**

A multimodal AI clinical decision-support tool that combines NLP-based text analysis, anomaly detection on lab results, and time-series vital sign monitoring to generate actionable patient summaries for healthcare providers.

---

## 📸 Demo

<!-- Add a GIF here: record your screen using QuickTime (Mac) or OBS, then upload to the repo -->
<!-- ![Demo GIF](assets/demo.gif) -->

> 🔴 **[Try the Live App →](https://your-huggingface-spaces-link-here)**  
> *(Deploy to HuggingFace Spaces — free, takes 5 minutes — instructions below)*

---

## 🧠 How It Works

The system fuses three independent data modalities:

```
┌─────────────────────────────────────────────────────────────┐
│                  Clinical-Summarizer-AI                      │
│                                                             │
│  📝 Physician Notes  ──► NLP Extraction + Sentiment        │
│  🧪 Lab Results CSV  ──► Anomaly Detection + Flagging      │
│  💓 Vitals Stream    ──► Time-Series Analysis               │
│                              │                              │
│                              ▼                              │
│              Cross-Modal Fusion Engine                       │
│                              │                              │
│                              ▼                              │
│         🚨 Actionable Clinical Report + Triage Level        │
└─────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Technique | Purpose |
|--------|-----------|---------|
| Text Summarization | NLTK sentence tokenization + clinical keyword extraction | Extract key clinical sentences from physician notes |
| Sentiment/Urgency | VADER lexicon + domain-specific urgency keywords | Classify patient state as Routine or Urgent/Critical |
| Vitals Analysis | Statistical thresholding (HR, SpO2) | Detect tachycardia, bradycardia, hypoxia |
| Lab Flagging | Rule-based reference range checking | Identify abnormal CBC, metabolic panel values |
| Cross-Modal Fusion | Conditional logic across all 3 modalities | Generate sepsis alerts, cardiac flags, escalation recommendations |

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

Paste this into the Physician Notes box and click **Generate Comprehensive Clinical Report**:

```
Patient presents with acute chest pain radiating to the left arm. History of hypertension 
and type 2 diabetes. Patient denies shortness of breath. Diagnosis: rule out ACS. 
Treatment initiated with aspirin and nitrates. Vitals currently stable.
```

Then select **Cardiac Distress** in the sidebar to simulate deteriorating vitals.

---

## 📊 Sample Output

**Patient Scenario: Sepsis Risk**

| Metric | Value | Status |
|--------|-------|--------|
| Avg Heart Rate | 118 bpm | 🔴 Tachycardic |
| Avg SpO2 | 91.2% | 🔴 Low |
| Abnormal Labs | 2 (WBC, Hemoglobin) | ⚠️ Attention Required |
| Clinical Tone | Urgent/Critical | 🚨 Warning |

**Generated Actions:**
- 🚨 HIGH ALERT: Possible Sepsis detected (Elevated WBC + Tachycardia). Initiate immediate protocol.
- ⚠️ Patient narrative indicates acute distress aligning with abnormal vitals. Immediate physician review required.

---

## 🗂️ Project Structure

```
Clinical-Summarizer-AI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── workflow.jpeg           # System architecture diagram
├── .devcontainer/          # VS Code dev container config
├── assets/                 # Screenshots and demo GIFs
│   └── demo.gif            # (add this after recording demo)
├── sample_data/
│   └── sample_labs.csv     # Example lab CSV for testing
├── tests/
│   └── test_app.py         # Unit tests for core logic
└── README.md
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/test_app.py -v
```

---

## ☁️ Deploy to HuggingFace Spaces (Free)

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

- [ ] Replace keyword extraction with fine-tuned BioBERT summarization model
- [ ] Add support for FHIR-format patient records (HL7 standard)
- [ ] Integrate with real-time vitals streaming via WebSockets
- [ ] Add SHAP-based explainability for lab anomaly flags
- [ ] Multi-patient dashboard view
- [ ] Export reports as structured PDF

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

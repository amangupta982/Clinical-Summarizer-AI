# Contributing to Clinical-Summarizer-AI

Thank you for your interest in contributing! This document explains how to get started.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Clinical-Summarizer-AI.git
   cd Clinical-Summarizer-AI
   ```
3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest
   ```

## Making Changes

- Create a new branch for every change:
  ```bash
  git checkout -b feat/your-feature-name
  ```
- Write or update tests for your change in `tests/test_app.py`
- Run tests before submitting:
  ```bash
  pytest tests/ -v
  ```

## Commit Message Format

Use conventional commits — this keeps the history readable:

```
feat: add BioBERT-based summarization module
fix: resolve NLTK download failure on cold start
docs: update README with HuggingFace deployment steps
test: add unit tests for vitals edge cases
refactor: extract lab flagging logic to separate module
```

## Good First Issues

New to the project? Look for issues labeled [`good first issue`](https://github.com/amangupta982/Clinical-Summarizer-AI/issues?q=label%3A%22good+first+issue%22). These are intentionally scoped to be approachable without deep domain knowledge.

## Ideas for Contribution

- Replace NLTK keyword extraction with a fine-tuned BioBERT/ClinicalBERT model
- Add FHIR-format patient record ingestion
- Add SHAP explainability for lab anomaly detection
- Improve test coverage (currently ~60%)
- Add a sample lab CSV file to `sample_data/`
- Improve the workflow diagram in `workflow.jpeg`

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Write a clear PR description explaining *what* and *why*
- Reference any related issue numbers: `Closes #12`
- Ensure all tests pass before requesting review

## Code of Conduct

Be respectful and constructive. This is a learning-oriented open-source project.

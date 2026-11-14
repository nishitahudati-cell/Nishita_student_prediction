# Student Performance Prediction System

Education Analytics | Machine Learning

A modular academic analytics project for estimating student pass/fail outcomes from historical academic indicators and background characteristics using leakage-safe machine learning pipelines.

---

## Dataset Benchmark

Primary benchmark: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).

- **Mathematics dataset (`student-mat.csv`)**: 395 records.
- **Portuguese dataset (`student-por.csv`)**: 649 records.
- **Pass Threshold**: Pass = `G3 >= 10`, Fail = `G3 < 10` (on a 0–20 scale).
- **Leakage Prevention & Timing Scenarios**:
  - `G3` is the target outcome and is **strictly excluded** from all predictor feature sets.
  - **Early-Term Scenario**: Prior period exam grades `G1` and `G2` are excluded. Evaluates baseline academic, behavioral, and demographic indicators (e.g. past failures, study time, absences, family support).
  - **Late-Term Scenario**: Mid-term grades `G1` and `G2` are included once they become available.

---

## Quick Start & Running the Project

### 1. Environment Setup
```bash
# Activate the virtual environment
source .venv/bin/activate

# (Optional) Reinstall dependencies if running on a new machine
pip install -r backend/requirements.txt
```

### 2. Dataset Acquisition & Inspection
```bash
# Download and unpack UCI datasets
python scripts/download_uci_dataset.py

# Inspect datasets and label distributions
python scripts/inspect_dataset.py
```

### 3. Model Training & Evaluation
```bash
# Train Random Forest, Logistic Regression, and Decision Tree with 5-fold CV
python backend/app/ml/trainer.py
```

### 4. Run the Full Application (Web Dashboard + REST API)
```bash
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Open **Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Run Test Suite
```bash
python -m unittest tests/test_pipeline_and_api.py
```

---

## Model Benchmark Results

Evaluated on held-out test splits with Stratified 5-Fold Cross Validation:

| Scenario | Best Algorithm | Test Accuracy | Test F1 | Test ROC-AUC | Key Predictors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Early-Term** *(No G1/G2)* | Random Forest | 65.8% | **0.752** | 0.630 | Past Failures, Absences, Going Out, Age, Study Time |
| **Late-Term** *(With G1/G2)* | Logistic Regression | 84.8% | **0.889** | 0.893 | G2 Grade, G1 Grade, Absences, Failures |

---

## Architecture & Features

- **Inference & Risk Stratification**:
  - Probability output mapped to **Low Risk** ($p \ge 0.65$), **Medium Risk** ($0.40 \le p < 0.65$), and **High Risk** ($p < 0.40$).
  - Top factor explainability identifying specific levers (absences, study habits, failure history).
- **Batch Processing**:
  - Upload cohort CSV files for aggregate risk distribution and student-level flags.
- **Persistence & Auditing**:
  - SQLite database storing prediction logs, timestamps, risk tiers, and model audit records.
- **REST API Endpoints**:
  - `GET /api/health` — System status and model counts
  - `POST /api/predict` — Single student prediction with factor explanations
  - `POST /api/predict/batch` — Batch CSV cohort prediction
  - `POST /api/train` — Retrain models on selected datasets
  - `GET /api/train/reports/{timing_scenario}` — Evaluation metrics and confusion matrices
  - `GET /api/history` — Query SQLite prediction logs

---

## Responsible-Use Note

Predictions are statistical estimates, not guaranteed outcomes or definitive verdicts on student capabilities. They are intended solely for academic advisors and educators to direct early supportive tutoring and resources. Privacy is maintained by utilizing pseudonymous identifiers and never storing personal identifiable information (PII).

---

## Project Status

- [x] **Phase 1**: Project scaffold, dataset acquisition utilities, and development fixtures
- [x] **Phase 2**: Leakage-safe data cleaning & Scikit-learn preprocessing pipelines
- [x] **Phase 3**: Multi-model training, 5-fold cross-validation, probability calibration, and metric reporting
- [x] **Phase 4**: FastAPI backend API with Pydantic validation
- [x] **Phase 5**: Interactive web dashboard with single prediction, batch processing, and benchmark tabs
- [x] **Phase 6**: SQLite persistence for prediction history and model runs
- [x] **Phase 7**: Comprehensive test suite (`tests/test_pipeline_and_api.py`)
- [x] **Phase 8**: Documentation and run guides

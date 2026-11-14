# System Architecture

## High-level flow

React + TypeScript frontend -> FastAPI REST API -> application services -> ML pipeline and SQLite database.

The ML pipeline owns data validation, preprocessing, model training, evaluation, probability calibration, prediction, and explanation. The API layer validates requests and delegates work to services. The database stores prediction metadata and history, not unnecessary personal identifiers.

## Component responsibilities

### Frontend
- `pages/`: Dashboard Overview, Student Prediction, Batch Prediction, Dataset Management, Model Comparison, Prediction History, About.
- `components/`: reusable cards, tables, charts, forms, status indicators.
- `services/`: typed API client.
- `types/`: request and response contracts.

### Backend
- `api/`: HTTP routes and request/response handling.
- `schemas/`: Pydantic validation models.
- `services/`: orchestration for datasets, training, predictions, and reports.
- `database/`: SQLite connection, models, and repositories.
- `ml/`: reusable sklearn pipelines and model artifacts.
- `core/`: environment configuration and privacy-aware logging.

### Data and artifacts
- `data/raw/`: immutable source datasets.
- `data/processed/`: validated model-ready datasets.
- `data/sample/`: synthetic data for development only.
- `models/`: versioned trained model artifacts.
- `reports/`: evaluation and monitoring outputs.

## Prediction data contract (planned)

Numeric examples: `study_hours_per_week`, `previous_score`, `assignment_completion_rate`, `internal_assessment_score`, `previous_failures`, and `attendance_percentage` when the source dataset truly provides the denominator needed to calculate it.

The UCI benchmark uses its own available columns. A dataset adapter will map source columns to canonical project features where valid; unsupported fields remain absent rather than fabricated.

## Leakage and timing

For the UCI benchmark, `G3` is the target. `G1` and `G2` are eligible only for the clearly stated prediction point after those earlier grades are known. All preprocessing must be fitted on training folds only through sklearn Pipelines. No target-derived field may enter predictors.

## Privacy and responsible use

Avoid logging names, student IDs, raw uploaded rows, or sensitive free text. Use pseudonymous record identifiers where needed. Explanations must be phrased as associations and uncertainty, not causal claims. Educators retain responsibility for decisions.

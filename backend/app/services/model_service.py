"""Service for managing model training and retrieving evaluation benchmarks."""
import json
from pathlib import Path
from typing import Dict, Any, List

from backend.app.core.config import MODEL_DIR, REPORT_DIR, RAW_DATA_DIR, SAMPLE_DATA_DIR
from backend.app.ml.trainer import train_and_evaluate
from backend.app.ml.predictor import clear_predictor_cache
from backend.app.database.repository import log_model_run


def trigger_retraining(
    dataset_name: str = "student-mat.csv",
    timing_scenario: str = "early",
    target_metric: str = "f1",
) -> Dict[str, Any]:
    """Execute model training on specified dataset and save to DB."""
    if dataset_name.startswith("synthetic"):
        dataset_path = SAMPLE_DATA_DIR / dataset_name
    else:
        dataset_path = RAW_DATA_DIR / "uci_student_performance" / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Requested dataset not found: {dataset_path}")

    summary = train_and_evaluate(
        dataset_path=dataset_path,
        timing_scenario=timing_scenario,
        target_metric=target_metric,
    )

    best_model_name = summary["best_model"]
    best_metrics = summary["models"][best_model_name]["test_metrics"]

    log_model_run(
        dataset_name=dataset_name,
        timing_scenario=timing_scenario,
        best_model=best_model_name,
        test_metrics=best_metrics,
        full_report=summary,
    )

    # Invalidate cached in-memory models so future inferences use new weights
    clear_predictor_cache()

    return summary


def get_evaluation_report(timing_scenario: str = "early") -> Dict[str, Any]:
    """Load latest evaluation report JSON."""
    report_file = REPORT_DIR / f"evaluation_{timing_scenario}.json"
    if not report_file.exists():
        # Trigger training to produce initial baseline report
        return trigger_retraining(timing_scenario=timing_scenario)
    
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


def list_trained_artifacts() -> List[Dict[str, Any]]:
    """Enumerate serialized models in models directory."""
    artifacts = []
    for file in sorted(MODEL_DIR.glob("*.joblib")):
        artifacts.append({
            "filename": file.name,
            "size_kb": round(file.stat().st_size / 1024, 2),
            "last_modified": file.stat().st_mtime,
        })
    return artifacts


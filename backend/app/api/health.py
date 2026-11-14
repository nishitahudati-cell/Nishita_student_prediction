"""Health and system status endpoints."""
import sys
import resource
from fastapi import APIRouter
from backend.app.core.config import APP_NAME, APP_ENV, RAW_DATA_DIR, MODEL_DIR
from backend.app.database.session import DB_PATH, get_db_connection

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """Verify system readiness, database connectivity, table metrics, and resource usage."""
    db_ok = DB_PATH.exists()
    db_size_kb = round(DB_PATH.stat().st_size / 1024, 1) if db_ok else 0.0
    dataset_mat = (RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv").exists()
    dataset_por = (RAW_DATA_DIR / "uci_student_performance" / "student-por.csv").exists()
    dataset_ok = dataset_mat and dataset_por
    models_available = len(list(MODEL_DIR.glob("*.joblib")))

    # Query table row counts
    prediction_records = 0
    model_run_records = 0
    if db_ok:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM predictions")
                prediction_records = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM model_runs")
                model_run_records = cursor.fetchone()[0]
        except Exception:
            pass

    # Memory RSS usage
    try:
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = round(raw_rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except Exception:
        rss_mb = 0.0

    is_healthy = db_ok and models_available > 0 and dataset_ok

    return {
        "status": "healthy" if is_healthy else "degraded",
        "app_name": APP_NAME,
        "environment": APP_ENV,
        "python_version": sys.version.split()[0],
        "memory_rss_mb": rss_mb,
        "database_connected": db_ok,
        "database_size_kb": db_size_kb,
        "database_records": {
            "predictions_count": prediction_records,
            "model_runs_count": model_run_records,
        },
        "uci_dataset_ready": dataset_ok,
        "uci_datasets": {
            "student_mat": dataset_mat,
            "student_por": dataset_por,
        },
        "models_count": models_available,
    }


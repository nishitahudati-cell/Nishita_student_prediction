"""Endpoints for model training orchestration and benchmark reports."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from backend.app.schemas.training import TrainRequest, ModelEvaluationSummary
from backend.app.services.model_service import (
    trigger_retraining,
    get_evaluation_report,
    list_trained_artifacts,
)

router = APIRouter(tags=["Training & Benchmarks"])


@router.post("/train", response_model=ModelEvaluationSummary)
def train_models(request: TrainRequest):
    """Trigger model training pipeline with leakage prevention and cross-validation."""
    try:
        return trigger_retraining(
            dataset_name=request.dataset,
            timing_scenario=request.timing_scenario,
            target_metric=request.target_metric,
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/train/reports/{timing_scenario}")
def get_report(timing_scenario: Literal["early", "late"]):
    """Retrieve full evaluation report and benchmark comparison for a timing scenario."""
    try:
        return get_evaluation_report(timing_scenario=timing_scenario)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report retrieval error: {str(e)}")


@router.get("/train/models")
def get_model_artifacts():
    """List all persisted model artifacts and their metadata."""
    return {"artifacts": list_trained_artifacts()}


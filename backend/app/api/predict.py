"""Endpoints for single and batch student performance prediction."""
from typing import Optional, Literal
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from backend.app.schemas.student import (
    StudentInput,
    PredictionResponse,
    SimulationRequest,
    SimulationResponse,
)
from backend.app.services.prediction_service import (
    predict_single_student,
    predict_batch_csv,
    simulate_intervention,
)

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_student(
    payload: StudentInput,
    timing_scenario: Literal["early", "late"] = Query(
        default="early",
        description="Select 'early' (pre-grades) or 'late' (with G1/G2 prior exam marks)",
    ),
    model_name: Optional[str] = Query(default=None, description="Model to use (default: active_model)"),
):
    """Predict pass/fail outcome, calculate risk tier, and extract top factor explanations."""
    scenario = "early" if not isinstance(timing_scenario, str) else timing_scenario
    model = None if not isinstance(model_name, str) else model_name
    try:
        return predict_single_student(
            student=payload,
            timing_scenario=scenario,
            model_name=model,
            persist=True,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/simulate", response_model=SimulationResponse)
def simulate_student_intervention(request: SimulationRequest):
    """Simulate intervention changes (e.g. attendance, study hours, tutoring) and calculate projected gain."""
    try:
        return simulate_intervention(
            student=request.student,
            interventions=request.interventions,
            timing_scenario=request.timing_scenario or "early",
            model_name=request.model_name,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.post("/predict/batch")
async def batch_predict(
    file: UploadFile = File(..., description="CSV file with student records"),
    timing_scenario: Literal["early", "late"] = Query(default="early"),
    model_name: Optional[str] = Query(default=None),
):
    """Upload a CSV file and receive batch predictions and aggregate risk distributions."""
    scenario = "early" if not isinstance(timing_scenario, str) else timing_scenario
    model = None if not isinstance(model_name, str) else model_name
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV file.")
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum supported CSV size is 10MB.")

        return predict_batch_csv(
            file_bytes=contents,
            timing_scenario=scenario,
            model_name=model,
            persist=True,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


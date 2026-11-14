"""Endpoints for prediction history and audit records."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query, HTTPException
from backend.app.database.repository import (
    get_recent_predictions,
    get_prediction_by_id,
    get_prediction_stats,
    clear_all_predictions,
    delete_prediction_by_id,
    update_prediction_status,
    add_advisory_note,
    get_advisory_notes,
)

router = APIRouter(tags=["History & Audit"])


class UpdatePredictionStatusRequest(BaseModel):
    status: Literal["Pending", "Reviewed", "Action Scheduled", "Resolved"] = Field(
        default="Reviewed",
        description="Updated advisory intervention workflow state",
    )
    advisor_notes: Optional[str] = Field(default=None, description="Advisory comments or intervention logs")


class CreateAdvisoryNoteRequest(BaseModel):
    author_role: str = Field(default="Academic Advisor", description="Role or identity of advisor")
    note_text: str = Field(..., description="Content of the advisory note or intervention plan")
    intervention_tag: Optional[str] = Field(default="Academic Counseling", description="Category tag")


@router.get("/history")
def get_history(
    limit: int = Query(default=50, ge=1, le=500),
    search: Optional[str] = Query(default=None, description="Search by student identifier or model"),
    risk_tier: Optional[str] = Query(default=None, description="Filter by risk tier (Low Risk, Medium Risk, High Risk)"),
    timing_scenario: Optional[Literal["all", "early", "late"]] = Query(default=None, description="Filter by timing scenario"),
):
    """Fetch past prediction records stored in SQLite database with optional filtering."""
    records = get_recent_predictions(
        limit=limit,
        search=search,
        risk_tier=risk_tier,
        timing_scenario=timing_scenario,
    )
    return {
        "count": len(records),
        "history": records,
    }


@router.get("/history/stats")
def get_audit_stats():
    """Retrieve aggregate statistics on historical prediction activity."""
    return get_prediction_stats()


@router.get("/history/{prediction_id}")
def get_single_prediction_audit(prediction_id: int):
    """Retrieve full audit detail for a specific historical inference."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")
    return record


@router.patch("/history/{prediction_id}")
def update_status(prediction_id: int, req: UpdatePredictionStatusRequest):
    """Update workflow status and advisory notes for an individual prediction audit."""
    existing = get_prediction_by_id(prediction_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")
    success = update_prediction_status(
        prediction_id=prediction_id,
        status=req.status,
        advisor_notes=req.advisor_notes,
    )
    return {
        "status": "success",
        "updated": success,
        "prediction_id": prediction_id,
        "new_workflow_status": req.status,
        "advisor_notes": req.advisor_notes,
    }


@router.delete("/history/{prediction_id}")
def delete_single_prediction(prediction_id: int):
    """Selectively delete a specific prediction audit record without affecting other data."""
    existing = get_prediction_by_id(prediction_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")
    success = delete_prediction_by_id(prediction_id)
    return {
        "status": "success",
        "deleted": success,
        "prediction_id": prediction_id,
        "message": f"Prediction #{prediction_id} successfully deleted.",
    }


@router.post("/history/{prediction_id}/notes")
def create_note_for_prediction(prediction_id: int, req: CreateAdvisoryNoteRequest):
    """Attach a structured advisory or intervention note to a prediction record."""
    existing = get_prediction_by_id(prediction_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")
    note_id = add_advisory_note(
        prediction_id=prediction_id,
        author_role=req.author_role,
        note_text=req.note_text,
        intervention_tag=req.intervention_tag,
    )
    return {
        "status": "success",
        "note_id": note_id,
        "prediction_id": prediction_id,
        "message": "Advisory note attached successfully.",
    }


@router.get("/history/{prediction_id}/notes")
def list_notes_for_prediction(prediction_id: int):
    """Retrieve all advisory notes associated with a prediction record."""
    existing = get_prediction_by_id(prediction_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Prediction record #{prediction_id} not found.")
    notes = get_advisory_notes(prediction_id)
    return {
        "prediction_id": prediction_id,
        "count": len(notes),
        "notes": notes,
    }


@router.delete("/history")
def clear_history():
    """Clear past prediction logs from SQLite database."""
    deleted_count = clear_all_predictions()
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": "Prediction audit history successfully reset.",
    }


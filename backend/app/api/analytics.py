"""API endpoints for cohort analytics, distributions, and benchmark aggregations."""
from typing import Literal, Optional
from fastapi import APIRouter
from backend.app.services.student_service import get_dashboard_analytics

router = APIRouter(tags=["Analytics & Dashboard"])


@router.get("/analytics/dashboard")
def get_dashboard_data(
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
):
    """Retrieve verified aggregated analytics from genuine SQLite database records."""
    return get_dashboard_analytics(subject=subject, department=department)


@router.get("/analytics/summary")
def get_analytics_summary(
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
):
    """Retrieve lightweight executive KPI summary from database."""
    data = get_dashboard_analytics(subject=subject, department=department)
    return {
        "subject_filter": data.get("subject_filter"),
        "department_filter": department,
        "total_students": data.get("total_students"),
        "predicted_pass_rate": data.get("predicted_pass_rate"),
        "average_performance": data.get("average_performance"),
        "median_performance": data.get("median_performance"),
        "performance_std": data.get("performance_std"),
        "at_risk_students": data.get("at_risk_students"),
        "high_risk_count": data.get("high_risk_count"),
        "medium_risk_count": data.get("medium_risk_count"),
        "low_risk_count": data.get("low_risk_count"),
        "average_attendance_rate": data.get("average_attendance_rate"),
        "is_empty": data.get("is_empty", False),
    }

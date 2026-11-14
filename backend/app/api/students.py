"""API endpoints for institutional student data management, records, and imports."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, File, Depends, status
from backend.app.core.auth import (
    get_current_user,
    require_auth,
    require_role,
    enforce_student_self_access,
)
from backend.app.services.student_service import (
    get_student_records,
    get_student_by_id,
    add_student,
    edit_student,
    remove_student,
    get_critical_at_risk_alerts,
    compare_students as service_compare_students,
    export_students_csv,
    preview_csv_import,
    commit_csv_import,
    get_departments,
    get_subjects,
)

router = APIRouter(tags=["Student Management"])


class StudentCreateRequest(BaseModel):
    student_identifier: Optional[str] = Field(None, description="Optional custom ID, e.g. MAT-0400")
    department_code: str = Field(default="math", description="Department code: 'math' or 'portuguese'")
    subject: str = Field(default="Mathematics", description="Subject name")
    school: str = Field(default="Amity University Kolkata", description="University name")
    sex: str = Field(default="F", description="F or M")
    age: int = Field(default=17, ge=10, le=30)
    address: str = Field(default="U", description="U (Urban) or R (Rural)")
    studytime: int = Field(default=2, ge=1, le=4, description="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
    failures: int = Field(default=0, ge=0, le=4)
    absences: int = Field(default=2, ge=0, le=100)
    G1: int = Field(default=10, ge=0, le=20)
    G2: int = Field(default=10, ge=0, le=20)
    G3: int = Field(default=10, ge=0, le=20)
    schoolsup: str = Field(default="no")
    famsup: str = Field(default="no")
    higher: str = Field(default="yes")
    internet: str = Field(default="yes")
    health: int = Field(default=3, ge=1, le=5)
    freetime: int = Field(default=3, ge=1, le=5)
    goout: int = Field(default=3, ge=1, le=5)


class StudentUpdateRequest(BaseModel):
    department_code: Optional[str] = None
    subject: Optional[str] = None
    school: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    studytime: Optional[int] = None
    failures: Optional[int] = None
    absences: Optional[int] = None
    G1: Optional[int] = None
    G2: Optional[int] = None
    G3: Optional[int] = None
    schoolsup: Optional[str] = None
    famsup: Optional[str] = None
    higher: Optional[str] = None
    internet: Optional[str] = None
    health: Optional[int] = None
    freetime: Optional[int] = None
    goout: Optional[int] = None


class ImportCommitRequest(BaseModel):
    records: List[Dict[str, Any]]


@router.get("/students")
def list_students(
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
    search: Optional[str] = None,
    school: Optional[str] = None,
    sex: Optional[str] = None,
    risk_tier: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "student_identifier",
    sort_order: Literal["asc", "desc"] = "asc",
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Retrieve genuine student records from SQLite. Enforces self-only view for STUDENT role."""
    # RBAC constraint: Students only view their own record
    if current_user and isinstance(current_user, dict) and (current_user.get("role") or "").upper() == "STUDENT":
        stu_id = current_user.get("student_identifier") or ""
        single = get_student_by_id(stu_id)
        records = [single] if single else []
        return {
            "total": len(records),
            "page": 1,
            "page_size": page_size,
            "total_pages": 1,
            "records": records,
        }

    return get_student_records(
        subject=subject,
        department=department,
        search=search,
        school=school,
        sex=sex,
        risk_tier=risk_tier,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.get("/students/alerts")
def get_student_alerts(
    limit: int = 5,
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Retrieve prioritized early warning alerts for students needing attention."""
    dept = department or (subject if subject in ("math", "portuguese") else None)
    return get_critical_at_risk_alerts(limit=limit, department=dept)


@router.get("/students/compare")
def compare_students(
    student_a: str = Query(..., description="Primary student identifier (e.g. MAT-0001)"),
    student_b: Optional[str] = Query("cohort_mean", description="Second student identifier or 'cohort_mean'"),
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Compare a student against another student or against cohort benchmarks."""
    enforce_student_self_access(current_user, student_a)
    result = service_compare_students(
        student_id_a=student_a,
        student_id_b=student_b,
        subject=subject,
        department=department,
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to compare students. Verify identifier '{student_a}' and '{student_b}' exist in database.",
        )
    return result


@router.get("/students/export")
def export_students(
    subject: Literal["all", "math", "portuguese"] = "all",
    department: Optional[str] = None,
    search: Optional[str] = None,
    school: Optional[str] = None,
    sex: Optional[str] = None,
    risk_tier: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Export filtered cohort as a downloadable RFC 4180 CSV file from genuine database."""
    if current_user and isinstance(current_user, dict) and (current_user.get("role") or "").upper() == "STUDENT":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Students are not authorized to export cohort data.",
        )
    csv_data = export_students_csv(
        subject=subject,
        department=department,
        search=search,
        school=school,
        sex=sex,
        risk_tier=risk_tier,
    )
    filename = f"student_records_{department or subject}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/students")
def create_student_endpoint(
    req: StudentCreateRequest,
    current_user: Dict[str, Any] = Depends(require_role(["SUPER_ADMIN", "INSTITUTION_ADMIN", "DEPARTMENT_ADMIN", "TEACHER"])),
):
    """Add a new verified student record into SQLite database."""
    stu_id = add_student(req.model_dump())
    created = get_student_by_id(stu_id)
    return {
        "status": "success",
        "message": f"Student record '{stu_id}' created successfully.",
        "student": created,
    }


@router.get("/students/{student_id}")
def get_student_detail(
    student_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Retrieve full profile, academic history, and background for a single student."""
    enforce_student_self_access(current_user, student_id)
    record = get_student_by_id(student_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Student record '{student_id}' not found.")
    return record


@router.put("/students/{student_id}")
def update_student_endpoint(
    student_id: str,
    req: StudentUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_role(["SUPER_ADMIN", "INSTITUTION_ADMIN", "DEPARTMENT_ADMIN", "TEACHER"])),
):
    """Update fields on an existing genuine student record."""
    clean_data = {k: v for k, v in req.model_dump().items() if v is not None}
    success = edit_student(student_id, clean_data)
    if not success:
        raise HTTPException(status_code=404, detail=f"Student record '{student_id}' not found.")
    updated = get_student_by_id(student_id)
    return {
        "status": "success",
        "message": f"Student record '{student_id}' updated successfully.",
        "student": updated,
    }


@router.delete("/students/{student_id}")
def delete_student_endpoint(
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_role(["SUPER_ADMIN", "INSTITUTION_ADMIN"])),
):
    """Delete a student record from SQLite database (Admins only)."""
    success = remove_student(student_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Student record '{student_id}' not found.")
    return {
        "status": "success",
        "message": f"Student record '{student_id}' deleted successfully.",
    }


@router.post("/students/import-preview")
async def import_preview_endpoint(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_role(["SUPER_ADMIN", "INSTITUTION_ADMIN", "DEPARTMENT_ADMIN", "TEACHER"])),
):
    """Validate uploaded CSV file, detect duplicate student IDs and invalid ranges, and preview records."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a comma- or semicolon-delimited CSV.",
        )
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty.",
        )
    preview = preview_csv_import(content)
    return preview


@router.post("/students/import-commit")
def import_commit_endpoint(
    req: ImportCommitRequest,
    current_user: Dict[str, Any] = Depends(require_role(["SUPER_ADMIN", "INSTITUTION_ADMIN", "DEPARTMENT_ADMIN"])),
):
    """Commit pre-validated student records into SQLite database."""
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided to commit.")
    saved_count = commit_csv_import(req.records)
    return {
        "status": "success",
        "message": f"Successfully committed {saved_count} genuine student records to database.",
        "records_saved": saved_count,
    }


@router.get("/departments")
def list_departments(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Retrieve all academic departments from SQLite database."""
    return get_departments()


@router.get("/subjects")
def list_subjects(
    department: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Retrieve academic subjects, optionally filtered by department."""
    return get_subjects(department_code=department)

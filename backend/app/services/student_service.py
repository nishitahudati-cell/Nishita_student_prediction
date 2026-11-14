"""Service for querying real student records and calculating genuine cohort analytics from SQLite."""
from functools import lru_cache
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.app.database.repository import (
    get_students_db,
    get_student_by_id_db,
    create_student_db,
    update_student_db,
    delete_student_db,
    get_dashboard_analytics_db,
    get_student_alerts_db,
    compare_students_db,
    get_departments_db,
    get_subjects_db,
)


def get_student_records(
    subject: str = "all",
    search: Optional[str] = None,
    school: Optional[str] = None,
    sex: Optional[str] = None,
    risk_tier: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "student_identifier",
    sort_order: str = "asc",
    department: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve filtered, sorted, and paginated student records from SQLite database."""
    res = get_students_db(
        department=department,
        subject=subject,
        school=school,
        search=search,
        risk_tier=risk_tier,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if sex and sex != "all":
        res["records"] = [r for r in res["records"] if r.get("sex") == sex]
        res["total"] = len(res["records"])
    return res


def get_student_by_id(student_identifier: str) -> Optional[Dict[str, Any]]:
    """Retrieve full detail for a single student from SQLite database."""
    return get_student_by_id_db(student_identifier)


def add_student(student_data: Dict[str, Any]) -> str:
    """Add a new verified student record into SQLite database."""
    return create_student_db(student_data)


def edit_student(student_identifier: str, update_data: Dict[str, Any]) -> bool:
    """Update student record in SQLite database."""
    return update_student_db(student_identifier, update_data)


def remove_student(student_identifier: str) -> bool:
    """Delete student record from SQLite database."""
    return delete_student_db(student_identifier)


def export_students_csv(
    subject: str = "all",
    search: Optional[str] = None,
    school: Optional[str] = None,
    sex: Optional[str] = None,
    risk_tier: Optional[str] = None,
    department: Optional[str] = None,
) -> str:
    """Filter students from database and return formatted RFC 4180 CSV text."""
    res = get_students_db(
        department=department,
        subject=subject,
        search=search,
        risk_tier=risk_tier,
        page=1,
        page_size=10000,
    )
    records = res.get("records", [])
    if not records:
        return ""
    df = pd.DataFrame(records)
    if school and school != "all" and "school" in df.columns:
        df = df[df["school"] == school]
    if sex and sex != "all" and "sex" in df.columns:
        df = df[df["sex"] == sex]
    return df.to_csv(index=False)


def get_critical_at_risk_alerts(
    limit: int = 5,
    subject: str = "all",
    department: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve prioritized genuine student records needing attention from SQLite database."""
    dept = department or (subject if subject in ("math", "portuguese") else None)
    return get_student_alerts_db(limit=limit, department=dept)


def compare_students(
    student_id_a: str,
    student_id_b: Optional[str] = "cohort_mean",
    subject: str = "all",
    department: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compare student records against each other or against cohort benchmarks."""
    dept = department or (subject if subject in ("math", "portuguese") else None)
    return compare_students_db(student_id_a=student_id_a, student_id_b=student_id_b, department=dept)


def get_dashboard_analytics(subject: str = "all", department: Optional[str] = None) -> Dict[str, Any]:
    """Calculate 100% genuine institutional analytics directly from SQLite database."""
    dept = department or (subject if subject in ("math", "portuguese") else None)
    return get_dashboard_analytics_db(department=dept, subject=subject)


def preview_csv_import(file_bytes: bytes) -> Dict[str, Any]:
    """Validate uploaded student CSV and return preview rows plus detailed errors."""
    import io
    import csv

    if not file_bytes or len(file_bytes) == 0:
        return {
            "total_rows": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "preview_records": [],
            "valid_records": [],
            "errors": ["Uploaded file is empty."],
        }

    sample = file_bytes[:4096].decode("utf-8", errors="replace")
    delimiter = ";" if sample.count(";") > sample.count(",") else ","

    stream = io.StringIO(file_bytes.decode("utf-8", errors="replace"))
    reader = csv.DictReader(stream, delimiter=delimiter)

    valid_records = []
    errors = []
    seen_ids = set()

    for idx, row in enumerate(reader, start=1):
        line_num = idx + 1
        stu_id = (row.get("student_identifier") or row.get("id") or "").strip().upper()

        dept = (row.get("department_code") or row.get("department") or "math").strip().lower()
        if dept not in ("math", "portuguese"):
            dept = "math" if "mat" in stu_id.lower() else "portuguese"

        subject = row.get("subject") or ("Mathematics" if dept == "math" else "Portuguese")

        row_errors = []
        try:
            age = int(row.get("age", 17))
            if age < 10 or age > 30:
                row_errors.append(f"Age {age} out of normal range (10-30)")
        except ValueError:
            row_errors.append("Age must be an integer")
            age = 17

        try:
            absences = int(row.get("absences", 0))
            if absences < 0 or absences > 100:
                row_errors.append(f"Absences {absences} out of range (0-100)")
        except ValueError:
            row_errors.append("Absences must be an integer")
            absences = 0

        try:
            g3 = int(row.get("G3", 10))
            if g3 < 0 or g3 > 20:
                row_errors.append(f"G3 mark {g3} out of range (0-20)")
        except ValueError:
            row_errors.append("G3 mark must be an integer (0-20)")
            g3 = 10

        try:
            failures = int(row.get("failures", 0))
            if failures < 0 or failures > 4:
                row_errors.append(f"Failures {failures} out of range (0-4)")
        except ValueError:
            failures = 0

        if stu_id:
            if stu_id in seen_ids:
                row_errors.append(f"Duplicate student ID '{stu_id}' in file")
            else:
                seen_ids.add(stu_id)
                if get_student_by_id_db(stu_id):
                    row_errors.append(f"Student ID '{stu_id}' already exists in database")

        if row_errors:
            errors.append(f"Row {line_num}: " + "; ".join(row_errors))
        else:
            clean_rec = {
                "student_identifier": stu_id or f"NEW-{len(valid_records)+1:04d}",
                "department_code": dept,
                "subject": subject,
                "school": row.get("school", "Amity University Kolkata"),
                "sex": row.get("sex", "F"),
                "age": age,
                "address": row.get("address", "U"),
                "famsize": row.get("famsize", "GT3"),
                "Pstatus": row.get("Pstatus", "T"),
                "Medu": int(row.get("Medu", 2)),
                "Fedu": int(row.get("Fedu", 2)),
                "Mjob": row.get("Mjob", "other"),
                "Fjob": row.get("Fjob", "other"),
                "reason": row.get("reason", "course"),
                "guardian": row.get("guardian", "mother"),
                "traveltime": int(row.get("traveltime", 1)),
                "studytime": int(row.get("studytime", 2)),
                "failures": failures,
                "schoolsup": row.get("schoolsup", "no"),
                "famsup": row.get("famsup", "no"),
                "paid": row.get("paid", "no"),
                "activities": row.get("activities", "no"),
                "nursery": row.get("nursery", "yes"),
                "higher": row.get("higher", "yes"),
                "internet": row.get("internet", "yes"),
                "romantic": row.get("romantic", "no"),
                "famrel": int(row.get("famrel", 4)),
                "freetime": int(row.get("freetime", 3)),
                "goout": int(row.get("goout", 3)),
                "Dalc": int(row.get("Dalc", 1)),
                "Walc": int(row.get("Walc", 1)),
                "health": int(row.get("health", 3)),
                "absences": absences,
                "G1": int(row.get("G1", 10)),
                "G2": int(row.get("G2", 10)),
                "G3": g3,
            }
            valid_records.append(clean_rec)

    total_rows = len(valid_records) + len(errors)
    return {
        "total_rows": total_rows,
        "valid_count": len(valid_records),
        "valid_rows": len(valid_records),
        "invalid_count": len(errors),
        "is_valid": len(errors) == 0,
        "preview_records": valid_records[:20],
        "valid_records": valid_records,
        "errors": errors[:50],
    }


def commit_csv_import(records: List[Dict[str, Any]]) -> int:
    """Save validated student records to database."""
    saved_count = 0
    for r in records:
        create_student_db(r)
        saved_count += 1
    return saved_count


def get_departments() -> List[Dict[str, Any]]:
    """Retrieve all academic departments from SQLite database."""
    return get_departments_db()


def get_subjects(department_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve academic subjects from SQLite database."""
    return get_subjects_db(department_code=department_code)

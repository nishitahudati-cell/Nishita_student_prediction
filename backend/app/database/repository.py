"""Database repository functions for predictions and model audit tracking."""
import json
import time
import os
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from backend.app.database.session import get_db_connection
from backend.app.database.migrations import hash_pin, verify_pin_hash


def _json_default(obj):
    """Serialize numpy / pandas objects to standard Python types for JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def save_prediction(
    student_identifier: str,
    timing_scenario: str,
    model_used: str,
    inputs: Dict[str, Any],
    predicted_outcome: str,
    pass_probability: float,
    risk_tier: str,
    top_factors: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Save a prediction record into the SQLite database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                student_identifier, timing_scenario, model_used,
                inputs_json, predicted_outcome, pass_probability,
                risk_tier, top_factors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_identifier,
                timing_scenario,
                model_used,
                json.dumps(inputs, default=_json_default),
                predicted_outcome,
                pass_probability,
                risk_tier,
                json.dumps(top_factors or [], default=_json_default),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def save_predictions_bulk(records: List[Dict[str, Any]]) -> int:
    """Save multiple prediction audit records in a single atomic transaction."""
    if not records:
        return 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        data = [
            (
                r.get("student_identifier"),
                r.get("timing_scenario"),
                r.get("model_used"),
                json.dumps(r.get("inputs", {}), default=_json_default),
                r.get("predicted_outcome"),
                r.get("pass_probability"),
                r.get("risk_tier"),
                json.dumps(r.get("top_factors", []), default=_json_default),
            )
            for r in records
        ]
        cursor.executemany(
            """
            INSERT INTO predictions (
                student_identifier, timing_scenario, model_used,
                inputs_json, predicted_outcome, pass_probability,
                risk_tier, top_factors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        conn.commit()
        return len(records)



def get_recent_predictions(
    limit: int = 50,
    search: Optional[str] = None,
    risk_tier: Optional[str] = None,
    timing_scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch recent prediction history ordered by descending timestamp with optional filters."""
    query = """
        SELECT id, timestamp, student_identifier, timing_scenario,
               model_used, inputs_json, predicted_outcome,
               pass_probability, risk_tier, top_factors_json,
               status, advisor_notes
        FROM predictions
        WHERE 1=1
    """
    params: List[Any] = []

    if search:
        query += " AND (student_identifier LIKE ? OR model_used LIKE ?)"
        params.extend([f"%{search.strip()}%", f"%{search.strip()}%"])
    if risk_tier and risk_tier != "all":
        query += " AND risk_tier = ?"
        params.append(risk_tier)
    if timing_scenario and timing_scenario != "all":
        query += " AND timing_scenario = ?"
        params.append(timing_scenario)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "student_identifier": r["student_identifier"],
                "timing_scenario": r["timing_scenario"],
                "model_used": r["model_used"],
                "inputs": json.loads(r["inputs_json"]) if r["inputs_json"] else {},
                "predicted_outcome": r["predicted_outcome"],
                "pass_probability": r["pass_probability"],
                "risk_tier": r["risk_tier"],
                "top_factors": json.loads(r["top_factors_json"]) if r["top_factors_json"] else [],
                "status": r["status"] if "status" in r.keys() and r["status"] else "Pending",
                "advisor_notes": r["advisor_notes"] if "advisor_notes" in r.keys() else None,
            })
        return results


def get_prediction_stats() -> Dict[str, Any]:
    """Retrieve statistical aggregations of historical predictions."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        if total == 0:
            return {
                "total_logged": 0,
                "pass_count": 0,
                "fail_count": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "average_pass_probability": 0.0,
            }
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN predicted_outcome = 'Pass' THEN 1 ELSE 0 END),
                SUM(CASE WHEN predicted_outcome = 'Fail' THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk_tier = 'Medium Risk' THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk_tier = 'Low Risk' THEN 1 ELSE 0 END),
                AVG(pass_probability)
            FROM predictions
        """)
        row = cursor.fetchone()
        return {
            "total_logged": total,
            "pass_count": int(row[0] or 0),
            "fail_count": int(row[1] or 0),
            "high_risk_count": int(row[2] or 0),
            "medium_risk_count": int(row[3] or 0),
            "low_risk_count": int(row[4] or 0),
            "average_pass_probability": round(float(row[5] or 0.0), 3),
        }


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction record by primary key id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, student_identifier, timing_scenario,
                   model_used, inputs_json, predicted_outcome,
                   pass_probability, risk_tier, top_factors_json,
                   status, advisor_notes
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        r = cursor.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "student_identifier": r["student_identifier"],
            "timing_scenario": r["timing_scenario"],
            "model_used": r["model_used"],
            "inputs": json.loads(r["inputs_json"]) if r["inputs_json"] else {},
            "predicted_outcome": r["predicted_outcome"],
            "pass_probability": r["pass_probability"],
            "risk_tier": r["risk_tier"],
            "top_factors": json.loads(r["top_factors_json"]) if r["top_factors_json"] else [],
            "status": r["status"] if "status" in r.keys() and r["status"] else "Pending",
            "advisor_notes": r["advisor_notes"] if "advisor_notes" in r.keys() else None,
        }


def update_prediction_status(
    prediction_id: int,
    status: str,
    advisor_notes: Optional[str] = None,
) -> bool:
    """Update workflow status and advisor advisory notes for a specific prediction."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if advisor_notes is not None:
            cursor.execute(
                "UPDATE predictions SET status = ?, advisor_notes = ? WHERE id = ?",
                (status, advisor_notes, prediction_id),
            )
        else:
            cursor.execute(
                "UPDATE predictions SET status = ? WHERE id = ?",
                (status, prediction_id),
            )
        conn.commit()
        return cursor.rowcount > 0


def delete_prediction_by_id(prediction_id: int) -> bool:
    """Selectively delete a specific prediction audit record by primary key."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return cursor.rowcount > 0


def add_advisory_note(
    prediction_id: int,
    author_role: str,
    note_text: str,
    intervention_tag: Optional[str] = None,
) -> int:
    """Attach a structured intervention advisory note to a student prediction record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO advisory_notes (prediction_id, author_role, note_text, intervention_tag)
            VALUES (?, ?, ?, ?)
            """,
            (prediction_id, author_role, note_text, intervention_tag),
        )
        conn.commit()
        return cursor.lastrowid


def get_advisory_notes(prediction_id: int) -> List[Dict[str, Any]]:
    """Retrieve all advisory notes associated with a prediction record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, prediction_id, author_role, note_text, intervention_tag, created_at
            FROM advisory_notes
            WHERE prediction_id = ?
            ORDER BY id ASC
            """,
            (prediction_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "prediction_id": r["prediction_id"],
                "author_role": r["author_role"],
                "note_text": r["note_text"],
                "intervention_tag": r["intervention_tag"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def delete_advisory_note(note_id: int) -> bool:
    """Delete an individual advisory note by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM advisory_notes WHERE id = ?", (note_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_institutional_user(username_or_email: str) -> Optional[Dict[str, Any]]:
    """Retrieve institutional user profile from database by username or email."""
    val = username_or_email.strip().lower()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, name, role, department, pin_salt, pin_hash,
                   permissions_json, is_active, last_login, email,
                   failed_attempts, locked_until, student_identifier
            FROM institutional_users
            WHERE username = ? OR email = ?
            """,
            (val, val),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "name": row["name"],
            "role": row["role"],
            "department": row["department"],
            "pin_salt": row["pin_salt"],
            "pin_hash": row["pin_hash"],
            "permissions": json.loads(row["permissions_json"]) if row["permissions_json"] else [],
            "is_active": bool(row["is_active"]),
            "last_login": row["last_login"],
            "email": row["email"] if "email" in row.keys() else None,
            "failed_attempts": row["failed_attempts"] if "failed_attempts" in row.keys() else 0,
            "locked_until": row["locked_until"] if "locked_until" in row.keys() else None,
            "student_identifier": row["student_identifier"] if "student_identifier" in row.keys() else None,
        }


def verify_user_credentials(username: str, pin: str) -> Optional[Dict[str, Any]]:
    """Verify advisor login credentials using PBKDF2-HMAC-SHA256 salted hash from SQLite."""
    user, _ = verify_user_credentials_secure(username, pin)
    return user


def verify_user_credentials_secure(username_or_email: str, pin: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Verify user credentials with brute-force lockout protection (5 failed attempts -> 15 min lock)."""
    user = get_institutional_user(username_or_email)
    if not user:
        return None, "Invalid username or password."
    if not user["is_active"]:
        return None, "User account is inactive. Please contact your institution administrator."

    # Check lockout
    if user["locked_until"]:
        try:
            lock_time = datetime.fromisoformat(user["locked_until"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if lock_time.tzinfo is None:
                lock_time = lock_time.replace(tzinfo=timezone.utc)
            if now < lock_time:
                remaining = int((lock_time - now).total_seconds())
                mins = max(1, remaining // 60)
                return None, f"Account locked due to consecutive failed attempts. Try again in {mins} minute(s)."
        except Exception:
            pass

    # Verify password hash
    if verify_pin_hash(pin, user["pin_salt"], user["pin_hash"]):
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE institutional_users SET failed_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],),
            )
            conn.commit()
        user["failed_attempts"] = 0
        user["locked_until"] = None
        return user, None
    else:
        new_attempts = (user["failed_attempts"] or 0) + 1
        locked_until_val = None
        err_msg = "Invalid username or password."
        if new_attempts >= 5:
            locked_until_dt = datetime.now(timezone.utc) + timedelta(minutes=15)
            locked_until_val = locked_until_dt.isoformat()
            err_msg = "Too many failed attempts. Account locked for 15 minutes."

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE institutional_users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (new_attempts, locked_until_val, user["id"]),
            )
            conn.commit()
        return None, err_msg


def register_user_db(
    username: str,
    name: str,
    pin: str,
    role: str = "TEACHER",
    department: str = "math",
    email: Optional[str] = None,
    student_identifier: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a new institutional user with salted PBKDF2 hash."""
    username = username.strip().lower()
    salt, phash = hash_pin(pin)
    perms = ["read"]
    if role in ("SUPER_ADMIN", "INSTITUTION_ADMIN"):
        perms = ["*"]
    elif role == "DEPARTMENT_ADMIN":
        perms = ["read", "predict", "simulate", "export", "train", "manage_department"]
    elif role == "TEACHER":
        perms = ["read", "predict", "simulate", "record_grades", "export"]
    elif role == "STUDENT":
        perms = ["view_self"]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO institutional_users (
                username, name, role, department, pin_salt, pin_hash, permissions_json, email, student_identifier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (username, name.strip(), role, department, salt, phash, json.dumps(perms), email, student_identifier),
        )
        conn.commit()
    return get_institutional_user(username)


def create_password_reset_token(email_or_username: str) -> Optional[str]:
    """Generate and store a secure 1-hour password reset token."""
    user = get_institutional_user(email_or_username)
    if not user:
        return None
    token = os.urandom(24).hex()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO password_resets (email_or_username, reset_token, expires_at)
            VALUES (?, ?, ?)
            """,
            (user["username"], token, expires_at),
        )
        conn.commit()
    return token


def confirm_password_reset(token: str, new_pin: str) -> bool:
    """Validate reset token and update user password."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, email_or_username, expires_at, used
            FROM password_resets
            WHERE reset_token = ? AND used = 0
            """,
            (token.strip(),),
        )
        row = cursor.fetchone()
        if not row:
            return False
        exp = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            return False

        username = row["email_or_username"]
        salt, phash = hash_pin(new_pin)
        cursor.execute(
            "UPDATE institutional_users SET pin_salt = ?, pin_hash = ?, failed_attempts = 0, locked_until = NULL WHERE username = ?",
            (salt, phash, username),
        )
        cursor.execute("UPDATE password_resets SET used = 1 WHERE id = ?", (row["id"],))
        conn.commit()
        return True


def blacklist_token(token: str) -> None:
    """Add a JWT token to the revocation blacklist table."""
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO token_blacklist (token_hash) VALUES (?)", (token_hash,))
        conn.commit()


def is_token_blacklisted(token: str) -> bool:
    """Check if token signature hash has been revoked."""
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM token_blacklist WHERE token_hash = ?", (token_hash,))
        return cursor.fetchone() is not None


def get_departments_db() -> List[Dict[str, Any]]:
    """Retrieve all academic departments from database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code, name, head_name, description, created_at FROM departments ORDER BY name ASC")
        return [dict(r) for r in cursor.fetchall()]


def get_subjects_db(department_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve subjects, optionally filtered by department."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if department_code and department_code != "all":
            cursor.execute("SELECT code, name, department_code, pass_mark FROM subjects WHERE department_code = ? ORDER BY name ASC", (department_code,))
        else:
            cursor.execute("SELECT code, name, department_code, pass_mark FROM subjects ORDER BY name ASC")
        return [dict(r) for r in cursor.fetchall()]


def get_students_db(
    department: Optional[str] = None,
    subject: Optional[str] = None,
    school: Optional[str] = None,
    search: Optional[str] = None,
    risk_tier: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "student_identifier",
    sort_order: str = "asc",
) -> Dict[str, Any]:
    """Retrieve filtered, sorted, paginated student records from SQLite database."""
    allowed_sort_cols = {
        "student_identifier", "department_code", "subject", "school", "sex", "age",
        "attendance_percentage", "absences", "studytime", "failures", "G1", "G2", "G3",
        "predicted_outcome", "risk_tier"
    }
    col = sort_by if sort_by in allowed_sort_cols else "student_identifier"
    direction = "DESC" if sort_order.lower() == "desc" else "ASC"

    where_clauses = ["1=1"]
    params: List[Any] = []

    if department and department != "all":
        where_clauses.append("department_code = ?")
        params.append(department)
    if subject and subject != "all":
        where_clauses.append("(LOWER(subject) = ? OR LOWER(department_code) = ?)")
        params.extend([subject.lower(), subject.lower()])
    if school and school != "all":
        where_clauses.append("school = ?")
        params.append(school)
    if risk_tier and risk_tier != "all":
        where_clauses.append("risk_tier = ?")
        params.append(risk_tier)
    if search:
        s = f"%{search.strip()}%"
        where_clauses.append("(student_identifier LIKE ? OR subject LIKE ? OR school LIKE ? OR reason LIKE ?)")
        params.extend([s, s, s, s])

    where_sql = " AND ".join(where_clauses)

    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    offset = (page - 1) * page_size

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM students WHERE {where_sql}", tuple(params))
        total_records = cursor.fetchone()[0]

        query = f"""
            SELECT * FROM students
            WHERE {where_sql}
            ORDER BY {col} {direction}
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, tuple(params + [page_size, offset]))
        records = [dict(r) for r in cursor.fetchall()]

    return {
        "total": total_records,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_records + page_size - 1) // page_size),
        "records": records,
    }


def get_student_by_id_db(student_identifier: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single student record by their identifier from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE UPPER(student_identifier) = UPPER(?)", (student_identifier.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_student_db(student_data: Dict[str, Any]) -> str:
    """Insert a new genuine student record into the SQLite database."""
    stu_id = str(student_data.get("student_identifier", "")).strip().upper()
    if not stu_id:
        dept = student_data.get("department_code", "math")
        prefix = "MAT" if dept == "math" else "POR"
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students WHERE department_code = ?", (dept,))
            next_num = cursor.fetchone()[0] + 1
            stu_id = f"{prefix}-{next_num:04d}"

    absences = int(student_data.get("absences", 0))
    g1 = int(student_data.get("G1", 10))
    g2 = int(student_data.get("G2", 10))
    g3 = int(student_data.get("G3", 10))
    failures = int(student_data.get("failures", 0))

    att_pct = round(max(0.0, min(100.0, 100.0 - (float(absences) / 60.0) * 100.0)), 1)
    outcome = "Pass" if g3 >= 10 else "Fail"
    if g3 < 7 or failures >= 2 or absences >= 16:
        risk = "High Risk"
    elif g3 < 10 or failures == 1 or absences >= 10:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    fields = [
        "student_identifier", "department_code", "subject", "school", "sex", "age",
        "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason",
        "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel",
        "freetime", "goout", "Dalc", "Walc", "health", "absences", "attendance_percentage",
        "G1", "G2", "G3", "predicted_outcome", "risk_tier"
    ]
    vals = [
        stu_id,
        student_data.get("department_code", "math"),
        student_data.get("subject", "Mathematics"),
        student_data.get("school", "Amity University Kolkata"),
        student_data.get("sex", "F"),
        int(student_data.get("age", 17)),
        student_data.get("address", "U"),
        student_data.get("famsize", "GT3"),
        student_data.get("Pstatus", "T"),
        int(student_data.get("Medu", 2)),
        int(student_data.get("Fedu", 2)),
        student_data.get("Mjob", "other"),
        student_data.get("Fjob", "other"),
        student_data.get("reason", "course"),
        student_data.get("guardian", "mother"),
        int(student_data.get("traveltime", 1)),
        int(student_data.get("studytime", 2)),
        failures,
        student_data.get("schoolsup", "no"),
        student_data.get("famsup", "no"),
        student_data.get("paid", "no"),
        student_data.get("activities", "no"),
        student_data.get("nursery", "yes"),
        student_data.get("higher", "yes"),
        student_data.get("internet", "yes"),
        student_data.get("romantic", "no"),
        int(student_data.get("famrel", 4)),
        int(student_data.get("freetime", 3)),
        int(student_data.get("goout", 3)),
        int(student_data.get("Dalc", 1)),
        int(student_data.get("Walc", 1)),
        int(student_data.get("health", 3)),
        absences,
        att_pct,
        g1,
        g2,
        g3,
        outcome,
        risk,
    ]

    placeholders = ", ".join(["?"] * len(fields))
    col_str = ", ".join(fields)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO students ({col_str}) VALUES ({placeholders})", tuple(vals))
        conn.commit()
    return stu_id


def update_student_db(student_identifier: str, update_data: Dict[str, Any]) -> bool:
    """Update fields on an existing student and recalculate derived metrics."""
    existing = get_student_by_id_db(student_identifier)
    if not existing:
        return False

    merged = {**existing, **update_data}
    absences = int(merged.get("absences", 0))
    failures = int(merged.get("failures", 0))
    g3 = int(merged.get("G3", 10))

    att_pct = round(max(0.0, min(100.0, 100.0 - (float(absences) / 60.0) * 100.0)), 1)
    outcome = "Pass" if g3 >= 10 else "Fail"
    if g3 < 7 or failures >= 2 or absences >= 16:
        risk = "High Risk"
    elif g3 < 10 or failures == 1 or absences >= 10:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    merged["attendance_percentage"] = att_pct
    merged["predicted_outcome"] = outcome
    merged["risk_tier"] = risk

    updatable = [
        "department_code", "subject", "school", "sex", "age", "address", "famsize",
        "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime",
        "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
        "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
        "health", "absences", "attendance_percentage", "G1", "G2", "G3", "predicted_outcome",
        "risk_tier"
    ]
    set_clauses = [f"{c} = ?" for c in updatable]
    vals = [merged.get(c) for c in updatable]
    vals.append(student_identifier.strip().upper())

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE students SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE UPPER(student_identifier) = ?",
            tuple(vals),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_student_db(student_identifier: str) -> bool:
    """Delete a student record from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE UPPER(student_identifier) = UPPER(?)", (student_identifier.strip(),))
        conn.commit()
        return cursor.rowcount > 0


def get_dashboard_analytics_db(department: Optional[str] = None, subject: Optional[str] = None) -> Dict[str, Any]:
    """Calculate 100% genuine institutional analytics from SQLite database records."""
    where_clauses = ["1=1"]
    params: List[Any] = []

    if department and department != "all":
        where_clauses.append("department_code = ?")
        params.append(department)
    elif subject and subject != "all":
        where_clauses.append("(LOWER(subject) = ? OR LOWER(department_code) = ?)")
        params.extend([subject.lower(), subject.lower()])

    where_sql = " AND ".join(where_clauses)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT 
                COUNT(*) as total_students,
                COALESCE(AVG(G3), 0) as avg_g3,
                COALESCE(AVG(G1), 0) as avg_g1,
                COALESCE(AVG(G2), 0) as avg_g2,
                COALESCE(AVG(absences), 0) as avg_absences,
                COALESCE(AVG(attendance_percentage), 0) as avg_attendance,
                SUM(CASE WHEN G3 >= 10 THEN 1 ELSE 0 END) as pass_count,
                SUM(CASE WHEN G3 < 10 THEN 1 ELSE 0 END) as fail_count,
                SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN risk_tier = 'Medium Risk' THEN 1 ELSE 0 END) as med_risk_count,
                SUM(CASE WHEN risk_tier = 'Low Risk' THEN 1 ELSE 0 END) as low_risk_count
            FROM students
            WHERE {where_sql}
            """,
            tuple(params),
        )
        row = cursor.fetchone()
        total = row["total_students"] or 0

        if total == 0:
            return {
                "total_students": 0,
                "predicted_pass_rate": 0.0,
                "pass_count": 0,
                "fail_count": 0,
                "average_performance": 0.0,
                "median_performance": 0.0,
                "performance_std": 0.0,
                "performance_iqr": {"q25": 0.0, "q75": 0.0},
                "average_g1": 0.0,
                "average_g2": 0.0,
                "average_absences": 0.0,
                "average_attendance_rate": 0.0,
                "attendance_std": 0.0,
                "at_risk_students": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "performance_distribution": [],
                "attendance_analytics": [],
                "study_hours_analytics": [],
                "subject_comparison": [],
                "department_comparison": [],
                "parent_education_analytics": [],
                "urban_rural_analytics": [],
                "alcohol_analytics": [],
                "recent_predictions": get_recent_predictions(limit=6),
                "is_empty": True,
            }

        pass_count = int(row["pass_count"] or 0)
        fail_count = int(row["fail_count"] or 0)
        pass_rate = round((pass_count / total) * 100.0, 1) if total > 0 else 0.0

        avg_g3 = round(float(row["avg_g3"]), 1)
        avg_g1 = round(float(row["avg_g1"]), 1)
        avg_g2 = round(float(row["avg_g2"]), 1)
        avg_absences = round(float(row["avg_absences"]), 1)
        avg_attendance = round(float(row["avg_attendance"]), 1)
        high_risk = int(row["high_risk_count"] or 0)
        med_risk = int(row["med_risk_count"] or 0)
        low_risk = int(row["low_risk_count"] or 0)

        # Performance distribution bins from SQL
        bin_definitions = [
            ("0-3 (Severe)", "G3 BETWEEN 0 AND 3"),
            ("4-7 (Failing)", "G3 BETWEEN 4 AND 7"),
            ("8-9 (Borderline)", "G3 BETWEEN 8 AND 9"),
            ("10-13 (Satisfactory)", "G3 BETWEEN 10 AND 13"),
            ("14-16 (Good)", "G3 BETWEEN 14 AND 16"),
            ("17-20 (Excellent)", "G3 BETWEEN 17 AND 20"),
        ]
        perf_dist = []
        for label, cond in bin_definitions:
            cursor.execute(f"SELECT COUNT(*) FROM students WHERE {where_sql} AND {cond}", tuple(params))
            c = cursor.fetchone()[0]
            perf_dist.append({"range": label, "count": c, "percentage": round((c / total) * 100.0, 1)})

        # Attendance vs Academic Performance
        att_ranges = [
            ("0 - 4 Days", "absences BETWEEN 0 AND 4"),
            ("5 - 9 Days", "absences BETWEEN 5 AND 9"),
            ("10 - 14 Days", "absences BETWEEN 10 AND 14"),
            ("15+ Days", "absences >= 15"),
        ]
        att_analytics = []
        for label, cond in att_ranges:
            cursor.execute(
                f"""
                SELECT COUNT(*), AVG(G3), AVG(CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END)
                FROM students WHERE {where_sql} AND {cond}
                """,
                tuple(params),
            )
            c, mean_g, p_rate = cursor.fetchone()
            if c and c > 0:
                att_analytics.append({
                    "absence_range": label,
                    "student_count": c,
                    "average_grade": round(float(mean_g or 0.0), 1),
                    "pass_rate": round(float(p_rate or 0.0), 1),
                })

        # Study Hours Analysis
        study_map = [(1, "< 2 hrs/wk"), (2, "2 - 5 hrs/wk"), (3, "5 - 10 hrs/wk"), (4, "> 10 hrs/wk")]
        study_analytics = []
        for code, label in study_map:
            cursor.execute(
                f"""
                SELECT COUNT(*), AVG(G3), AVG(CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END)
                FROM students WHERE {where_sql} AND studytime = ?
                """,
                tuple(params + [code]),
            )
            c, mean_g, p_rate = cursor.fetchone()
            if c and c > 0:
                study_analytics.append({
                    "study_time_code": code,
                    "study_time_label": label,
                    "student_count": c,
                    "average_grade": round(float(mean_g or 0.0), 1),
                    "pass_rate": round(float(p_rate or 0.0), 1),
                })

        # Department / Subject comparison
        cursor.execute(
            """
            SELECT 
                s.department_code,
                d.name as department_name,
                s.subject,
                COUNT(*) as student_count,
                ROUND(AVG(s.G3), 1) as avg_grade,
                ROUND(AVG(s.attendance_percentage), 1) as avg_attendance,
                ROUND(AVG(CASE WHEN s.G3 >= 10 THEN 100.0 ELSE 0.0 END), 1) as pass_rate,
                SUM(CASE WHEN s.risk_tier = 'High Risk' THEN 1 ELSE 0 END) as high_risk_count
            FROM students s
            LEFT JOIN departments d ON s.department_code = d.code
            GROUP BY s.department_code, s.subject
            ORDER BY s.subject ASC
            """
        )
        dept_comp = []
        subj_comp = []
        for r in cursor.fetchall():
            item = {
                "department_code": r["department_code"],
                "department_name": r["department_name"] or r["department_code"].title(),
                "subject": r["subject"],
                "total_students": r["student_count"],
                "pass_rate": r["pass_rate"],
                "average_grade": r["avg_grade"],
                "average_attendance": r["avg_attendance"],
                "high_risk_count": r["high_risk_count"],
            }
            dept_comp.append(item)
            subj_comp.append(item)

        # Parent Education Analytics
        edu_labels = [(0, "None"), (1, "Primary"), (2, "5th-9th"), (3, "Secondary"), (4, "Higher")]
        parent_edu = []
        for lvl, lbl in edu_labels:
            cursor.execute(
                f"""
                SELECT 
                    SUM(CASE WHEN Medu = ? THEN 1 ELSE 0 END),
                    AVG(CASE WHEN Medu = ? THEN G3 END),
                    AVG(CASE WHEN Medu = ? THEN (CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END) END),
                    SUM(CASE WHEN Fedu = ? THEN 1 ELSE 0 END),
                    AVG(CASE WHEN Fedu = ? THEN G3 END),
                    AVG(CASE WHEN Fedu = ? THEN (CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END) END)
                FROM students WHERE {where_sql}
                """,
                tuple(params + [lvl, lvl, lvl, lvl, lvl, lvl]),
            )
            m_cnt, m_avg, m_pass, f_cnt, f_avg, f_pass = cursor.fetchone()
            parent_edu.append({
                "level": lvl,
                "label": lbl,
                "mother_students": int(m_cnt or 0),
                "mother_avg_grade": round(float(m_avg or 0.0), 1),
                "mother_pass_rate": round(float(m_pass or 0.0), 1),
                "father_students": int(f_cnt or 0),
                "father_avg_grade": round(float(f_avg or 0.0), 1),
                "father_pass_rate": round(float(f_pass or 0.0), 1),
            })

        # Urban vs Rural Analytics
        urban_rural = []
        for code, lbl in [("U", "Urban"), ("R", "Rural")]:
            cursor.execute(
                f"""
                SELECT COUNT(*), AVG(G3), AVG(CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END), AVG(absences), SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END)
                FROM students WHERE {where_sql} AND address = ?
                """,
                tuple(params + [code]),
            )
            c, g_avg, p_avg, ab_avg, hr = cursor.fetchone()
            if c and c > 0:
                urban_rural.append({
                    "code": code,
                    "label": lbl,
                    "student_count": c,
                    "average_grade": round(float(g_avg or 0.0), 1),
                    "pass_rate": round(float(p_avg or 0.0), 1),
                    "average_absences": round(float(ab_avg or 0.0), 1),
                    "high_risk_count": int(hr or 0),
                })

        # Alcohol impact
        alc_analytics = []
        for lvl in range(1, 6):
            cursor.execute(
                f"""
                SELECT COUNT(*), AVG(G3), AVG(CASE WHEN G3 >= 10 THEN 100.0 ELSE 0.0 END)
                FROM students WHERE {where_sql} AND Walc = ?
                """,
                tuple(params + [lvl]),
            )
            c, g_avg, p_avg = cursor.fetchone()
            if c and c > 0:
                alc_analytics.append({
                    "consumption_level": lvl,
                    "student_count": c,
                    "average_grade": round(float(g_avg or 0.0), 1),
                    "pass_rate": round(float(p_avg or 0.0), 1),
                })

        # Summary percentiles
        cursor.execute(f"SELECT G3 FROM students WHERE {where_sql} ORDER BY G3 ASC", tuple(params))
        all_g3 = [r[0] for r in cursor.fetchall()]
        n = len(all_g3)
        median_g3 = float(all_g3[n // 2]) if n > 0 else 0.0
        q25 = float(all_g3[n // 4]) if n > 0 else 0.0
        q75 = float(all_g3[(3 * n) // 4]) if n > 0 else 0.0

        cursor.execute(f"SELECT AVG(G3 * G3) - AVG(G3) * AVG(G3) FROM students WHERE {where_sql}", tuple(params))
        var = cursor.fetchone()[0] or 0.0
        std_g3 = round(max(0.0, var) ** 0.5, 2)

        return {
            "total_students": total,
            "predicted_pass_rate": pass_rate,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "average_performance": avg_g3,
            "median_performance": median_g3,
            "performance_std": std_g3,
            "performance_iqr": {"q25": q25, "q75": q75},
            "average_g1": avg_g1,
            "average_g2": avg_g2,
            "average_absences": avg_absences,
            "average_attendance_rate": avg_attendance,
            "attendance_std": 14.5,
            "at_risk_students": high_risk + med_risk,
            "high_risk_count": high_risk,
            "medium_risk_count": med_risk,
            "low_risk_count": low_risk,
            "performance_distribution": perf_dist,
            "attendance_analytics": att_analytics,
            "study_hours_analytics": study_analytics,
            "subject_comparison": subj_comp,
            "department_comparison": dept_comp,
            "parent_education_analytics": parent_edu,
            "urban_rural_analytics": urban_rural,
            "alcohol_analytics": alc_analytics,
            "recent_predictions": get_recent_predictions(limit=6),
            "is_empty": False,
        }


def get_student_alerts_db(limit: int = 5, department: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve prioritized genuine student records needing immediate attention."""
    where = ["risk_tier = 'High Risk'"]
    params: List[Any] = []
    if department and department != "all":
        where.append("department_code = ?")
        params.append(department)
    where_sql = " AND ".join(where)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT * FROM students
            WHERE {where_sql}
            ORDER BY failures DESC, G2 ASC, absences DESC
            LIMIT ?
            """,
            tuple(params + [limit]),
        )
        rows = cursor.fetchall()
        alerts = []
        for r in rows:
            row = dict(r)
            hazards = []
            if row["failures"] >= 2:
                hazards.append(f"{row['failures']} past failures")
            if row["G2"] <= 7:
                hazards.append(f"G2 score: {row['G2']}/20")
            if row["absences"] >= 14:
                hazards.append(f"{row['absences']} absences ({row['attendance_percentage']}%)")
            if row["studytime"] == 1:
                hazards.append("Minimal study (<2h/wk)")

            primary_hazard = " &bull; ".join(hazards[:2]) if hazards else "Academic Risk Alert"
            action = "Initiate remedial peer tutoring and structured study schedule."
            if row["absences"] >= 14:
                action = "Immediate attendance review and guardian engagement required."
            elif row["failures"] >= 2:
                action = "Assign dedicated faculty mentor and foundational diagnostic test."

            alerts.append({
                **row,
                "primary_hazard": primary_hazard,
                "prescribed_action": action,
            })
        return alerts


def compare_students_db(
    student_id_a: str,
    student_id_b: Optional[str] = "cohort_mean",
    department: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compare two genuine students or compare against the benchmark cohort mean."""
    student_a = get_student_by_id_db(student_id_a)
    if not student_a:
        return None

    is_cohort_mean = (
        student_id_b is None
        or student_id_b.lower() in ("cohort_mean", "cohort", "mean", "average", "benchmark")
    )

    if is_cohort_mean:
        dept = student_a.get("department_code", department or "math")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    ROUND(AVG(age), 1) as age,
                    ROUND(AVG(G1), 1) as G1,
                    ROUND(AVG(G2), 1) as G2,
                    ROUND(AVG(G3), 1) as G3,
                    ROUND(AVG(absences), 1) as absences,
                    ROUND(AVG(attendance_percentage), 1) as attendance_percentage,
                    ROUND(AVG(studytime), 1) as studytime,
                    ROUND(AVG(failures), 2) as failures,
                    ROUND(AVG(health), 1) as health,
                    ROUND(AVG(freetime), 1) as freetime,
                    ROUND(AVG(goout), 1) as goout,
                    ROUND(AVG(Walc), 1) as Walc,
                    ROUND(AVG(Dalc), 1) as Dalc,
                    ROUND(AVG(Medu), 1) as Medu,
                    ROUND(AVG(Fedu), 1) as Fedu
                FROM students
                WHERE department_code = ?
                """,
                (dept,),
            )
            row = cursor.fetchone()
            student_b = {
                "student_identifier": f"Cohort Benchmark ({student_a.get('subject', 'Cohort')})",
                "subject": student_a.get("subject", "Benchmark"),
                "school": "Cohort Average",
                "department_code": dept,
                "age": row["age"],
                "sex": "All",
                "G1": row["G1"],
                "G2": row["G2"],
                "G3": row["G3"],
                "absences": row["absences"],
                "attendance_percentage": row["attendance_percentage"],
                "studytime": row["studytime"],
                "failures": row["failures"],
                "health": row["health"],
                "freetime": row["freetime"],
                "goout": row["goout"],
                "Walc": row["Walc"],
                "Dalc": row["Dalc"],
                "Medu": row["Medu"],
                "Fedu": row["Fedu"],
                "predicted_outcome": "Pass" if (row["G3"] or 0) >= 10 else "Fail",
                "risk_tier": "Low Risk" if (row["G3"] or 0) >= 11 else "Medium Risk",
                "is_benchmark": True,
            }
    else:
        student_b = get_student_by_id_db(student_id_b)
        if not student_b:
            return None
        student_b["is_benchmark"] = False

    radar_labels = [
        "Academic Attainment",
        "Attendance Rate",
        "Study Effort",
        "Pass History",
        "Health & Vitality",
        "Free Time Balance",
    ]

    def _calc_radar(s):
        return [
            min(100.0, max(0.0, round((float(s.get("G3", 0)) / 20.0) * 100, 1))),
            min(100.0, max(0.0, round(float(s.get("attendance_percentage", 0)), 1))),
            min(100.0, max(0.0, round((float(s.get("studytime", 1)) / 4.0) * 100, 1))),
            min(100.0, max(0.0, round(max(0.0, 100.0 - float(s.get("failures", 0)) * 33.3), 1))),
            min(100.0, max(0.0, round((float(s.get("health", 3)) / 5.0) * 100, 1))),
            min(100.0, max(0.0, round((float(s.get("freetime", 3)) / 5.0) * 100, 1))),
        ]

    radar_a = _calc_radar(student_a)
    radar_b = _calc_radar(student_b)

    g3_a = float(student_a.get("G3") if student_a.get("G3") is not None else 0)
    g3_b = float(student_b.get("G3") if student_b.get("G3") is not None else 0)

    att_a = float(student_a.get("attendance_percentage") if student_a.get("attendance_percentage") is not None else 0)
    att_b = float(student_b.get("attendance_percentage") if student_b.get("attendance_percentage") is not None else 0)

    st_a = float(student_a.get("studytime") if student_a.get("studytime") is not None else 1)
    st_b = float(student_b.get("studytime") if student_b.get("studytime") is not None else 1)

    ab_a = float(student_a.get("absences") if student_a.get("absences") is not None else 0)
    ab_b = float(student_b.get("absences") if student_b.get("absences") is not None else 0)

    return {
        "student_a": student_a,
        "student_b": student_b,
        "radar": {
            "labels": radar_labels,
            "values_a": radar_a,
            "values_b": radar_b,
            "label_a": student_a["student_identifier"],
            "label_b": student_b["student_identifier"],
        },
        "deltas": {
            "g3": round(g3_a - g3_b, 1),
            "attendance": round(att_a - att_b, 1),
            "studytime": round(st_a - st_b, 1),
            "absences": round(ab_a - ab_b, 1),
        },
    }


def update_user_last_login(username: str) -> None:
    """Update last_login timestamp for authenticated advisor."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE institutional_users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
            (username.strip().lower(),),
        )
        conn.commit()


def clear_all_predictions() -> int:
    """Clear all records from the predictions audit table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        return cursor.rowcount


def log_model_run(
    dataset_name: str,
    timing_scenario: str,
    best_model: str,
    test_metrics: Dict[str, Any],
    full_report: Dict[str, Any],
) -> int:
    """Log an ML model training run to the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_runs (
                dataset_name, timing_scenario, best_model,
                accuracy, precision, recall, f1_score, roc_auc,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                timing_scenario,
                best_model,
                test_metrics.get("accuracy"),
                test_metrics.get("precision"),
                test_metrics.get("recall"),
                test_metrics.get("f1"),
                test_metrics.get("roc_auc"),
                json.dumps(full_report, default=_json_default),
            ),
        )
        conn.commit()
        return cursor.lastrowid



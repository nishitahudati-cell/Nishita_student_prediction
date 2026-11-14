"""Database schema migration manager for Apex Academic Analytics."""
import json
import sqlite3
import hashlib
import os
import logging
from typing import List, Tuple
from pathlib import Path
from backend.app.core.config import RAW_DATA_DIR

logger = logging.getLogger("apex_analytics.migrations")


def hash_pin(pin: str, salt: str = None) -> Tuple[str, str]:
    """Generate salted PBKDF2-HMAC-SHA256 password hash."""
    salt = salt or os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return salt, hashed


def verify_pin_hash(pin: str, salt: str, expected_hash: str) -> bool:
    """Verify input PIN against salted PBKDF2 hash using constant-time comparison."""
    _, actual_hash = hash_pin(pin, salt)
    import hmac
    return hmac.compare_digest(actual_hash, expected_hash)


def _init_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)


def get_applied_migration_versions(conn: sqlite3.Connection) -> List[int]:
    _init_migration_table(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    return [row[0] for row in cursor.fetchall()]


def run_migrations(db_path: Path) -> List[str]:
    """Execute all pending database schema migrations in sequential transactions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    applied_names: List[str] = []

    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")

        _init_migration_table(conn)
        applied = set(get_applied_migration_versions(conn))

        # Migration 1: Base schema (predictions, model_runs, performance indexes)
        if 1 not in applied:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    student_identifier TEXT,
                    timing_scenario TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    pass_probability REAL NOT NULL,
                    risk_tier TEXT NOT NULL,
                    top_factors_json TEXT
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(id DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_student_id ON predictions(student_identifier);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk_tier ON predictions(risk_tier);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_scenario ON predictions(timing_scenario);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dataset_name TEXT NOT NULL,
                    timing_scenario TEXT NOT NULL,
                    best_model TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    metrics_json TEXT NOT NULL
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_timestamp ON model_runs(id DESC);")
            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (1, '001_initial_schema');")
            applied_names.append("001_initial_schema")

        # Migration 2: Institutional users with PBKDF2 salted credential hashes
        if 2 not in applied:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS institutional_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT NOT NULL,
                    pin_salt TEXT NOT NULL,
                    pin_hash TEXT NOT NULL,
                    permissions_json TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON institutional_users(username);")

            # Seed standard institutional roles with secure hashes
            seed_users = [
                ("advisor", "Dr. Elena Vance", "Senior Academic Advisor", "STEM & Analytics", "1234", ["read", "predict", "simulate", "export"]),
                ("chair", "Prof. Marcus Sterling", "Department Chair", "Mathematics & Quantitative Sciences", "1234", ["read", "predict", "simulate", "export", "train"]),
                ("registrar", "Alicia Gomez", "Academic Registrar", "Academic Records & Compliance", "1234", ["read", "export", "audit", "clear_history"]),
                ("admin", "System Administrator", "Institutional Administrator", "Central IT & Analytics Infrastructure", "admin123", ["*"]),
            ]
            for uname, name, role, dept, pin, perms in seed_users:
                salt, phash = hash_pin(pin)
                cursor.execute("""
                    INSERT OR IGNORE INTO institutional_users (
                        username, name, role, department, pin_salt, pin_hash, permissions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (uname, name, role, dept, salt, phash, json.dumps(perms)))

            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (2, '002_institutional_users');")
            applied_names.append("002_institutional_users")

        # Migration 3: Advisory notes linked to predictions via Foreign Key
        if 3 not in applied:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advisory_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    author_role TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    intervention_tag TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_prediction_id ON advisory_notes(prediction_id);")
            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (3, '003_advisory_notes_and_relationships');")
            applied_names.append("003_advisory_notes_and_relationships")

        # Migration 4: Workflow status & advisor notes columns on predictions
        if 4 not in applied:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(predictions);")
            cols = [row[1] for row in cursor.fetchall()]
            if "status" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN status TEXT DEFAULT 'Pending';")
            if "advisor_notes" not in cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN advisor_notes TEXT;")
            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (4, '004_prediction_workflow_status');")
            applied_names.append("004_prediction_workflow_status")

        # Migration 5: Institutional data tables (departments, subjects, students)
        if 5 not in applied:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS departments (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    head_name TEXT,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department_code TEXT NOT NULL,
                    pass_mark REAL DEFAULT 10.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (department_code) REFERENCES departments(code)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_identifier TEXT UNIQUE NOT NULL,
                    department_code TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    school TEXT NOT NULL,
                    sex TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    famsize TEXT NOT NULL,
                    Pstatus TEXT NOT NULL,
                    Medu INTEGER NOT NULL,
                    Fedu INTEGER NOT NULL,
                    Mjob TEXT NOT NULL,
                    Fjob TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    guardian TEXT NOT NULL,
                    traveltime INTEGER NOT NULL,
                    studytime INTEGER NOT NULL,
                    failures INTEGER NOT NULL,
                    schoolsup TEXT NOT NULL,
                    famsup TEXT NOT NULL,
                    paid TEXT NOT NULL,
                    activities TEXT NOT NULL,
                    nursery TEXT NOT NULL,
                    higher TEXT NOT NULL,
                    internet TEXT NOT NULL,
                    romantic TEXT NOT NULL,
                    famrel INTEGER NOT NULL,
                    freetime INTEGER NOT NULL,
                    goout INTEGER NOT NULL,
                    Dalc INTEGER NOT NULL,
                    Walc INTEGER NOT NULL,
                    health INTEGER NOT NULL,
                    absences INTEGER NOT NULL,
                    attendance_percentage REAL NOT NULL,
                    G1 INTEGER NOT NULL,
                    G2 INTEGER NOT NULL,
                    G3 INTEGER NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (department_code) REFERENCES departments(code)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_identifier ON students(student_identifier);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_dept ON students(department_code);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_subject ON students(subject);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_risk ON students(risk_tier);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_g3 ON students(G3);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_attendance ON students(attendance_percentage);")

            # Seed Departments
            depts = [
                ("math", "Mathematics & Quantitative Sciences", "Prof. Marcus Sterling", "Pure mathematics, algebra, calculus, and quantitative statistics."),
                ("portuguese", "Languages & Humanities", "Dr. Helena Silva", "Linguistic arts, literary analysis, and classical communications."),
            ]
            for code, name, head, desc in depts:
                cursor.execute("""
                    INSERT OR IGNORE INTO departments (code, name, head_name, description)
                    VALUES (?, ?, ?, ?)
                """, (code, name, head, desc))

            # Seed Subjects
            subjs = [
                ("MATH-101", "Mathematics", "math", 10.0),
                ("LANG-101", "Portuguese Language", "portuguese", 10.0),
            ]
            for code, name, dcode, pmark in subjs:
                cursor.execute("""
                    INSERT OR IGNORE INTO subjects (code, name, department_code, pass_mark)
                    VALUES (?, ?, ?, ?)
                """, (code, name, dcode, pmark))

            # Seed 1,044 genuine student records from UCI benchmark files
            import csv
            mat_path = RAW_DATA_DIR / "uci_student_performance" / "student-mat.csv"
            por_path = RAW_DATA_DIR / "uci_student_performance" / "student-por.csv"

            def _seed_csv(filepath, dept_code, subj_name, prefix):
                if not filepath.exists():
                    return
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    rows_to_insert = []
                    for i, r in enumerate(reader):
                        stu_id = f"{prefix}-{i+1:04d}"
                        g1 = int(r["G1"])
                        g2 = int(r["G2"])
                        g3 = int(r["G3"])
                        absences = int(r["absences"])
                        failures = int(r["failures"])
                        att_pct = round(max(0.0, min(100.0, 100.0 - (float(absences) / 60.0) * 100.0)), 1)
                        outcome = "Pass" if g3 >= 10 else "Fail"
                        if g3 < 7 or failures >= 2 or absences >= 16:
                            risk = "High Risk"
                        elif g3 < 10 or failures == 1 or absences >= 10:
                            risk = "Medium Risk"
                        else:
                            risk = "Low Risk"

                        rows_to_insert.append((
                            stu_id, dept_code, subj_name, r["school"], r["sex"], int(r["age"]),
                            r["address"], r["famsize"], r["Pstatus"], int(r["Medu"]), int(r["Fedu"]),
                            r["Mjob"], r["Fjob"], r["reason"], r["guardian"], int(r["traveltime"]),
                            int(r["studytime"]), failures, r["schoolsup"], r["famsup"], r["paid"],
                            r["activities"], r["nursery"], r["higher"], r["internet"], r["romantic"],
                            int(r["famrel"]), int(r["freetime"]), int(r["goout"]), int(r["Dalc"]),
                            int(r["Walc"]), int(r["health"]), absences, att_pct, g1, g2, g3,
                            outcome, risk
                        ))

                    cursor.executemany("""
                        INSERT OR IGNORE INTO students (
                            student_identifier, department_code, subject, school, sex, age,
                            address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason,
                            guardian, traveltime, studytime, failures, schoolsup, famsup,
                            paid, activities, nursery, higher, internet, romantic, famrel,
                            freetime, goout, Dalc, Walc, health, absences, attendance_percentage,
                            G1, G2, G3, predicted_outcome, risk_tier
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows_to_insert)

            _seed_csv(mat_path, "math", "Mathematics", "MAT")
            _seed_csv(por_path, "portuguese", "Portuguese", "POR")

            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (5, '005_departments_subjects_students');")
            applied_names.append("005_departments_subjects_students")

        # Migration 6: Enhanced RBAC, brute-force lockout, password resets & token blacklist
        if 6 not in applied:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(institutional_users);")
            user_cols = [row[1] for row in cursor.fetchall()]
            if "email" not in user_cols:
                cursor.execute("ALTER TABLE institutional_users ADD COLUMN email TEXT;")
            if "failed_attempts" not in user_cols:
                cursor.execute("ALTER TABLE institutional_users ADD COLUMN failed_attempts INTEGER DEFAULT 0;")
            if "locked_until" not in user_cols:
                cursor.execute("ALTER TABLE institutional_users ADD COLUMN locked_until DATETIME;")
            if "student_identifier" not in user_cols:
                cursor.execute("ALTER TABLE institutional_users ADD COLUMN student_identifier TEXT;")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS password_resets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_or_username TEXT NOT NULL,
                    reset_token TEXT UNIQUE NOT NULL,
                    expires_at DATETIME NOT NULL,
                    used INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(reset_token);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_blacklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT UNIQUE NOT NULL,
                    revoked_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_blacklist_hash ON token_blacklist(token_hash);")

            # Ensure existing institutional users have their verified titles
            cursor.execute("UPDATE institutional_users SET role = 'Senior Academic Advisor' WHERE username = 'advisor';")
            cursor.execute("UPDATE institutional_users SET role = 'Department Chair' WHERE username = 'chair';")
            cursor.execute("UPDATE institutional_users SET role = 'Academic Registrar' WHERE username = 'registrar';")
            cursor.execute("UPDATE institutional_users SET role = 'Institutional Administrator' WHERE username = 'admin';")

            seed_rbac = [
                ("superadmin", "Chief Administrator", "SUPER_ADMIN", "Central IT", "admin123", ["*"], "superadmin@institution.edu", None),
                ("instadmin", "Dean Catherine Morland", "INSTITUTION_ADMIN", "College Leadership", "admin123", ["manage_institution", "read", "export", "analytics", "manage_users"], "dean@institution.edu", None),
                ("mathchair", "Prof. Marcus Sterling", "DEPARTMENT_ADMIN", "math", "1234", ["manage_department", "read", "predict", "simulate", "export", "train"], "m.sterling@institution.edu", None),
                ("teacher1", "Dr. Elena Vance", "TEACHER", "math", "1234", ["read", "predict", "simulate", "record_grades", "export"], "e.vance@institution.edu", None),
                ("student1", "Student MAT-0001", "STUDENT", "math", "1234", ["view_self"], "student1@institution.edu", "MAT-0001"),
            ]
            for uname, name, role, dept, pin, perms, email, stu_id in seed_rbac:
                salt, phash = hash_pin(pin)
                cursor.execute("""
                    INSERT OR REPLACE INTO institutional_users (
                        username, name, role, department, pin_salt, pin_hash, permissions_json, email, student_identifier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (uname, name, role, dept, salt, phash, json.dumps(perms), email, stu_id))

            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (6, '006_enhanced_security_and_rbac');")
            applied_names.append("006_enhanced_security_and_rbac")

        # Migration 7: Update schools to the 5 designated universities
        if 7 not in applied:
            cursor = conn.cursor()
            universities = [
                "Amity University Kolkata",
                "Adamas University",
                "Brainware University",
                "St. Xavier's University",
                "Techno India University"
            ]
            cursor.execute("SELECT id FROM students ORDER BY id ASC;")
            student_ids = [row[0] for row in cursor.fetchall()]
            for idx, s_id in enumerate(student_ids):
                uni = universities[idx % len(universities)]
                cursor.execute("UPDATE students SET school = ? WHERE id = ?;", (uni, s_id))

            cursor.execute("INSERT INTO schema_migrations (version, name) VALUES (7, '007_update_schools_to_universities');")
            applied_names.append("007_update_schools_to_universities")

        conn.commit()

    if applied_names:
        logger.info(f"Applied {len(applied_names)} database migrations: {', '.join(applied_names)}")
    return applied_names


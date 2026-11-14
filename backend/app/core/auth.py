"""Institutional authentication, role-based authorization, and token management."""
import os
import time
import hmac
import hashlib
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import Header, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("APEX_SECRET_KEY", "apex_academic_analytics_secure_auth_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Standard institutional role definitions
ROLES = {
    "SUPER_ADMIN": "Super Administrator with global platform authority",
    "INSTITUTION_ADMIN": "Institutional Administrator managing college departments and users",
    "DEPARTMENT_ADMIN": "Department Administrator managing department academics & staff",
    "TEACHER": "Faculty Member / Teacher recording grades and viewing student progress",
    "STUDENT": "Enrolled Student with restricted self-only view of academic records",
}

# Pre-configured fallback / seed institutional credentials for reference
INSTITUTIONAL_USERS: Dict[str, Dict[str, Any]] = {
    "superadmin": {
        "username": "superadmin",
        "name": "Chief Administrator",
        "role": "SUPER_ADMIN",
        "department": "Central Administration",
        "pin": "admin123",
        "permissions": ["*"],
    },
    "instadmin": {
        "username": "instadmin",
        "name": "Dean Catherine Morland",
        "role": "INSTITUTION_ADMIN",
        "department": "College Leadership",
        "pin": "admin123",
        "permissions": ["manage_institution", "read", "export", "analytics", "manage_users"],
    },
    "mathchair": {
        "username": "mathchair",
        "name": "Prof. Marcus Sterling",
        "role": "DEPARTMENT_ADMIN",
        "department": "math",
        "pin": "1234",
        "permissions": ["manage_department", "read", "predict", "simulate", "export", "train"],
    },
    "teacher1": {
        "username": "teacher1",
        "name": "Dr. Elena Vance",
        "role": "TEACHER",
        "department": "math",
        "pin": "1234",
        "permissions": ["read", "predict", "simulate", "record_grades", "export"],
    },
    "student1": {
        "username": "student1",
        "name": "Student MAT-0001",
        "role": "STUDENT",
        "department": "math",
        "student_identifier": "MAT-0001",
        "pin": "1234",
        "permissions": ["view_self"],
    },
    "advisor": {
        "username": "advisor",
        "name": "Dr. Elena Vance",
        "role": "Senior Academic Advisor",
        "department": "STEM & Analytics",
        "pin": "1234",
        "permissions": ["read", "predict", "simulate", "export"],
    },
    "chair": {
        "username": "chair",
        "name": "Prof. Marcus Sterling",
        "role": "Department Chair",
        "department": "Mathematics & Quantitative Sciences",
        "pin": "1234",
        "permissions": ["read", "predict", "simulate", "export", "train"],
    },
    "registrar": {
        "username": "registrar",
        "name": "Alicia Gomez",
        "role": "Academic Registrar",
        "department": "Academic Records & Compliance",
        "pin": "1234",
        "permissions": ["read", "export", "audit", "clear_history"],
    },
    "admin": {
        "username": "admin",
        "name": "System Administrator",
        "role": "Institutional Administrator",
        "department": "Central IT & Analytics Infrastructure",
        "pin": "admin123",
        "permissions": ["*"],
    },
}

security = HTTPBearer(auto_error=False)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64decode(data: str) -> bytes:
    padding = "=" * (4 - (len(data) % 4)) if (len(data) % 4) else ""
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate a cryptographically signed HMAC-SHA256 bearer token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "token_type": "access",
    })

    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = _b64encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64encode(json.dumps(to_encode, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Generate a cryptographically signed 7-day refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "token_type": "refresh",
    })

    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = _b64encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64encode(json.dumps(to_encode, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify HMAC-SHA256 signature, token revocation, and expiration timestamp."""
    try:
        from backend.app.database.repository import is_token_blacklisted
        if is_token_blacklisted(token):
            return None

        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _b64decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
        if "exp" in payload and int(payload["exp"]) < int(time.time()):
            return None

        return payload
    except Exception:
        return None


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_advisor_role: Optional[str] = Header(None, alias="X-Advisor-Role"),
) -> Optional[Dict[str, Any]]:
    """Resolve authenticated institutional user from Bearer token or fallback header."""
    if auth and auth.credentials:
        payload = decode_access_token(auth.credentials)
        if payload and "sub" in payload:
            from backend.app.database.repository import get_institutional_user
            db_user = get_institutional_user(payload["sub"])
            if db_user:
                return {**db_user, "token_payload": payload, "token": auth.credentials}
            user = INSTITUTIONAL_USERS.get(payload["sub"])
            if user:
                return {**user, "token_payload": payload, "token": auth.credentials}
            return {
                "username": payload.get("sub", "authenticated_user"),
                "name": payload.get("name", "Institutional User"),
                "role": payload.get("role", "TEACHER"),
                "department": payload.get("department", "math"),
                "student_identifier": payload.get("student_identifier"),
                "permissions": payload.get("permissions", ["read"]),
                "token_payload": payload,
                "token": auth.credentials,
            }

    if x_advisor_role:
        for u in INSTITUTIONAL_USERS.values():
            if u["role"].lower() == x_advisor_role.lower():
                return u

    return None


def require_auth(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency enforcing that request must include valid authorization."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token. Provide Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_role(allowed_roles: List[str]):
    """Factory dependency enforcing Role-Based Access Control on protected endpoints."""
    def role_checker(user: Any = Depends(require_auth)) -> Dict[str, Any]:
        if not isinstance(user, dict):
            return user
        user_role = (user.get("role") or "").upper()
        if user_role == "SUPER_ADMIN" or "INSTITUTIONAL ADMINISTRATOR" in user_role or "*" in user.get("permissions", []):
            return user

        allowed_upper = [r.upper() for r in allowed_roles]
        if user_role in allowed_upper:
            return user

        # Support synonyms for seamless institutional titles
        role_synonyms = {
            "SENIOR ACADEMIC ADVISOR": "TEACHER",
            "DEPARTMENT CHAIR": "DEPARTMENT_ADMIN",
            "ACADEMIC REGISTRAR": "INSTITUTION_ADMIN",
            "INSTITUTIONAL ADMINISTRATOR": "SUPER_ADMIN",
        }
        mapped = role_synonyms.get(user_role, user_role)
        if mapped in allowed_upper:
            return user

        for ar in allowed_upper:
            if ar in user_role or user_role in ar:
                return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. Required roles: {allowed_roles}. Current role: {user_role}",
        )
    return role_checker


def enforce_student_self_access(current_user: Any, requested_student_id: str) -> None:
    """Ensure a user with STUDENT role can only view or interact with their own record."""
    if not current_user or not isinstance(current_user, dict):
        return
    role = (current_user.get("role") or "").upper()
    if role == "STUDENT":
        assigned_id = current_user.get("student_identifier") or ""
        if assigned_id.upper() != requested_student_id.strip().upper():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Students are only authorized to view their own academic record.",
            )


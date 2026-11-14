"""Institutional Authentication & Authorization API Endpoints."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, EmailStr
from fastapi import APIRouter, HTTPException, Depends, status
from backend.app.core.auth import (
    ROLES,
    INSTITUTIONAL_USERS,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    get_current_user,
    require_auth,
)
from backend.app.database.repository import (
    verify_user_credentials_secure,
    register_user_db,
    get_institutional_user,
    create_password_reset_token,
    confirm_password_reset,
    blacklist_token,
    get_student_by_id_db,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


class LoginRequest(BaseModel):
    username: str = Field(..., description="Institutional username or email (e.g., 'superadmin', 'instadmin', 'mathchair', 'teacher1', 'student1')")
    password: Optional[str] = Field(None, description="Access password or PIN")
    pin: Optional[str] = Field(None, description="PIN alias for backwards compatibility")


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 28800
    username: str
    name: str
    role: str
    department: str
    student_identifier: Optional[str] = None
    permissions: List[str]


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, description="Unique username")
    name: str = Field(..., min_length=2, max_length=100, description="Full real name")
    password: Optional[str] = Field(None, description="Secure password or PIN")
    pin: Optional[str] = Field(None, description="PIN alias")
    role: str = Field(default="TEACHER", description="Role: SUPER_ADMIN, INSTITUTION_ADMIN, DEPARTMENT_ADMIN, TEACHER, or STUDENT")
    department: str = Field(default="math", description="Department code: 'math' or 'portuguese'")
    email: Optional[str] = Field(None, description="Official institutional email")
    student_identifier: Optional[str] = Field(None, description="Required for STUDENT role, e.g. MAT-0001")


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email_or_username: str


class PasswordResetConfirm(BaseModel):
    token: Optional[str] = None
    reset_token: Optional[str] = None
    new_password: Optional[str] = None
    new_pin: Optional[str] = None


class TokenVerifyRequest(BaseModel):
    token: str


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """Authenticate user credentials and issue signed access and refresh tokens."""
    username = request.username.strip().lower()
    pwd = (request.password or request.pin or "").strip()
    user, err_msg = verify_user_credentials_secure(username, pwd)

    if err_msg and "locked" in err_msg.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=err_msg,
        )

    if not user:
        # Fallback to demonstration memory dictionary for backwards compatibility
        mem_user = INSTITUTIONAL_USERS.get(username)
        if mem_user and mem_user["pin"] == pwd:
            user = mem_user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=err_msg or "Invalid username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    token_data = {
        "sub": user["username"],
        "name": user["name"],
        "role": user["role"],
        "department": user["department"],
        "student_identifier": user.get("student_identifier"),
        "permissions": user["permissions"],
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=28800,
        username=user["username"],
        name=user["name"],
        role=user["role"],
        department=user["department"],
        student_identifier=user.get("student_identifier"),
        permissions=user["permissions"],
    )


@router.post("/register")
def register(request: RegisterRequest):
    """Register a new institutional user with input validation and role assignment."""
    clean_username = request.username.strip().lower()
    existing = get_institutional_user(clean_username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{clean_username}' is already registered.",
        )

    # Validate role
    role_upper = request.role.strip().upper()
    valid_roles = list(ROLES.keys())
    if role_upper not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{request.role}'. Must be one of: {valid_roles}",
        )

    # If student, validate student_identifier
    stu_id = None
    if role_upper == "STUDENT":
        if not request.student_identifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student registration requires a valid student_identifier (e.g., MAT-0001).",
            )
        stu_id = request.student_identifier.strip().upper()
        student_record = get_student_by_id_db(stu_id)
        if not student_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No student record found with identifier '{stu_id}'.",
            )

    new_user = register_user_db(
        username=clean_username,
        name=request.name.strip(),
        pin=request.password.strip(),
        role=role_upper,
        department=request.department.strip().lower(),
        email=request.email.strip().lower() if request.email else None,
        student_identifier=stu_id,
    )

    return {
        "status": "success",
        "message": f"User '{clean_username}' successfully registered with role '{role_upper}'.",
        "username": new_user["username"],
        "role": new_user["role"],
    }


@router.post("/refresh")
def refresh_token_endpoint(req: RefreshRequest):
    """Issue a fresh access token and rotated refresh token from a valid refresh token."""
    payload = decode_access_token(req.refresh_token)
    if not payload or payload.get("token_type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, revoked, or expired refresh token.",
        )
    username = payload.get("sub")
    user = get_institutional_user(username)
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or not found.",
        )
    token_data = {
        "sub": user["username"],
        "name": user["name"],
        "role": user["role"],
        "department": user["department"],
        "student_identifier": user.get("student_identifier"),
        "permissions": user["permissions"],
    }
    new_access = create_access_token(token_data)
    new_refresh = create_refresh_token(token_data)
    # Revoke used refresh token for strict rotation
    blacklist_token(req.refresh_token)
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "expires_in": 28800,
    }


@router.post("/logout")
def logout(current_user: Dict[str, Any] = Depends(require_auth)):
    """Revoke active session token."""
    token = current_user.get("token")
    if token:
        blacklist_token(token)
    return {"status": "success", "message": "Successfully logged out. Session token revoked."}


@router.post("/password-reset-request")
def request_password_reset(req: PasswordResetRequest):
    """Generate a password reset token for account recovery."""
    token = create_password_reset_token(req.email_or_username.strip())
    if not token:
        # Avoid user enumeration in public responses
        return {"status": "info", "message": "If an account exists, a reset token has been issued."}
    return {
        "status": "success",
        "message": "Password reset token generated.",
        "reset_token": token,
    }


@router.post("/password-reset")
def confirm_password_reset_endpoint(req: PasswordResetConfirm):
    """Reset user password with valid reset token."""
    tok = (req.token or req.reset_token or "").strip()
    new_pwd = (req.new_password or req.new_pin or "").strip()
    if not tok or not new_pwd:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token and new password or PIN are required.",
        )
    success = confirm_password_reset(tok, new_pwd)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already used password reset token.",
        )
    return {"status": "success", "message": "Password reset successful. You may now log in."}


@router.get("/me")
def get_authenticated_profile(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Retrieve identity, role permissions, and session status for active user."""
    if not current_user:
        return {
            "authenticated": False,
            "role": "GUEST",
            "permissions": ["read"],
            "message": "Public observation mode. Sign in to access institutional management features.",
        }
    return {
        "authenticated": True,
        "username": current_user.get("username"),
        "name": current_user.get("name"),
        "role": current_user.get("role"),
        "department": current_user.get("department"),
        "student_identifier": current_user.get("student_identifier"),
        "permissions": current_user.get("permissions", []),
    }


@router.post("/verify")
def verify_token(req: TokenVerifyRequest):
    """Verify validity and signature of a session token."""
    payload = decode_access_token(req.token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid, malformed, or expired.",
        )
    return {
        "valid": True,
        "sub": payload.get("sub"),
        "role": payload.get("role"),
        "expires_at": payload.get("exp"),
    }


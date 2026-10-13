"""app/backend/routers/auth.py
Authentication endpoints: register, login, logout, refresh, me, change-password.
Follows strict security: Argon2id, token family rotation, rate limiting, and cookie flags.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
import secrets
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.models.entities import User, RefreshToken, AuditLog
from app.backend.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    ChangePasswordRequest,
    TokenResponse,
    UserResponse,
)
from app.backend.schemas.common import MessageResponse
from app.backend.core.security import (
    get_password_hash,
    verify_password,
    validate_password_strength,
    create_access_token,
    generate_token_pair,
    hash_token,
    generate_csrf_token,
)
from app.backend.core.config import settings
from app.backend.routers.deps import get_current_user, get_current_user_optional, verify_csrf
from app.backend.core.rate_limit import limiter, get_login_rate_limit_key


router = APIRouter(prefix="/auth", tags=["auth"])


def set_auth_cookies(response: Response, access_token: str, refresh_raw: str, csrf_token: str):
    """Set secure httpOnly cookies for access and refresh tokens, and readable cookie for CSRF."""
    cookie_secure = settings.is_prod
    # Access token cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_MINUTES * 60,
        path="/"
    )
    # Refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_raw,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_DAYS * 24 * 3600,
        path="/"
    )
    # CSRF cookie (accessible to client for double-submit header)
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        httponly=False,
        secure=cookie_secure,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_DAYS * 24 * 3600,
        path="/"
    )
    response.headers["X-CSRF-Token"] = csrf_token


def clear_auth_cookies(response: Response):
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/")
    response.delete_cookie(key="csrf_token", path="/")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    email = payload.email.lower().strip()
    # 1. Password policy
    pwd_err = validate_password_strength(payload.password, email)
    if pwd_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=pwd_err)

    # 2. Check existing user
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )

    # 3. Create user (always role 'user')
    hashed = get_password_hash(payload.password)
    user = User(
        email=email,
        hashed_password=hashed,
        role="user",
        is_active=True
    )
    db.add(user)
    db.flush()

    # 4. Generate tokens
    access_token = create_access_token({"sub": str(user.id), "role": user.role})
    refresh_raw = generate_token_pair()
    csrf_token = generate_csrf_token()
    family_id = secrets.token_hex(16)

    rt_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_raw),
        family_id=family_id,
        is_revoked=False,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_DAYS)
    )
    db.add(rt_record)

    # Audit log
    audit = AuditLog(
        actor_id=user.id,
        actor_email=user.email,
        action="USER_REGISTER",
        target=str(user.id),
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    set_auth_cookies(response, access_token, refresh_raw, csrf_token)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        csrf_token=csrf_token,
        user=UserResponse.model_validate(user)
    )


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@router.post("/login", response_model=TokenResponse)
@limiter.limit("60/minute", key_func=get_login_rate_limit_key)
def login(
    payload: LoginRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    # Check lock
    now = datetime.now(timezone.utc)
    if user and user.locked_until and ensure_utc(user.locked_until) > now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is temporarily locked due to multiple failed login attempts. Please try again later."
        )

    if not user or not verify_password(payload.password, user.hashed_password):
        if user:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 10:
                user.locked_until = now + timedelta(minutes=15)
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been disabled."
        )

    # Successful login: reset failed attempts
    user.failed_login_attempts = 0
    user.locked_until = None

    access_token = create_access_token({"sub": str(user.id), "role": user.role})
    refresh_raw = generate_token_pair()
    csrf_token = generate_csrf_token()
    family_id = secrets.token_hex(16)

    rt_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_raw),
        family_id=family_id,
        is_revoked=False,
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_DAYS)
    )
    db.add(rt_record)

    audit = AuditLog(
        actor_id=user.id,
        actor_email=user.email,
        action="USER_LOGIN",
        target=str(user.id),
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    set_auth_cookies(response, access_token, refresh_raw, csrf_token)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        csrf_token=csrf_token,
        user=UserResponse.model_validate(user)
    )


@router.post("/logout", response_model=MessageResponse)
def logout(
    response: Response,
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    refresh_raw = request.cookies.get("refresh_token")
    if refresh_raw:
        rt_hash = hash_token(refresh_raw)
        db.query(RefreshToken).filter(RefreshToken.token_hash == rt_hash).update({"is_revoked": True})
        db.commit()

    clear_auth_cookies(response)
    return MessageResponse(message="Successfully logged out.")


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    refresh_raw = request.cookies.get("refresh_token")
    if not refresh_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing."
        )

    rt_hash = hash_token(refresh_raw)
    rt = db.query(RefreshToken).filter(RefreshToken.token_hash == rt_hash).first()
    now = datetime.now(timezone.utc)

    if not rt or rt.is_revoked or (ensure_utc(rt.expires_at) < now):
        # Detected reuse or expired token: revoke whole family if found
        if rt:
            db.query(RefreshToken).filter(RefreshToken.family_id == rt.family_id).update({"is_revoked": True})
            db.commit()
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token."
        )

    user = db.query(User).filter(User.id == rt.user_id).first()
    if not user or not user.is_active:
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled."
        )

    # Rotate refresh token: revoke current, issue new with same family_id
    rt.is_revoked = True
    new_refresh_raw = generate_token_pair()
    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(new_refresh_raw),
        family_id=rt.family_id,
        is_revoked=False,
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_DAYS)
    )
    db.add(new_rt)

    access_token = create_access_token({"sub": str(user.id), "role": user.role})
    csrf_token = generate_csrf_token()
    db.commit()

    set_auth_cookies(response, access_token, new_refresh_raw, csrf_token)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        csrf_token=csrf_token,
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse.model_validate(user)


@router.post("/change-password", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def change_password(
    payload: ChangePasswordRequest,
    response: Response,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not verify_password(payload.old_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password does not match."
        )

    pwd_err = validate_password_strength(payload.new_password, user.email)
    if pwd_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=pwd_err)

    user.hashed_password = get_password_hash(payload.new_password)
    # Revoke all existing refresh tokens
    db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update({"is_revoked": True})
    db.commit()

    clear_auth_cookies(response)
    return MessageResponse(message="Password successfully changed. Please log in again.")

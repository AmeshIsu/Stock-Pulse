"""app/backend/routers/deps.py
FastAPI dependencies for authentication, RBAC, database session, and CSRF protection.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.models.entities import User
from app.backend.core.security import decode_access_token
from app.backend.core.exceptions import APIError


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Extract authenticated user if token present, or None if anonymous."""
    token = None
    # 1. Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    # 2. Check httpOnly cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user or not user.is_active:
            return None
        return user
    except Exception:
        return None


def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Require authenticated active user (returns 401)."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required."
        )
    return user


def get_current_admin(
    user: User = Depends(get_current_user)
) -> User:
    """Require user with role 'admin' (returns 403 for non-admins)."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required."
        )
    return user


def verify_csrf(request: Request):
    """Verify CSRF double-submit token for state-changing requests."""
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return
    csrf_cookie = request.cookies.get("csrf_token")
    csrf_header = request.headers.get("X-CSRF-Token")
    if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing CSRF token."
        )

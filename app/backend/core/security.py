"""app/backend/core/security.py
Security primitives: Argon2id password hashing, JWT access tokens,
refresh token family rotation hashing, and CSRF token double-submit protection.
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
import jwt

from app.backend.core.config import settings

ph = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)

COMMON_PASSWORDS = {
    "password", "password123", "1234567890", "12345678", "qwertyuiop",
    "admin12345", "welcome123", "stockpulse", "stockpulse123", "letmein123"
}


def get_password_hash(password: str) -> str:
    """Hash password using Argon2id."""
    return ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against Argon2id hash."""
    try:
        return ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError):
        return False


def validate_password_strength(password: str, email: str) -> Optional[str]:
    """Validate password strength rules:
    - Minimum 10 characters
    - Not equal to email
    - Not in common password list
    """
    if len(password) < 10:
        return "Password must be at least 10 characters long."
    if password.lower() == email.lower():
        return "Password cannot be equal to the email address."
    if password.lower() in COMMON_PASSWORDS:
        return "Password is too common or easily guessable."
    return None


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create short-lived JWT access token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_MINUTES)
    to_encode.update({"iat": now, "exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError as e:
        raise ValueError(f"Invalid or expired token: {str(e)}")


def generate_token_pair() -> str:
    """Generate cryptographically secure random token string."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Hash token using SHA-256 for persistent database storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    """Generate CSRF double-submit token."""
    return secrets.token_hex(24)

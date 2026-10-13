"""app/backend/core/rate_limit.py
SlowAPI rate limiting setup and helpers.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


def get_login_rate_limit_key(request: Request) -> str:
    """Generate composite key using client IP."""
    ip = get_remote_address(request)
    return f"login:{ip}"

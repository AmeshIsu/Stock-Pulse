"""app/backend/core/exceptions.py
Standardized exception definitions and error JSON shapes.
Format:
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "request_id": "req_xyz"
    }
}
"""
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import uuid


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


def get_request_id(request: Request) -> str:
    """Retrieve or generate request ID."""
    req_id = getattr(request.state, "request_id", None)
    if not req_id:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
    return req_id


def error_response(code: str, message: str, status_code: int, request_id: str) -> JSONResponse:
    """Return standard JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id
            }
        },
        headers={"X-Request-ID": request_id}
    )

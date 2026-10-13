"""app/backend/schemas/common.py
Common response schemas and pagination helpers.
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class MessageResponse(BaseModel):
    message: str
    success: bool = True

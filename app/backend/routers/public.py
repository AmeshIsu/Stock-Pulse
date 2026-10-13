"""app/backend/routers/public.py
Public informational endpoints accessible to unauthenticated visitors.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from app.backend.services.serving import serving_repo

router = APIRouter(tags=["public"])


class PublicInfoResponse(BaseModel):
    app_name: str
    data_as_of: str
    disclaimer_text: str


@router.get("/public/info", response_model=PublicInfoResponse)
def get_public_info():
    """Retrieve public application information and compliance disclaimer."""
    current_date = serving_repo.get_current_as_of()
    return PublicInfoResponse(
        app_name="Stock Pulse",
        data_as_of=current_date,
        disclaimer_text=(
            "Stock Pulse provides data-driven rankings for information only. "
            "It is not financial advice. Signals shown have limited historical reliability."
        )
    )

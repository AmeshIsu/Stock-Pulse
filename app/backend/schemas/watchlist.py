"""app/backend/schemas/watchlist.py
Watchlist request and response schemas.
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class WatchlistItemCreate(BaseModel):
    series_id: str
    note: Optional[str] = None


class WatchlistItemResponse(BaseModel):
    id: int
    user_id: int
    series_id: str
    note: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

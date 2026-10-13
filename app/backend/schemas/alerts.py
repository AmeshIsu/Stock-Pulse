"""app/backend/schemas/alerts.py
Alert and notification request/response schemas.
"""
from typing import Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class AlertCreate(BaseModel):
    series_id: str
    type: str  # price_above, price_below, daily_change_above, daily_change_below, enters_high_tier, becomes_consistent_leader, risk_changes
    params: Optional[Dict[str, Any]] = None
    horizon: Optional[str] = None  # for enters_high_tier
    cooldown: int = 1


class AlertUpdate(BaseModel):
    active: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    cooldown: Optional[int] = None


class AlertResponse(BaseModel):
    id: int
    user_id: int
    series_id: str
    type: str
    params: Optional[Dict[str, Any]] = None
    horizon: Optional[str] = None
    active: bool
    cooldown: int
    last_triggered_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NotificationResponse(BaseModel):
    id: int
    user_id: int
    alert_id: Optional[int] = None
    message: str
    data_version: str
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True

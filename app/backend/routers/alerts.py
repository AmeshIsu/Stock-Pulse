"""app/backend/routers/alerts.py
Alerts CRUD endpoints with strict object-level authorization (returns 404 for other users' alerts).
Types: price_above, price_below, daily_change_above, daily_change_below,
enters_high_tier, becomes_consistent_leader, risk_changes.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.models.entities import User, Alert
from app.backend.schemas.alerts import AlertCreate, AlertUpdate, AlertResponse
from app.backend.schemas.common import MessageResponse
from app.backend.routers.deps import get_current_user, verify_csrf

router = APIRouter(prefix="/alerts", tags=["alerts"])

VALID_ALERT_TYPES = {
    "price_above", "price_below", "daily_change_above", "daily_change_below",
    "enters_high_tier", "becomes_consistent_leader", "risk_changes"
}


@router.get("", response_model=List[AlertResponse])
def get_alerts(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Alert).filter(Alert.user_id == user.id).all()


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_csrf)])
def create_alert(
    payload: AlertCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if payload.type not in VALID_ALERT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid alert type '{payload.type}'. Valid types: {sorted(list(VALID_ALERT_TYPES))}"
        )

    alert = Alert(
        user_id=user.id,
        series_id=payload.series_id.upper().strip(),
        type=payload.type,
        params=payload.params or {},
        horizon=payload.horizon,
        active=True,
        cooldown=payload.cooldown
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


@router.patch("/{alert_id}", response_model=AlertResponse, dependencies=[Depends(verify_csrf)])
def update_alert(
    alert_id: int,
    payload: AlertUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    # Object-level authorization: 404 for other user's alert
    if not alert or alert.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")

    if payload.active is not None:
        alert.active = payload.active
    if payload.params is not None:
        alert.params = payload.params
    if payload.cooldown is not None:
        alert.cooldown = payload.cooldown

    db.commit()
    db.refresh(alert)
    return alert


@router.delete("/{alert_id}", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def delete_alert(
    alert_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert or alert.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")

    db.delete(alert)
    db.commit()
    return MessageResponse(message="Alert deleted.")

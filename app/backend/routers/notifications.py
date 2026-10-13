"""app/backend/routers/notifications.py
Notifications endpoints with strict object-level authorization (returns 404 for other users' notifications).
"""
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.models.entities import User, Notification
from app.backend.schemas.alerts import NotificationResponse
from app.backend.schemas.common import MessageResponse
from app.backend.routers.deps import get_current_user, verify_csrf

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=List[NotificationResponse])
def get_notifications(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Notification).filter(
        Notification.user_id == user.id
    ).order_by(Notification.created_at.desc()).all()


@router.post("/{notification_id}/read", response_model=NotificationResponse, dependencies=[Depends(verify_csrf)])
def mark_notification_read(
    notification_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notif = db.query(Notification).filter(Notification.id == notification_id).first()
    # Object-level authorization: 404 if not user's notification
    if not notif or notif.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found.")

    notif.read_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(notif)
    return notif


@router.post("/read-all", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def mark_all_notifications_read(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.now(timezone.utc)
    db.query(Notification).filter(
        Notification.user_id == user.id,
        Notification.read_at.is_(None)
    ).update({"read_at": now})
    db.commit()
    return MessageResponse(message="All notifications marked as read.")

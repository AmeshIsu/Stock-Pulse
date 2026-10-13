"""app/backend/routers/watchlist.py
User watchlist endpoints with strict object-level authorization (returns 404 for unauthorized items).
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from app.backend.db.session import get_db
from app.backend.models.entities import User, WatchlistItem, ModelSetting
from app.backend.schemas.watchlist import WatchlistItemCreate, WatchlistItemResponse
from app.backend.schemas.common import MessageResponse
from app.backend.routers.deps import get_current_user, verify_csrf
from app.backend.services.serving import serving_repo

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


class EnrichedWatchlistItem(BaseModel):
    id: int
    series_id: str
    company_name: str
    note: Optional[str] = None
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None
    overall_upside_score: Optional[float] = None
    tier_short: str
    tier_mid: str
    tier_long: str
    risk_category: str
    liquidity_warning: bool
    consistent_leader: bool
    created_at: datetime


@router.get("", response_model=List[EnrichedWatchlistItem])
def get_watchlist(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    if not items:
        return []

    # Get latest snapshot
    db_settings = {r.horizon: (r.mode, r.model_id) for r in db.query(ModelSetting).all()}
    active_models = serving_repo.resolve_active_models(db_settings)
    snapshot = serving_repo.compute_serving_snapshot(active_models)
    stock_dict = {s["series_id"]: s for s in snapshot["stocks"]}
    meta_dict = {s["code"]: s for s in serving_repo.get_stocks_metadata()}

    results = []
    for it in items:
        s_data = stock_dict.get(it.series_id, {})
        m_data = meta_dict.get(it.series_id, {})
        results.append(EnrichedWatchlistItem(
            id=it.id,
            series_id=it.series_id,
            company_name=m_data.get("company_name", it.series_id.split(".")[0]),
            note=it.note,
            last_price=m_data.get("last_price"),
            daily_change_pct=m_data.get("daily_change_pct"),
            overall_upside_score=s_data.get("overall_upside_score"),
            tier_short=s_data.get("tier_short", "N/A"),
            tier_mid=s_data.get("tier_mid", "N/A"),
            tier_long=s_data.get("tier_long", "N/A"),
            risk_category=s_data.get("risk_category", "N/A"),
            liquidity_warning=bool(s_data.get("liquidity_warning", False)),
            consistent_leader=bool(s_data.get("consistent_leader", False)),
            created_at=it.created_at
        ))
    return results


@router.post("", response_model=WatchlistItemResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_csrf)])
def add_watchlist_item(
    payload: WatchlistItemCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sid = payload.series_id.upper().strip()
    existing = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id,
        WatchlistItem.series_id == sid
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stock {sid} is already in your watchlist."
        )

    item = WatchlistItem(
        user_id=user.id,
        series_id=sid,
        note=payload.note
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def delete_watchlist_item(
    item_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
    # Object-level authorization: return 404 for another user's item
    if not item or item.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist item not found."
        )

    db.delete(item)
    db.commit()
    return MessageResponse(message="Watchlist item deleted.")

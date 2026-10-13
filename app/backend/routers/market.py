"""app/backend/routers/market.py
Market summary endpoint (public/user accessible, strictly user-facing).
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.services.serving import serving_repo
from app.backend.schemas.market import MarketSummaryResponse, TierCounts
from app.backend.models.entities import ModelSetting
from app.backend.routers.deps import get_current_user

router = APIRouter(prefix="/market", tags=["market"], dependencies=[Depends(get_current_user)])


@router.get("/summary", response_model=MarketSummaryResponse)
def get_market_summary(db: Session = Depends(get_db)):
    # Read model settings overrides if any
    db_settings = {}
    rows = db.query(ModelSetting).all()
    for r in rows:
        db_settings[r.horizon] = (r.mode, r.model_id)

    active_models = serving_repo.resolve_active_models(db_settings)
    snapshot = serving_repo.compute_serving_snapshot(active_models)

    stocks = snapshot["stocks"]
    counts = {
        "short": {"High": 0, "Neutral": 0, "Low": 0},
        "mid": {"High": 0, "Neutral": 0, "Low": 0},
        "long": {"High": 0, "Neutral": 0, "Low": 0}
    }
    cl_count = 0
    for s in stocks:
        if s["rankable"]:
            for h in ["short", "mid", "long"]:
                t = s.get(f"tier_{h}")
                if t in counts[h]:
                    counts[h][t] += 1
            if s.get("consistent_leader"):
                cl_count += 1

    return MarketSummaryResponse(
        as_of=snapshot["as_of"],
        data_version=snapshot["data_version"],
        tier_counts_short=TierCounts(**counts["short"]),
        tier_counts_mid=TierCounts(**counts["mid"]),
        tier_counts_long=TierCounts(**counts["long"]),
        consistent_leaders_count=cl_count,
        market_volatility_note=snapshot.get("market_volatility_note")
    )

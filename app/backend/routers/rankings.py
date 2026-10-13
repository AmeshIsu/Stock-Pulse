"""app/backend/routers/rankings.py
Rankings, Consistent Leaders, 21-day Movers, and CSV Export endpoints.
Strictly user-facing: zero model names, metrics, or IC leaks.
"""
import io
import csv
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.services.serving import serving_repo
from app.backend.schemas.rankings import (
    RankingsResponse,
    RankingStockItem,
    LeadersResponse,
    ConsistentLeaderItem,
    MoversResponse,
    MoverItem
)
from app.backend.models.entities import ModelSetting
from app.backend.routers.deps import get_current_user

router = APIRouter(tags=["rankings"], dependencies=[Depends(get_current_user)])


def parse_custom_weights(weights_str: Optional[str]) -> Optional[dict]:
    """Parse comma-separated weights e.g. 'short:0.3,mid:0.4,long:0.3' or '0.3,0.4,0.3'."""
    if not weights_str:
        return None
    try:
        parts = weights_str.split(",")
        if len(parts) != 3:
            return None
        if ":" in parts[0]:
            w_dict = {}
            for p in parts:
                k, v = p.split(":")
                w_dict[k.strip().lower()] = float(v)
            if set(w_dict.keys()) == {"short", "mid", "long"}:
                return w_dict
        else:
            return {
                "short": float(parts[0]),
                "mid": float(parts[1]),
                "long": float(parts[2])
            }
    except Exception:
        return None
    return None


@router.get("/rankings", response_model=RankingsResponse)
def get_rankings(
    horizon: str = Query("overall", pattern="^(short|mid|long|overall)$"),
    weights: Optional[str] = None,
    risk: Optional[str] = None,
    exclude_liquidity_warning: bool = False,
    limit: Optional[int] = None,
    db: Session = Depends(get_db)
):
    db_settings = {}
    rows = db.query(ModelSetting).all()
    for r in rows:
        db_settings[r.horizon] = (r.mode, r.model_id)

    active_models = serving_repo.resolve_active_models(db_settings)
    custom_w = parse_custom_weights(weights)
    snapshot = serving_repo.compute_serving_snapshot(active_models, custom_weights=custom_w)

    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}
    stocks = snapshot["stocks"]

    ranked_items = []
    for s in stocks:
        if not s["rankable"]:
            continue
        if risk and s["risk_category"] != risk:
            continue
        if exclude_liquidity_warning and s["liquidity_warning"]:
            continue

        sid = s["series_id"]
        meta = stocks_meta.get(sid, {})
        comp_name = meta.get("company_name", sid.split(".")[0])
        last_p = meta.get("last_price")
        d_chg = meta.get("daily_change_pct")

        if horizon == "overall":
            score_val = s["overall_upside_score"]
            tier_val = s["tier_mid"]  # indicative tier
            conf_val = s["confidence_mid"]
            chg_val = s["rank_change_21d_mid"]
        else:
            score_val = s[f"score_{horizon}"]
            tier_val = s[f"tier_{horizon}"]
            conf_val = s[f"confidence_{horizon}"]
            chg_val = s[f"rank_change_21d_{horizon}"]

        if score_val is not None:
            ranked_items.append({
                "code": sid,
                "company_name": comp_name,
                "score": score_val,
                "tier": tier_val,
                "confidence": conf_val,
                "rank_change_21d": chg_val,
                "risk_category": s["risk_category"],
                "liquidity_warning": s["liquidity_warning"],
                "consistent_leader": s["consistent_leader"],
                "last_price": last_p,
                "daily_change_pct": d_chg
            })

    # Sort descending by score
    ranked_items.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(ranked_items, start=1):
        item["rank"] = idx

    if limit:
        ranked_items = ranked_items[:limit]

    res_stocks = [RankingStockItem(**item) for item in ranked_items]
    return RankingsResponse(
        horizon=horizon,
        as_of=snapshot["as_of"],
        weights_used=snapshot["horizon_weights"] if horizon == "overall" else None,
        stocks=res_stocks
    )


@router.get("/leaders", response_model=LeadersResponse)
def get_consistent_leaders(db: Session = Depends(get_db)):
    db_settings = {}
    rows = db.query(ModelSetting).all()
    for r in rows:
        db_settings[r.horizon] = (r.mode, r.model_id)

    active_models = serving_repo.resolve_active_models(db_settings)
    snapshot = serving_repo.compute_serving_snapshot(active_models)
    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}

    leaders = []
    for s in snapshot["stocks"]:
        if s["rankable"] and s["consistent_leader"]:
            sid = s["series_id"]
            meta = stocks_meta.get(sid, {})
            leaders.append(ConsistentLeaderItem(
                code=sid,
                company_name=meta.get("company_name", sid.split(".")[0]),
                overall_upside_score=s["overall_upside_score"],
                tier_short=s["tier_short"],
                tier_mid=s["tier_mid"],
                tier_long=s["tier_long"],
                last_price=meta.get("last_price"),
                daily_change_pct=meta.get("daily_change_pct")
            ))

    leaders.sort(key=lambda x: x.overall_upside_score, reverse=True)
    return LeadersResponse(
        as_of=snapshot["as_of"],
        count=len(leaders),
        leaders=leaders
    )


@router.get("/movers", response_model=MoversResponse)
def get_movers(db: Session = Depends(get_db)):
    db_settings = {}
    rows = db.query(ModelSetting).all()
    for r in rows:
        db_settings[r.horizon] = (r.mode, r.model_id)

    active_models = serving_repo.resolve_active_models(db_settings)
    snapshot = serving_repo.compute_serving_snapshot(active_models)
    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}

    # Check movers across mid horizon
    items = []
    for s in snapshot["stocks"]:
        if s["rankable"] and s["rank_change_21d_mid"] is not None:
            sid = s["series_id"]
            meta = stocks_meta.get(sid, {})
            chg = s["rank_change_21d_mid"]
            items.append(MoverItem(
                code=sid,
                company_name=meta.get("company_name", sid.split(".")[0]),
                horizon="mid",
                score=s["score_mid"],
                rank_change_21d=chg,
                direction="up" if chg >= 0 else "down"
            ))

    items.sort(key=lambda x: x.rank_change_21d, reverse=True)
    gainers = [x for x in items if x.rank_change_21d > 0][:5]
    decliners = [x for x in reversed(items) if x.rank_change_21d < 0][:5]

    return MoversResponse(
        as_of=snapshot["as_of"],
        biggest_gainers=gainers,
        biggest_decliners=decliners
    )


@router.get("/rankings/export.csv")
def export_rankings_csv(
    horizon: str = Query("overall", pattern="^(short|mid|long|overall)$"),
    weights: Optional[str] = None,
    risk: Optional[str] = None,
    exclude_liquidity_warning: bool = False,
    db: Session = Depends(get_db)
):
    """Export current rankings view to CSV with strictly user-visible columns."""
    res = get_rankings(
        horizon=horizon,
        weights=weights,
        risk=risk,
        exclude_liquidity_warning=exclude_liquidity_warning,
        limit=None,
        db=db
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank", "Code", "Company Name", "Upside Score", "Tier",
        "Confidence", "21D Change", "Risk Category", "Liquidity Warning",
        "Consistent Leader", "Last Price", "Daily Change %"
    ])

    for s in res.stocks:
        writer.writerow([
            s.rank,
            s.code,
            s.company_name,
            f"{s.score:.2f}",
            s.tier,
            s.confidence,
            f"{s.rank_change_21d:+.2f}" if s.rank_change_21d is not None else "N/A",
            s.risk_category,
            "Yes" if s.liquidity_warning else "No",
            "Yes" if s.consistent_leader else "No",
            f"{s.last_price:.2f}" if s.last_price is not None else "N/A",
            f"{s.daily_change_pct:+.2f}%" if s.daily_change_pct is not None else "N/A"
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=stockpulse_rankings_{horizon}_{res.as_of}.csv"}
    )

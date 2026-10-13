"""app/backend/routers/stocks.py
Stocks listing, detail, and technical price history chart endpoints.
Strictly user-facing: zero model names, metrics, or IC leaks.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from app.backend.db.session import get_db
from app.backend.services.serving import serving_repo
from app.backend.schemas.stocks import (
    StockListItem,
    StockDetailResponse,
    StockReasons,
    PricePoint,
    StockPricesResponse
)
from app.backend.models.entities import ModelSetting
from app.backend.routers.deps import get_current_user

router = APIRouter(prefix="/stocks", tags=["stocks"], dependencies=[Depends(get_current_user)])


def get_active_serving_snapshot(db: Session):
    db_settings = {}
    rows = db.query(ModelSetting).all()
    for r in rows:
        db_settings[r.horizon] = (r.mode, r.model_id)
    active_models = serving_repo.resolve_active_models(db_settings)
    return serving_repo.compute_serving_snapshot(active_models)


@router.get("", response_model=List[StockListItem])
def list_stocks(
    search: Optional[str] = None,
    risk: Optional[str] = None,
    tier_short: Optional[str] = None,
    tier_mid: Optional[str] = None,
    tier_long: Optional[str] = None,
    status: Optional[str] = None,
    exclude_liquidity_warning: bool = False,
    sort_by: str = "overall_upside_score",
    ascending: bool = False,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    snapshot = get_active_serving_snapshot(db)
    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}
    items = []

    for s in snapshot["stocks"]:
        sid = s["series_id"]
        meta = stocks_meta.get(sid, {})
        comp_name = meta.get("company_name", sid.split(".")[0])
        last_price = meta.get("last_price")
        daily_chg = meta.get("daily_change_pct")

        # Filters
        if search:
            q = search.lower()
            if q not in sid.lower() and q not in comp_name.lower():
                continue
        if risk and s["risk_category"] != risk:
            continue
        if tier_short and s["tier_short"] != tier_short:
            continue
        if tier_mid and s["tier_mid"] != tier_mid:
            continue
        if tier_long and s["tier_long"] != tier_long:
            continue
        if status and s["status"] != status:
            continue
        if exclude_liquidity_warning and s["liquidity_warning"]:
            continue

        item = StockListItem(
            code=sid,
            company_name=comp_name,
            last_price=last_price,
            daily_change_pct=daily_chg,
            overall_upside_score=s["overall_upside_score"],
            tier_short=s["tier_short"],
            tier_mid=s["tier_mid"],
            tier_long=s["tier_long"],
            risk_category=s["risk_category"],
            liquidity_warning=s["liquidity_warning"],
            consistent_leader=s["consistent_leader"],
            status=s["status"]
        )
        items.append(item)

    # Sorting
    def sort_key(x: StockListItem):
        val = getattr(x, sort_by, None)
        if val is None:
            return -1e9 if not ascending else 1e9
        return val

    items.sort(key=sort_key, reverse=not ascending)
    return items[skip : skip + limit]


@router.get("/{code}", response_model=StockDetailResponse)
def get_stock_detail(
    code: str,
    db: Session = Depends(get_db)
):
    snapshot = get_active_serving_snapshot(db)
    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}

    code_upper = code.upper()
    stock_record = next((s for s in snapshot["stocks"] if s["series_id"] == code_upper), None)
    if not stock_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Stock {code} not found.")

    meta = stocks_meta.get(code_upper, {})
    comp_name = meta.get("company_name", code_upper.split(".")[0])
    last_price = meta.get("last_price")
    daily_chg = meta.get("daily_change_pct")
    r_52w = meta.get("range_52w")
    raw_sub = meta.get("sub_type")
    sub_type = str(raw_sub) if raw_sub not in [None, 0, 0.0, "0.0", "nan", "0"] else "Ordinary"


    # One sentence confidence explanations per horizon
    conf_short = stock_record["confidence_short"]
    conf_mid = stock_record["confidence_mid"]
    conf_long = stock_record["confidence_long"]

    expl_map = {
        "Low confidence": "Evaluation indicates limited predictive signal that requires caution.",
        "Very low confidence": "Evaluation indicates very weak or unverified predictive signal across historical periods.",
        "Moderate confidence": "Evaluation indicates consistent predictive performance across historical testing."
    }

    return StockDetailResponse(
        code=code_upper,
        company_name=comp_name,
        last_price=last_price,
        daily_change_pct=daily_chg,
        range_52w=r_52w,
        sub_type=sub_type,
        status=stock_record["status"],
        last_traded_date=stock_record["last_traded_date"],
        overall_upside_score=stock_record["overall_upside_score"],
        consistent_leader=stock_record["consistent_leader"],
        score_short=stock_record["score_short"],
        tier_short=stock_record["tier_short"],
        confidence_short=conf_short,
        confidence_explanation_short=expl_map.get(conf_short, "Evaluated upside potential for short horizon."),
        rank_change_21d_short=stock_record["rank_change_21d_short"],
        reasons_short=StockReasons(**stock_record["reasons_short"]),
        score_mid=stock_record["score_mid"],
        tier_mid=stock_record["tier_mid"],
        confidence_mid=conf_mid,
        confidence_explanation_mid=expl_map.get(conf_mid, "Evaluated upside potential for medium horizon."),
        rank_change_21d_mid=stock_record["rank_change_21d_mid"],
        reasons_mid=StockReasons(**stock_record["reasons_mid"]),
        score_long=stock_record["score_long"],
        tier_long=stock_record["tier_long"],
        confidence_long=conf_long,
        confidence_explanation_long=expl_map.get(conf_long, "Evaluated upside potential for long horizon."),
        rank_change_21d_long=stock_record["rank_change_21d_long"],
        reasons_long=StockReasons(**stock_record["reasons_long"]),
        risk_category=stock_record["risk_category"],
        liquidity_warning=stock_record["liquidity_warning"],
        thin_trading_flag=stock_record["thin_trading_flag"],
        thin_trading_note=stock_record.get("thin_trading_note"),
        rank_change_explanation=stock_record["rank_change_explanation"],
        market_volatility_note=stock_record.get("market_volatility_note")
    )


@router.get("/{code}/prices", response_model=StockPricesResponse)
def get_stock_prices(
    code: str,
    range: str = Query("1y", pattern="^(1m|3m|6m|1y|5y|max)$")
):
    code_upper = code.upper()
    try:
        df_p = serving_repo.get_stock_prices_df(code_upper)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Price history for {code} not found.")

    df_p = df_p.sort_values("Date").reset_index(drop=True)
    n_days_map = {
        "1m": 21,
        "3m": 63,
        "6m": 126,
        "1y": 252,
        "5y": 252 * 5,
        "max": len(df_p)
    }
    limit_bars = n_days_map.get(range, 252)
    sub = df_p.iloc[-limit_bars:]

    bars = []
    for _, row in sub.iterrows():
        bars.append(PricePoint(
            date=str(row["Date"])[:10],
            open=float(row["Open"]) if pd.notnull(row.get("Open")) else None,
            high=float(row["High"]) if pd.notnull(row.get("High")) else None,
            low=float(row["Low"]) if pd.notnull(row.get("Low")) else None,
            close=float(row["Close"]) if pd.notnull(row.get("Close")) else None,
            volume=float(row["TradeVolume"]) if pd.notnull(row.get("TradeVolume")) else None,
            sma_20=float(row["sma_20"]) if pd.notnull(row.get("sma_20")) else None,
            sma_50=float(row["sma_50"]) if pd.notnull(row.get("sma_50")) else None,
            sma_200=float(row["sma_200"]) if pd.notnull(row.get("sma_200")) else None,
            rsi_14=float(row["rsi_14"]) if pd.notnull(row.get("rsi_14")) else None,
            macd_line=float(row["macd_line"]) if pd.notnull(row.get("macd_line")) else None,
            macd_signal=float(row["macd_signal"]) if pd.notnull(row.get("macd_signal")) else None,
            macd_hist=float(row["macd_hist"]) if pd.notnull(row.get("macd_hist")) else None,
            bollinger_upper=float(row["bollinger_upper"]) if pd.notnull(row.get("bollinger_upper")) else None,
            bollinger_middle=float(row["bollinger_middle"]) if pd.notnull(row.get("bollinger_middle")) else None,
            bollinger_lower=float(row["bollinger_lower"]) if pd.notnull(row.get("bollinger_lower")) else None
        ))

    return StockPricesResponse(code=code_upper, range=range, prices=bars)

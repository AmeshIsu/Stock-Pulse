"""app/backend/schemas/stocks.py
Stocks request and response schemas (strictly user-facing, zero model names or metrics).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class StockListItem(BaseModel):
    code: str
    company_name: str
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None
    overall_upside_score: Optional[float] = None
    tier_short: str
    tier_mid: str
    tier_long: str
    risk_category: str
    liquidity_warning: bool
    consistent_leader: bool
    status: str


class StockReasons(BaseModel):
    top_positive: List[str]
    top_negative: List[str]


class StockDetailResponse(BaseModel):
    code: str
    company_name: str
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None
    range_52w: Optional[List[float]] = None
    sub_type: Optional[str] = None
    status: str
    last_traded_date: str
    overall_upside_score: Optional[float] = None
    consistent_leader: bool

    score_short: Optional[float] = None
    tier_short: str
    confidence_short: str
    confidence_explanation_short: str
    rank_change_21d_short: Optional[float] = None
    reasons_short: StockReasons

    score_mid: Optional[float] = None
    tier_mid: str
    confidence_mid: str
    confidence_explanation_mid: str
    rank_change_21d_mid: Optional[float] = None
    reasons_mid: StockReasons

    score_long: Optional[float] = None
    tier_long: str
    confidence_long: str
    confidence_explanation_long: str
    rank_change_21d_long: Optional[float] = None
    reasons_long: StockReasons

    risk_category: str
    trade_day_share_63m: Optional[float] = None
    liquidity_warning: bool
    thin_trading_flag: bool
    thin_trading_note: Optional[str] = None
    rank_change_explanation: str
    market_volatility_note: Optional[str] = None


class PricePoint(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None


class StockPricesResponse(BaseModel):
    code: str
    range: str
    prices: List[PricePoint]

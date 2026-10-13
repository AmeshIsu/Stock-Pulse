"""app/backend/schemas/rankings.py
Rankings, leaders, and movers response schemas (strictly user-facing, zero model names or metrics).
"""
from typing import Optional, List
from pydantic import BaseModel


class RankingStockItem(BaseModel):
    rank: int
    code: str
    company_name: str
    score: float
    tier: str
    confidence: str
    rank_change_21d: Optional[float] = None
    risk_category: str
    liquidity_warning: bool
    consistent_leader: bool
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None


class RankingsResponse(BaseModel):
    horizon: str
    as_of: str
    weights_used: Optional[dict] = None
    stocks: List[RankingStockItem]


class ConsistentLeaderItem(BaseModel):
    code: str
    company_name: str
    overall_upside_score: float
    tier_short: str
    tier_mid: str
    tier_long: str
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None


class LeadersResponse(BaseModel):
    as_of: str
    count: int
    leaders: List[ConsistentLeaderItem]


class MoverItem(BaseModel):
    code: str
    company_name: str
    horizon: str
    score: float
    rank_change_21d: float
    direction: str  # "up" | "down"
    explanation: str = "indicative: computed with the current model, not an out-of-sample result"


class MoversResponse(BaseModel):
    as_of: str
    biggest_gainers: List[MoverItem]
    biggest_decliners: List[MoverItem]

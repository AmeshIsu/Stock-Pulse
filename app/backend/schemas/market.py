"""app/backend/schemas/market.py
Market overview response schema (strictly user-facing, zero model names or metrics).
"""
from typing import Optional, Dict
from pydantic import BaseModel


class TierCounts(BaseModel):
    High: int
    Neutral: int
    Low: int


class MarketSummaryResponse(BaseModel):
    as_of: str
    data_version: str
    tier_counts_short: TierCounts
    tier_counts_mid: TierCounts
    tier_counts_long: TierCounts
    consistent_leaders_count: int
    market_volatility_note: Optional[str] = None
    disclaimer: str = (
        "Stock Pulse rankings are for informational research purposes only and do not constitute "
        "investment advice or recommendations to buy or sell securities."
    )

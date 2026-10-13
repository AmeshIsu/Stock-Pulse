"""Stock Pulse Tier Assignment & Consistent Leader Evaluation.

Shared implementation of cross-sectional tier binning:
For N rankable stocks and cutoff fraction f:
n_tier = max(1, round(f * N))
High = the n_tier highest scores
Low = the n_tier lowest scores
Neutral = otherwise
Ties broken by series_id for determinism.

Consistent Leader:
High in at least two of the three horizons, Low in none, and no liquidity warning.
"""
import pandas as pd
import numpy as np


def assign_tiers(scores: pd.Series, cutoff_fraction: float) -> pd.Series:
    """Assign High, Neutral, Low tiers according to Section A1.
    
    Args:
        scores: Series indexed by series_id with numeric percentile scores [0, 100].
        cutoff_fraction: Float cutoff fraction (e.g. 0.25, 0.30).
        
    Returns:
        Series indexed by series_id with tier values ('High', 'Neutral', 'Low', or NaN).
    """
    if scores.empty:
        return pd.Series(dtype=object)

    valid_mask = scores.notna()
    valid_scores = scores[valid_mask]
    N = len(valid_scores)
    
    result = pd.Series(index=scores.index, dtype=object)
    result[~valid_mask] = np.nan
    
    if N == 0:
        return result

    n_tier = max(1, int(round(cutoff_fraction * N)))

    # Sort ascending by score, ascending by series_id so that within tied scores,
    # lower rank stays with Low/Neutral and higher rank goes to Neutral/High (monotonic).
    sort_df = pd.DataFrame({
        "series_id": valid_scores.index,
        "score": valid_scores.values
    }).sort_values(by=["score", "series_id"], ascending=[True, True]).reset_index(drop=True)

    low_ids = set(sort_df.iloc[:n_tier]["series_id"])
    high_ids = set(sort_df.iloc[-n_tier:]["series_id"])


    for sid in valid_scores.index:
        if sid in high_ids:
            result[sid] = "High"
        elif sid in low_ids:
            result[sid] = "Low"
        else:
            result[sid] = "Neutral"

    return result


def is_consistent_leader(tier_short: str, tier_mid: str, tier_long: str, liquidity_warning: bool) -> bool:
    """Determine Consistent Leader status according to Section A2.
    
    Rule:
        High in at least 2 of the 3 horizons, Low in none, and no liquidity warning.
    """
    tiers = [tier_short, tier_mid, tier_long]
    high_count = sum(1 for t in tiers if t == "High")
    has_low = any(t == "Low" for t in tiers)
    return bool((high_count >= 2) and (not has_low) and (not liquidity_warning))

"""src/scoring/hit_rate.py
Standardized Top-Tier Hit Rate calculation for Stock Pulse.

Canonical Definition:
The share of High-tier stock-dates, using non-overlapping rebalance periods,
whose forward return beat that date's cross-sectional median, on development
out-of-sample predictions only.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.scoring.tiers import assign_tiers

H_DAYS_MAP = {
    "short": 21,
    "mid": 63,
    "long": 252
}

DEFAULT_CUTOFFS = {
    "short": 0.3333,
    "mid": 0.3333,
    "long": 0.3333
}


def compute_top_tier_hit_rate(
    df_oos: pd.DataFrame,
    horizon: str,
    cutoff: float = 0.3333,
    n_boot: int = 2000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Calculate the share of High-tier stock-dates, using non-overlapping rebalance periods,
    whose forward return beat that date's cross-sectional median, on development
    out-of-sample predictions only.

    Parameters
    ----------
    df_oos : pd.DataFrame
        Out-of-sample predictions containing 'Date', 'series_id', 'pred_score',
        f'fwd_ret_{horizon}', 'split_type', and 'fold'.
    horizon : str
        'short', 'mid', or 'long'.
    cutoff : float
        Fraction of stocks assigned to High tier (default 0.3333).
    n_boot : int
        Number of cluster bootstrap draws across folds for confidence interval.
    random_seed : int
        RNG seed for bootstrap replication.

    Returns
    -------
    dict
        Dictionary containing:
        - hit_rate_pct: float (e.g. 49.37)
        - hit_rate_90_ci: list of [float, float]
        - observations: int
        - n_effective_blocks: int
    """
    h = horizon.lower()
    h_days = H_DAYS_MAP[h]
    fwd_ret_col = f"fwd_ret_{h}"

    # Filter strictly to development split
    dev_df = df_oos[df_oos["split_type"] == "dev"].copy()
    if dev_df.empty:
        return {
            "hit_rate_pct": None,
            "hit_rate_90_ci": None,
            "observations": 0,
            "n_effective_blocks": 0
        }

    # Percentile ranks per cross-section
    dev_df["rank"] = dev_df.groupby("Date")["pred_score"].rank(ascending=True, method="average")
    dev_df["n_stocks"] = dev_df.groupby("Date")["pred_score"].transform("count")
    dev_df["pct_rank"] = (dev_df["rank"] - 0.5) / dev_df["n_stocks"]
    dev_df["score"] = dev_df["pct_rank"] * 100.0

    # Assign tiers per date
    tier_list = []
    for dt, grp in dev_df.groupby("Date"):
        s_scores = grp.set_index("series_id")["score"]
        t_assigned = assign_tiers(s_scores, cutoff)
        for sid, t_val in t_assigned.items():
            tier_list.append((dt, sid, t_val))
    t_df = pd.DataFrame(tier_list, columns=["Date", "series_id", "tier"])
    dev_df = pd.merge(dev_df, t_df, on=["Date", "series_id"])

    # Forward return vs cross-sectional median
    med_ret = dev_df.groupby("Date")[fwd_ret_col].transform("median")
    dev_df["beat_med"] = np.where(
        dev_df[fwd_ret_col] > med_ret, 1.0,
        np.where(dev_df[fwd_ret_col] == med_ret, 0.5, 0.0)
    )

    # Non-overlapping rebalance dates
    dates = sorted(dev_df["Date"].unique())
    rebal_dates = set(dates[::h_days])
    rebal_sub = dev_df[dev_df["Date"].isin(rebal_dates)].copy()
    n_eff = len(rebal_dates)

    # Filter to High tier only
    high_sub = rebal_sub[rebal_sub["tier"] == "High"]
    if high_sub.empty:
        return {
            "hit_rate_pct": None,
            "hit_rate_90_ci": None,
            "observations": 0,
            "n_effective_blocks": n_eff
        }

    hit_rate = float(high_sub["beat_med"].mean()) * 100.0

    # Fold cluster bootstrap for 90% CI
    y_beat_med = [grp["beat_med"].values for _, grp in high_sub.groupby("fold")]
    n_y = len(y_beat_med)
    rng = np.random.default_rng(random_seed)
    boot_hrs = []
    for _ in range(n_boot):
        sample_idx = rng.choice(n_y, size=n_y, replace=True)
        b_cat = np.concatenate([y_beat_med[i] for i in sample_idx])
        boot_hrs.append(float(np.mean(b_cat) * 100.0))

    hr_ci = [
        round(float(np.percentile(boot_hrs, 5.0)), 2),
        round(float(np.percentile(boot_hrs, 95.0)), 2)
    ]

    return {
        "hit_rate_pct": round(hit_rate, 2),
        "hit_rate_90_ci": hr_ci,
        "observations": len(high_sub),
        "n_effective_blocks": n_eff
    }

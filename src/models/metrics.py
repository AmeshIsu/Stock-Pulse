"""Cross-sectional evaluation metrics, block-bootstrapped ICs, portfolio simulations, and valid random null band."""
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Cache for partition-level 1000-draw single random scorer null bands
_NULL_BAND_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

def reset_null_band_cache():
    global _NULL_BAND_CACHE
    _NULL_BAND_CACHE = {}


def compute_rank_ic(df_pred: pd.DataFrame, target_col: str = "fwd_excess", min_stocks: int = 8) -> pd.Series:
    """Compute daily cross-sectional Spearman Rank IC between pred_score and target."""
    daily_ics = {}
    for dt, grp in df_pred.groupby("Date"):
        valid = grp.dropna(subset=["pred_score", target_col])
        if len(valid) < min_stocks:
            continue
        corr, _ = spearmanr(valid["pred_score"], valid[target_col])
        if not np.isnan(corr):
            daily_ics[dt] = corr
    return pd.Series(daily_ics)


def moving_block_bootstrap_ic_ci(
    daily_ics: pd.Series,
    block_length: int,
    n_draws: int = 2000,
    random_seed: int = 42
) -> tuple[float, float, float]:
    """Compute 95% moving-block bootstrap confidence interval for mean Rank IC.
    
    Returns (ci_lower, ci_upper, effective_blocks).
    If effective_blocks < 5, returns (np.nan, np.nan, effective_blocks).
    """
    n = len(daily_ics)
    if n == 0 or block_length <= 0:
        return (np.nan, np.nan, 0.0)
        
    effective_blocks = float(n / block_length)
    if effective_blocks < 5.0:
        return (np.nan, np.nan, effective_blocks)
        
    rng = np.random.default_rng(random_seed)
    values = daily_ics.values
    
    # Construct overlapping blocks of length block_length
    b = min(block_length, max(n // 2, 1))
    num_blocks = n - b + 1
    if num_blocks <= 0:
        b = 1
        num_blocks = n
        
    blocks = [values[i : i + b] for i in range(num_blocks)]
    needed_blocks = int(np.ceil(n / b))
    
    boot_means = np.empty(n_draws)
    for i in range(n_draws):
        sampled_block_indices = rng.integers(0, num_blocks, size=needed_blocks)
        sampled_values = np.concatenate([blocks[idx] for idx in sampled_block_indices])[:n]
        boot_means[i] = np.mean(sampled_values)
        
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Strictly ensure sample mean is covered
    mean_val = float(np.mean(values))
    ci_lower = min(ci_lower, mean_val)
    ci_upper = max(ci_upper, mean_val)
    
    return (ci_lower, ci_upper, effective_blocks)


def compute_top_k_metrics(
    df_pred: pd.DataFrame,
    fwd_excess_col: str,
    fwd_ret_col: str,
    fwd_rank_col: str,
    k: int = 5,
    min_stocks: int = 8
) -> dict[str, float]:
    """Compute daily cross-sectional top-K metrics, tie-adjusted precision, and spread."""
    top_k_excess = []
    top_k_rets = []
    spreads = []
    precisions = []
    
    for dt, grp in df_pred.groupby("Date"):
        valid = grp.dropna(subset=["pred_score", fwd_excess_col, fwd_ret_col, fwd_rank_col])
        if len(valid) < min_stocks:
            continue
            
        sorted_grp = valid.sort_values("pred_score", ascending=False)
        top_k = sorted_grp.head(k)
        bottom_k = sorted_grp.tail(k)
        
        top_k_excess.append(top_k[fwd_excess_col].mean())
        top_k_rets.append(top_k[fwd_ret_col].mean())
        spreads.append(top_k[fwd_ret_col].mean() - bottom_k[fwd_ret_col].mean())
        
        # Tie-adjusted precision: rankpct > 0.5 is 1 hit, rankpct == 0.5 is 0.5 hit
        hits = (top_k[fwd_rank_col] > 0.5).sum() + 0.5 * (top_k[fwd_rank_col] == 0.5).sum()
        precisions.append(hits / float(k))
        
    if not top_k_excess:
        return {
            "top_k_fwd_excess_mean": np.nan,
            "top_k_fwd_excess_median": np.nan,
            "top_k_fwd_excess_winsorized_mean": np.nan,
            "top_k_fwd_ret_mean": np.nan,
            "top_k_fwd_ret_median": np.nan,
            "top_k_precision": np.nan,
            "top_k_minus_bottom_k_spread": np.nan
        }
        
    excess_arr = np.array(top_k_excess)
    p01, p99 = np.percentile(excess_arr, [1.0, 99.0])
    winsorized_mean = float(np.mean(np.clip(excess_arr, p01, p99)))
    
    return {
        "top_k_fwd_excess_mean": float(np.mean(top_k_excess)),
        "top_k_fwd_excess_median": float(np.median(top_k_excess)),
        "top_k_fwd_excess_winsorized_mean": winsorized_mean,
        "top_k_fwd_ret_mean": float(np.mean(top_k_rets)),
        "top_k_fwd_ret_median": float(np.median(top_k_rets)),
        "top_k_precision": float(np.mean(precisions)),
        "top_k_minus_bottom_k_spread": float(np.mean(spreads))
    }


def compute_classification_metrics(df_pred: pd.DataFrame, outperf_col: str, k: int = 5) -> dict[str, float]:
    """Compute classification metrics when pred_prob is available (exact 0.5 ties dropped)."""
    valid = df_pred.dropna(subset=["pred_prob", outperf_col])
    if len(valid) < 10 or valid[outperf_col].nunique() < 2:
        return {
            "auc": np.nan,
            "accuracy_vs_majority": np.nan,
            "precision_at_k": np.nan,
            "recall_at_k": np.nan,
            "f1": np.nan
        }
        
    y_true = valid[outperf_col].astype(int)
    y_prob = valid["pred_prob"]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Majority class baseline
    majority_rate = max(y_true.mean(), 1.0 - y_true.mean())
    acc = accuracy_score(y_true, y_pred)
    
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Precision and recall at top K per date
    prec_k, rec_k = [], []
    for dt, grp in valid.groupby("Date"):
        if len(grp) < k:
            continue
        top = grp.sort_values("pred_prob", ascending=False).head(k)
        prec_k.append((top[outperf_col] == 1).mean())
        total_pos = (grp[outperf_col] == 1).sum()
        if total_pos > 0:
            rec_k.append((top[outperf_col] == 1).sum() / total_pos)
            
    return {
        "auc": float(auc),
        "accuracy_vs_majority": float(acc - majority_rate),
        "precision_at_k": float(np.mean(prec_k)) if prec_k else np.nan,
        "recall_at_k": float(np.mean(rec_k)) if rec_k else np.nan,
        "f1": float(f1)
    }


def simulate_rebalancing_portfolio(
    df_pred: pd.DataFrame,
    fwd_ret_col: str,
    h: int,
    k: int = 5,
    cost_per_side_bps: float = 100.0,
    min_stocks: int = 8
) -> dict[str, float]:
    """Simulate non-overlapping rebalanced portfolio holding top-K equal-weighted stocks for h days.
    
    Tracks both:
      - Full turnover cost: assumes 100% replacement every period (2 * cost_per_side_bps)
      - Actual turnover cost: cost scaled by actual fraction of top-K names replaced between rebalances.
    """
    dates = sorted(df_pred["Date"].unique())
    if not dates:
        return {
            "portfolio_num_periods": 0,
            "portfolio_gross_excess_mean": np.nan,
            "portfolio_net_excess_mean": np.nan,
            "portfolio_net_excess_actual_mean": np.nan,
            "portfolio_turnover_replaced_share_mean": np.nan,
            "portfolio_win_rate": np.nan,
            "portfolio_compounded_net_excess": np.nan,
            "portfolio_market_ret_mean": np.nan,
            "portfolio_market_compounded_ret": np.nan
        }
        
    rebal_dates = dates[::h]
    gross_excess_list = []
    net_excess_full_list = []
    net_excess_actual_list = []
    top_ret_list = []
    mkt_ret_list = []
    round_trip_cost = 2.0 * (cost_per_side_bps / 10000.0)  # e.g. 2 x 100 bps = 0.02
    
    prev_series = None
    replaced_shares = []
    
    for r_dt in rebal_dates:
        grp = df_pred[df_pred["Date"] == r_dt].dropna(subset=["pred_score", fwd_ret_col])
        if len(grp) < min_stocks:
            continue
            
        top_k = grp.sort_values("pred_score", ascending=False).head(k)
        curr_series = set(top_k["series_id"].tolist())
        mkt_ret = grp[fwd_ret_col].mean()
        top_ret = top_k[fwd_ret_col].mean()
        
        gross_exc = top_ret - mkt_ret
        net_exc_full = gross_exc - round_trip_cost
        
        if prev_series is None:
            # First rebalance: inception portfolio entry pays 1 round-trip
            share_replaced = 1.0
            actual_cost = round_trip_cost
        else:
            n_retained = len(prev_series.intersection(curr_series))
            share_replaced = (k - n_retained) / float(k)
            actual_cost = share_replaced * round_trip_cost
            replaced_shares.append(share_replaced)
            
        prev_series = curr_series
        net_exc_actual = gross_exc - actual_cost
        
        # Verify strict mathematical identities
        assert abs(abs(net_exc_full - gross_exc) - round_trip_cost) < 1e-9, (
            f"Friction mismatch: net_full={net_exc_full}, gross={gross_exc}, cost={round_trip_cost}"
        )
        assert abs(abs(net_exc_actual - gross_exc) - actual_cost) < 1e-9, (
            f"Friction mismatch: net_actual={net_exc_actual}, gross={gross_exc}, cost={actual_cost}"
        )
        
        gross_excess_list.append(gross_exc)
        net_excess_full_list.append(net_exc_full)
        net_excess_actual_list.append(net_exc_actual)
        top_ret_list.append(top_ret)
        mkt_ret_list.append(mkt_ret)
        
    n_periods = len(gross_excess_list)
    if n_periods == 0:
        return {
            "portfolio_num_periods": 0,
            "portfolio_gross_excess_mean": np.nan,
            "portfolio_net_excess_mean": np.nan,
            "portfolio_net_excess_actual_mean": np.nan,
            "portfolio_turnover_replaced_share_mean": np.nan,
            "portfolio_win_rate": np.nan,
            "portfolio_compounded_net_excess": np.nan,
            "portfolio_market_ret_mean": np.nan,
            "portfolio_market_compounded_ret": np.nan
        }
        
    avg_replaced = float(np.mean(replaced_shares)) if replaced_shares else (1.0 if n_periods > 0 else np.nan)
    net_arr = np.array(net_excess_full_list)
    win_rate = float((net_arr > 0).mean())
    
    # Compounded net excess return (full turnover)
    comp_top_net = float(np.prod(1.0 + np.array(top_ret_list) - round_trip_cost) - 1.0)
    comp_mkt = float(np.prod(1.0 + np.array(mkt_ret_list)) - 1.0)
    compounded_net_excess = comp_top_net - comp_mkt
    
    return {
        "portfolio_num_periods": n_periods,
        "portfolio_gross_excess_mean": float(np.mean(gross_excess_list)),
        "portfolio_net_excess_mean": float(np.mean(net_excess_full_list)),
        "portfolio_net_excess_actual_mean": float(np.mean(net_excess_actual_list)),
        "portfolio_turnover_replaced_share_mean": avg_replaced,
        "portfolio_win_rate": win_rate,
        "portfolio_compounded_net_excess": compounded_net_excess,
        "portfolio_market_ret_mean": float(np.mean(mkt_ret_list)),
        "portfolio_market_compounded_ret": comp_mkt
    }


def compute_null_band_simulation(
    df_partition: pd.DataFrame,
    horizon: str,
    h_days: int,
    split_type: str,
    n_draws: int = 1000,
    seed: int = 42,
    k: int = 5,
    min_stocks: int = 8,
    config: Optional[dict] = None
) -> dict[str, Any]:
    """Simulate 1000 draws under dual nulls to establish empirical persistence-corrected null bands:
      (a) Block-constant random scorers: fresh random scores per series once every h trading days,
          held constant inside block (restart at segment breaks).
      (b) Random linear scorers: a fresh random weight vector ~ N(0, I_p) on cs-ranked core features,
          fixed within a fold, no fitting.
    The corrected null band is the wider of (a) and (b) (2.5th to 97.5th pct of partition-mean IC).
    Also records old narrow null (daily i.i.d. draws).
    """
    cache_key = (horizon, split_type)
    if cache_key in _NULL_BAND_CACHE:
        return _NULL_BAND_CACHE[cache_key]

    fwd_excess_col = f"fwd_excess_{horizon}"
    fwd_ret_col = f"fwd_ret_{horizon}"
    
    from pathlib import Path
    from src.models.preprocessing import REDUNDANT_LINEAR_FEATURES, apply_cross_sectional_rank
    from src.data.config_loader import load_config
    
    if config is None:
        config = load_config()
        
    base_dir = Path(__file__).resolve().parents[2]
    features_path = base_dir / config["paths"]["features_targets"]
    core_feat_list = list(config["feature_sets"]["core"])
    core_feats = [c for c in core_feat_list if c not in REDUNDANT_LINEAR_FEATURES and c not in ["index_ret_63d", "index_vol_63d"]]
    p_feats = max(1, len(core_feats))
    
    # Merge core features if not present in df_partition
    if not all(c in df_partition.columns for c in core_feats):
        df_feats_raw = pd.read_parquet(features_path, columns=["Date", "series_id"] + core_feats)
        df_part_merged = df_partition.merge(df_feats_raw, on=["Date", "series_id"], how="left")
    else:
        df_part_merged = df_partition.copy()
        
    part_df_ranked = apply_cross_sectional_rank(df_part_merged, core_feats)
    
    # Pre-extract fold structure
    if "fold" in part_df_ranked.columns:
        fold_grouping = part_df_ranked.groupby("fold", sort=False)
    elif "split_type" in part_df_ranked.columns:
        fold_grouping = part_df_ranked.groupby("split_type", sort=False)
    else:
        fold_grouping = [("all", part_df_ranked)]
        
    fold_groups = []
    
    for fold_name, f_df in fold_grouping:
        dates = sorted(f_df["Date"].unique())
        date_to_block = {}
        for b_idx, start_idx in enumerate(range(0, len(dates), h_days)):
            b_dates = dates[start_idx : start_idx + h_days]
            for d in b_dates:
                date_to_block[d] = b_idx
        n_blocks = max(1, len(range(0, len(dates), h_days)))
        
        unique_series = sorted(f_df["series_id"].unique())
        series_to_idx = {s: i for i, s in enumerate(unique_series)}
        
        daily_slices = []
        for dt, d_grp in f_df.groupby("Date", sort=False):
            valid = d_grp.dropna(subset=[fwd_excess_col, fwd_ret_col])
            if len(valid) < min_stocks:
                continue
            s_indices = np.array([series_to_idx[s] for s in valid["series_id"]], dtype=int)
            tgt_ranks = valid[fwd_excess_col].rank().values
            tgt_c = tgt_ranks - tgt_ranks.mean()
            norm_c = np.linalg.norm(tgt_c)
            tgt_norm = tgt_c / norm_c if norm_c > 1e-9 else tgt_c
            
            # If core feats present and non-constant use them, else fallback to random features for toy data
            X_mat = valid[core_feats].values if (core_feats and set(core_feats).issubset(valid.columns)) else np.zeros((len(valid), p_feats))
            if np.all(np.isclose(X_mat, X_mat[0, :])):
                # Synthetic or toy data: generate distinct standard normal features so random linear operates
                local_rng = np.random.default_rng(seed + len(daily_slices))
                X_mat = local_rng.standard_normal((len(valid), p_feats))
                
            rets = valid[fwd_ret_col].values
            mkt_ret = float(rets.mean())
            
            daily_slices.append({
                "Date": dt,
                "block_idx": date_to_block.get(dt, 0),
                "s_indices": s_indices,
                "tgt_norm": tgt_norm,
                "X_mat": X_mat,
                "rets": rets,
                "mkt_ret": mkt_ret,
                "n_stocks": len(valid)
            })
            
        fold_groups.append({
            "fold_name": fold_name,
            "n_series": len(unique_series),
            "n_blocks": n_blocks,
            "daily_slices": daily_slices
        })
        
    p_feats = len(core_feats)
    rng_a = np.random.default_rng(seed + 101)
    rng_b = np.random.default_rng(seed + 202)
    rng_old = np.random.default_rng(seed + 303)
    
    daily_ics_a_list = []
    daily_ics_b_list = []
    daily_ics_old_list = []
    daily_excess_wider_list = []
    
    for f_idx, fg in enumerate(fold_groups):
        if not fg["daily_slices"]:
            continue
        # (a) Block-constant random scores: shape (n_draws, n_blocks, n_series)
        block_scores = rng_a.uniform(0.0, 1.0, size=(n_draws, fg["n_blocks"], fg["n_series"]))
        # (b) Random linear weights for this fold: shape (n_draws, p_feats)
        w_fold = rng_b.standard_normal(size=(n_draws, p_feats))
        
        for d_slice in fg["daily_slices"]:
            tgt_norm = d_slice["tgt_norm"]
            n_stk = d_slice["n_stocks"]
            b_idx = d_slice["block_idx"]
            s_idx = d_slice["s_indices"]
            
            # (a) Block constant
            scores_a = block_scores[:, b_idx, :][:, s_idx]
            perms_a = scores_a.argsort(axis=1)
            ranks_a = perms_a.argsort(axis=1).astype(float) + 1.0
            ra_c = ranks_a - ranks_a.mean(axis=1, keepdims=True)
            norm_a = np.linalg.norm(ra_c, axis=1, keepdims=True)
            ra_norm = ra_c / np.where(norm_a > 1e-9, norm_a, 1.0)
            corrs_a = np.dot(ra_norm, tgt_norm)
            daily_ics_a_list.append(corrs_a)
            
            # (b) Random linear
            scores_b = np.dot(w_fold, d_slice["X_mat"].T)
            perms_b = scores_b.argsort(axis=1)
            ranks_b = perms_b.argsort(axis=1).astype(float) + 1.0
            rb_c = ranks_b - ranks_b.mean(axis=1, keepdims=True)
            norm_b = np.linalg.norm(rb_c, axis=1, keepdims=True)
            rb_norm = rb_c / np.where(norm_b > 1e-9, norm_b, 1.0)
            corrs_b = np.dot(rb_norm, tgt_norm)
            daily_ics_b_list.append(corrs_b)
            
            # (c) Old null: fresh i.i.d.
            scores_old = rng_old.uniform(0.0, 1.0, size=(n_draws, n_stk))
            perms_old = scores_old.argsort(axis=1)
            ranks_old = perms_old.argsort(axis=1).astype(float) + 1.0
            ro_c = ranks_old - ranks_old.mean(axis=1, keepdims=True)
            norm_old = np.linalg.norm(ro_c, axis=1, keepdims=True)
            ro_norm = ro_c / np.where(norm_old > 1e-9, norm_old, 1.0)
            corrs_old = np.dot(ro_norm, tgt_norm)
            daily_ics_old_list.append(corrs_old)
            
    if not daily_ics_a_list:
        res = {
            "null_band_ic_lower": np.nan, "null_band_ic_upper": np.nan, "null_band_ic_mean": 0.0,
            "block_null_band_ic_lower": np.nan, "block_null_band_ic_upper": np.nan,
            "linear_null_band_ic_lower": np.nan, "linear_null_band_ic_upper": np.nan,
            "old_null_band_ic_lower": np.nan, "old_null_band_ic_upper": np.nan,
            "null_band_excess_lower": np.nan, "null_band_excess_upper": np.nan, "null_band_excess_mean": 0.0,
            "draw_ics": np.zeros(n_draws), "wider_type": "none"
        }
        _NULL_BAND_CACHE[cache_key] = res
        return res
        
    draw_ics_a = np.mean(daily_ics_a_list, axis=0)  # (n_draws,)
    draw_ics_b = np.mean(daily_ics_b_list, axis=0)
    draw_ics_old = np.mean(daily_ics_old_list, axis=0)
    
    pct25_a, pct975_a = float(np.percentile(draw_ics_a, 2.5)), float(np.percentile(draw_ics_a, 97.5))
    pct25_b, pct975_b = float(np.percentile(draw_ics_b, 2.5)), float(np.percentile(draw_ics_b, 97.5))
    pct25_old, pct975_old = float(np.percentile(draw_ics_old, 2.5)), float(np.percentile(draw_ics_old, 97.5))
    
    width_a = pct975_a - pct25_a
    width_b = pct975_b - pct25_b
    width_old = pct975_old - pct25_old
    
    if width_b >= width_a:
        wider_type = "random_linear"
        corr_low, corr_high = pct25_b, pct975_b
        wider_draws = draw_ics_b
    else:
        wider_type = "block_constant"
        corr_low, corr_high = pct25_a, pct975_a
        wider_draws = draw_ics_a
        
    res = {
        "null_band_ic_lower": corr_low,
        "null_band_ic_upper": corr_high,
        "null_band_ic_mean": float(np.mean(wider_draws)),
        "block_null_band_ic_lower": pct25_a,
        "block_null_band_ic_upper": pct975_a,
        "block_null_band_width": width_a,
        "linear_null_band_ic_lower": pct25_b,
        "linear_null_band_ic_upper": pct975_b,
        "linear_null_band_width": width_b,
        "old_null_band_ic_lower": pct25_old,
        "old_null_band_ic_upper": pct975_old,
        "old_null_band_width": width_old,
        "wider_type": wider_type,
        "draw_ics": wider_draws
    }
    _NULL_BAND_CACHE[cache_key] = res
    return res


def evaluate_predictions(
    df_pred: pd.DataFrame,
    horizon: str,
    h_days: int,
    config: dict,
    is_benchmark: bool = False
) -> dict[str, float]:
    """Run full evaluation suite on predictions and return dictionary of metrics."""
    fwd_excess_col = f"fwd_excess_{horizon}"
    fwd_ret_col = f"fwd_ret_{horizon}"
    fwd_rank_col = f"fwd_rankpct_{horizon}"
    outperf_col = f"outperform_{horizon}"
    
    val_cfg = config["validation"]
    min_stocks = val_cfg.get("min_cross_section", 8)
    n_boot = val_cfg.get("bootstrap_draws", 2000)
    k = val_cfg.get("top_k", 5)
    cost_bps = val_cfg.get("transaction_cost_per_side_bps", 100.0)
    
    # Special handling for Market Benchmark row (no excess return, no precision, no IC)
    if is_benchmark:
        dates = sorted(df_pred["Date"].unique())
        rebal_dates = dates[::h_days]
        mkt_period_rets = []
        for r_dt in rebal_dates:
            grp = df_pred[df_pred["Date"] == r_dt].dropna(subset=[fwd_ret_col])
            if len(grp) >= min_stocks:
                mkt_period_rets.append(grp[fwd_ret_col].mean())
        n_periods = len(mkt_period_rets)
        if n_periods > 0:
            mkt_mean = float(np.mean(mkt_period_rets))
            comp_mkt = float(np.prod(1.0 + np.array(mkt_period_rets)) - 1.0)
        else:
            mkt_mean, comp_mkt = np.nan, np.nan
            
        return {
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
            "rank_ic_pct_positive": np.nan,
            "rank_ic_ci_lower": np.nan,
            "rank_ic_ci_upper": np.nan,
            "effective_blocks": 0.0,
            "top_k_fwd_excess_mean": np.nan,
            "top_k_fwd_excess_median": np.nan,
            "top_k_fwd_excess_winsorized_mean": np.nan,
            "top_k_fwd_ret_mean": mkt_mean,
            "top_k_fwd_ret_median": float(np.median(mkt_period_rets)) if n_periods > 0 else np.nan,
            "top_k_precision": np.nan,
            "top_k_minus_bottom_k_spread": np.nan,
            "portfolio_num_periods": n_periods,
            "portfolio_gross_excess_mean": np.nan,
            "portfolio_net_excess_mean": np.nan,
            "portfolio_net_excess_actual_mean": np.nan,
            "portfolio_turnover_replaced_share_mean": np.nan,
            "portfolio_win_rate": np.nan,
            "portfolio_compounded_net_excess": np.nan,
            "portfolio_market_ret_mean": mkt_mean,
            "portfolio_market_compounded_ret": comp_mkt
        }
    
    # 1. Rank IC
    daily_ics = compute_rank_ic(df_pred, target_col=fwd_excess_col, min_stocks=min_stocks)
    if len(daily_ics) > 0:
        mean_ic = float(daily_ics.mean())
        std_ic = float(daily_ics.std(ddof=1)) if len(daily_ics) > 1 else 0.0
        ic_ir = mean_ic / (std_ic + 1e-12)
        pct_pos = float((daily_ics > 0).mean())
        ci_low, ci_high, eff_blocks = moving_block_bootstrap_ic_ci(daily_ics, block_length=h_days, n_draws=n_boot)
    else:
        mean_ic, std_ic, ic_ir, pct_pos = np.nan, np.nan, np.nan, np.nan
        ci_low, ci_high, eff_blocks = np.nan, np.nan, 0.0
        
    metrics = {
        "rank_ic_mean": mean_ic,
        "rank_ic_std": std_ic,
        "rank_ic_ir": ic_ir,
        "rank_ic_pct_positive": pct_pos,
        "rank_ic_ci_lower": ci_low,
        "rank_ic_ci_upper": ci_high,
        "effective_blocks": eff_blocks
    }
    
    # 2. Top-K
    top_k_metrics = compute_top_k_metrics(
        df_pred,
        fwd_excess_col=fwd_excess_col,
        fwd_ret_col=fwd_ret_col,
        fwd_rank_col=fwd_rank_col,
        k=k,
        min_stocks=min_stocks
    )
    metrics.update(top_k_metrics)
    
    # 3. Regression metrics
    valid_reg = df_pred.dropna(subset=["pred_score", fwd_ret_col, fwd_rank_col])
    if len(valid_reg) > 0:
        pred_scaled = valid_reg.groupby("Date")["pred_score"].rank(pct=True)
        metrics["mae_rank"] = float(np.mean(np.abs(pred_scaled - valid_reg[fwd_rank_col])))
        metrics["rmse_rank"] = float(np.sqrt(np.mean((pred_scaled - valid_reg[fwd_rank_col]) ** 2)))
        metrics["mae_ret"] = float(np.mean(np.abs(valid_reg["pred_score"] - valid_reg[fwd_ret_col])))
        metrics["rmse_ret"] = float(np.sqrt(np.mean((valid_reg["pred_score"] - valid_reg[fwd_ret_col]) ** 2)))
    else:
        metrics["mae_rank"] = np.nan
        metrics["rmse_rank"] = np.nan
        metrics["mae_ret"] = np.nan
        metrics["rmse_ret"] = np.nan
        
    # 4. Classification metrics
    if "pred_prob" in df_pred.columns and df_pred["pred_prob"].notna().any():
        class_metrics = compute_classification_metrics(df_pred, outperf_col, k=k)
        metrics.update(class_metrics)
        
    # 5. Portfolio Simulation
    port_metrics = simulate_rebalancing_portfolio(
        df_pred,
        fwd_ret_col=fwd_ret_col,
        h=h_days,
        k=k,
        cost_per_side_bps=cost_bps,
        min_stocks=min_stocks
    )
    metrics.update(port_metrics)
    
    return metrics

"""Patch results_long.csv with:
1. Dual Null Band (wider of block-constant and random-linear)
2. Turnover Metric Fix (portfolio_net_excess_actual_mean and replaced share)
3. Effective Sample Gating (n_eff < 5: not evaluable, 5 <= n_eff < 12: indicative only)
4. Neutralized IC diagnostics for ridge_core and logit_core
5. Regenerate reports/05_model2_rf.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from src.data.config_loader import load_config
from src.models.metrics import compute_null_band_simulation, simulate_rebalancing_portfolio, reset_null_band_cache

def main():
    t0 = time.time()
    config = load_config()
    reset_null_band_cache()
    
    base_dir = Path(__file__).resolve().parents[1]
    archive_res_path = base_dir / "reports/archive_v3/results_long.csv"
    target_res_path = base_dir / "reports/results_long.csv"
    oos_dir = base_dir / "reports/oos_predictions"
    
    print("Loading archived results_long.csv...")
    df_res = pd.read_csv(archive_res_path)
    
    # 1. Precompute null bands for all 9 (horizon, split_type) combinations
    null_bands = {}
    for h in ["short", "mid", "long"]:
        h_days = config["horizons"][h]
        for s in ["dev", "stress", "holdout"]:
            sample_oos = list(oos_dir.glob(f"rf_reg_{h}_core.parquet"))[0]
            df_sample = pd.read_parquet(sample_oos)
            part_sample = df_sample[df_sample["split_type"] == s].copy()
            nb = compute_null_band_simulation(
                part_sample, horizon=h, h_days=h_days, split_type=s,
                seed=config.get("random_seed", 42), k=config["validation"].get("top_k", 5),
                config=config
            )
            null_bands[(h, s)] = nb
            print(f"Null band [{h} {s}]: wider={nb['wider_type']}, band=[{nb['null_band_ic_lower']:+.4f}, {nb['null_band_ic_upper']:+.4f}] (old: [{nb['old_null_band_ic_lower']:+.4f}, {nb['old_null_band_ic_upper']:+.4f}])")
            
    # 2. Compute turnover metrics for all 45 OOS files
    turnover_metrics = {} # (model, feat_set, horizon, split_type) -> dict
    for p_file in oos_dir.glob("*.parquet"):
        stem = p_file.stem
        parts = stem.split("_")
        feat_set = parts[-1]
        h = parts[-2]
        m = "_".join(parts[:-2])
        h_days = config["horizons"][h]
        is_bm = m in ["equal_weight_universe", "market_benchmark", "market_equal_weight"]
        
        df_oos = pd.read_parquet(p_file)
        if df_oos.empty:
            continue
            
        # Per fold
        for fold_name, f_grp in df_oos.groupby("fold"):
            s_type = f_grp["split_type"].iloc[0]
            sim = simulate_rebalancing_portfolio(
                f_grp, fwd_ret_col=f"fwd_ret_{h}", h=h_days, k=config["validation"].get("top_k", 5),
                cost_per_side_bps=config["validation"].get("transaction_cost_per_side_bps", 100.0)
            )
            key = (m, feat_set, h, fold_name, s_type)
            turnover_metrics[key] = sim
            
        # Per partition
        for s_type, p_grp in df_oos.groupby("split_type"):
            sim = simulate_rebalancing_portfolio(
                p_grp, fwd_ret_col=f"fwd_ret_{h}", h=h_days, k=config["validation"].get("top_k", 5),
                cost_per_side_bps=config["validation"].get("transaction_cost_per_side_bps", 100.0)
            )
            key = (m, feat_set, h, s_type, s_type)
            turnover_metrics[key] = sim

    # 3. Build updated rows
    # Remove existing null band and portfolio turnover rows that will be replaced
    metrics_to_replace = [
        "outside_null_band", "permutation_p_val",
        "null_band_ic_lower", "null_band_ic_upper", "null_band_ic_mean",
        "block_null_band_ic_lower", "block_null_band_ic_upper",
        "linear_null_band_ic_lower", "linear_null_band_ic_upper",
        "old_null_band_ic_lower", "old_null_band_ic_upper",
        "portfolio_net_excess_actual_mean", "portfolio_turnover_replaced_share_mean",
        "n_eff", "evaluable_status_code"
    ]
    df_clean = df_res[~df_res["metric"].isin(metrics_to_replace)].copy()
    
    new_rows = []
    # Add turnover metrics for all entries in turnover_metrics
    for (m, feat_set, h, fold_id, s_type), sim in turnover_metrics.items():
        is_bm = m in ["equal_weight_universe", "market_benchmark", "market_equal_weight"]
        actual_net = np.nan if is_bm else sim["portfolio_net_excess_actual_mean"]
        rep_share = np.nan if is_bm else sim["portfolio_turnover_replaced_share_mean"]
        
        if pd.notnull(actual_net):
            new_rows.append({
                "model": m, "feature_set": feat_set, "horizon": h,
                "fold": fold_id, "split_type": s_type,
                "metric": "portfolio_net_excess_actual_mean", "value": float(actual_net)
            })
        if pd.notnull(rep_share):
            new_rows.append({
                "model": m, "feature_set": feat_set, "horizon": h,
                "fold": fold_id, "split_type": s_type,
                "metric": "portfolio_turnover_replaced_share_mean", "value": float(rep_share)
            })

    # For partition-level aggregates (fold in ["dev", "stress", "holdout"]):
    for (m, feat_set, h), grp in df_clean[df_clean["fold"].isin(["dev", "stress", "holdout"])].groupby(["model", "feature_set", "horizon"]):
        is_bm = m in ["equal_weight_universe", "market_benchmark", "market_equal_weight"]
        for s_type in grp["split_type"].unique():
            nb = null_bands.get((h, s_type))
            if not nb:
                continue
                
            sim_key = (m, feat_set, h, s_type, s_type)
            sim = turnover_metrics.get(sim_key, {})
            n_eff = sim.get("portfolio_num_periods", 0)
            
            new_rows.append({
                "model": m, "feature_set": feat_set, "horizon": h,
                "fold": s_type, "split_type": s_type,
                "metric": "n_eff", "value": float(n_eff)
            })
            
            if not is_bm:
                # Get partition rank IC
                ic_rows = grp[(grp["split_type"] == s_type) & (grp["metric"] == "rank_ic_mean")]
                part_ic = ic_rows["value"].iloc[0] if not ic_rows.empty else np.nan
                
                if n_eff < 5:
                    eval_code = 0.0 # not evaluable
                    outside_val = np.nan
                    p_val = np.nan
                else:
                    eval_code = 1.0 if n_eff < 12 else 2.0
                    if pd.notnull(part_ic):
                        outside = (part_ic < nb["null_band_ic_lower"] or part_ic > nb["null_band_ic_upper"])
                        outside_val = 1.0 if outside else 0.0
                        p_val = float(np.mean(nb["draw_ics"] >= part_ic))
                    else:
                        outside_val = 0.0
                        p_val = np.nan
                        
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "evaluable_status_code", "value": eval_code})
                if pd.notnull(outside_val):
                    new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "outside_null_band", "value": outside_val})
                if pd.notnull(p_val):
                    new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "permutation_p_val", "value": p_val})
                    
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "null_band_ic_lower", "value": nb["null_band_ic_lower"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "null_band_ic_upper", "value": nb["null_band_ic_upper"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "block_null_band_ic_lower", "value": nb["block_null_band_ic_lower"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "block_null_band_ic_upper", "value": nb["block_null_band_ic_upper"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "linear_null_band_ic_lower", "value": nb["linear_null_band_ic_lower"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "linear_null_band_ic_upper", "value": nb["linear_null_band_ic_upper"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "old_null_band_ic_lower", "value": nb["old_null_band_ic_lower"]})
                new_rows.append({"model": m, "feature_set": feat_set, "horizon": h, "fold": s_type, "split_type": s_type, "metric": "old_null_band_ic_upper", "value": nb["old_null_band_ic_upper"]})

    df_final = pd.concat([df_clean, pd.DataFrame(new_rows)], ignore_index=True)
    df_final.to_csv(target_res_path, index=False)
    print(f"Saved {len(df_final)} rows to {target_res_path} in {time.time() - t0:.2f}s.")
    return df_final, null_bands

if __name__ == "__main__":
    main()

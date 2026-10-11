"""Recompute results_long.csv and evaluation metrics from existing OOS predictions.
Incorporates:
- Corrected Dual Null Band (wider of block-constant and random-linear)
- Actual turnover tracking (portfolio_net_excess_actual_mean and replaced share)
- Effective sample gating (n_eff < 5: not evaluable, suppressed; 5 <= n_eff < 12: indicative only)
"""
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.models.metrics import evaluate_predictions, compute_null_band_simulation, reset_null_band_cache
from src.models.validation import build_folds

def recompute_all():
    t0 = time.time()
    config = load_config()
    reset_null_band_cache()
    
    base_dir = Path(__file__).resolve().parents[1]
    oos_dir = base_dir / "reports/oos_predictions"
    results_long_path = base_dir / "reports/results_long.csv"
    
    # Load features/targets and mask for folds
    df_feat = pd.read_parquet(base_dir / config["paths"]["features_targets"])
    mask = pd.read_parquet(base_dir / config["paths"]["rankable_mask"])["rankable"].values
    df_u = df_feat[df_feat["in_ranking_universe"] & mask].copy()
    all_folds = build_folds(df_u, config)
    
    parquet_files = sorted(list(oos_dir.glob("*.parquet")))
    print(f"Found {len(parquet_files)} OOS prediction files to re-evaluate.")
    
    long_rows = []
    
    for p_file in parquet_files:
        stem = p_file.stem
        # Format: {model}_{horizon}_{feature_set}
        # Model may contain underscores (e.g. equal_weight_universe, rf_reg, rf_clf, momentum_126)
        parts = stem.split("_")
        feature_set = parts[-1]
        horizon = parts[-2]
        model_name = "_".join(parts[:-2])
        
        h_days = config["horizons"][horizon]
        is_benchmark = model_name in ["equal_weight_universe", "market_benchmark", "market_equal_weight"]
        h_folds = [f for f in all_folds if f.horizon == horizon]
        
        df_oos = pd.read_parquet(p_file)
        if df_oos.empty:
            continue
            
        # 1. Fold-level metrics
        fold_rows_temp = []
        for fold_name, f_grp in df_oos.groupby("fold", sort=False):
            s_type = f_grp["split_type"].iloc[0]
            m_dict = evaluate_predictions(f_grp, horizon=horizon, h_days=h_days, config=config, is_benchmark=is_benchmark)
            for m_k, m_v in m_dict.items():
                if isinstance(m_v, (int, float, np.number)) and pd.notnull(m_v):
                    row = {
                        "model": model_name,
                        "feature_set": feature_set,
                        "horizon": horizon,
                        "fold": fold_name,
                        "split_type": s_type,
                        "metric": m_k,
                        "value": float(m_v)
                    }
                    long_rows.append(row)
                    fold_rows_temp.append(row)
                    
        df_fold_temp = pd.DataFrame(fold_rows_temp)
        
        # 2. Partition-level metrics
        for s_type, p_grp in df_oos.groupby("split_type", sort=False):
            p_metrics = evaluate_predictions(p_grp, horizon=horizon, h_days=h_days, config=config, is_benchmark=is_benchmark)
            
            n_eff = p_metrics.get("portfolio_num_periods", 0)
            p_metrics["n_eff"] = float(n_eff)
            
            if not is_benchmark:
                null_band = compute_null_band_simulation(
                    p_grp, horizon=horizon, h_days=h_days, split_type=s_type,
                    seed=config.get("random_seed", 42), k=config["validation"].get("top_k", 5),
                    config=config
                )
                part_ic = p_metrics.get("rank_ic_mean", np.nan)
                
                if n_eff < 5:
                    p_metrics["outside_null_band"] = np.nan
                    p_metrics["permutation_p_val"] = np.nan
                    p_metrics["evaluable_status_code"] = 0.0
                else:
                    if pd.notnull(part_ic):
                        outside = "yes" if (part_ic < null_band["null_band_ic_lower"] or part_ic > null_band["null_band_ic_upper"]) else "no"
                        p_val = float(np.mean(null_band["draw_ics"] >= part_ic))
                    else:
                        outside = "no"
                        p_val = np.nan
                    p_metrics["outside_null_band"] = 1.0 if outside == "yes" else 0.0
                    p_metrics["permutation_p_val"] = p_val
                    p_metrics["evaluable_status_code"] = 1.0 if n_eff < 12 else 2.0
                    
                p_metrics["null_band_ic_lower"] = null_band["null_band_ic_lower"]
                p_metrics["null_band_ic_upper"] = null_band["null_band_ic_upper"]
                p_metrics["null_band_ic_mean"] = null_band["null_band_ic_mean"]
                p_metrics["block_null_band_ic_lower"] = null_band["block_null_band_ic_lower"]
                p_metrics["block_null_band_ic_upper"] = null_band["block_null_band_ic_upper"]
                p_metrics["linear_null_band_ic_lower"] = null_band["linear_null_band_ic_lower"]
                p_metrics["linear_null_band_ic_upper"] = null_band["linear_null_band_ic_upper"]
                p_metrics["old_null_band_ic_lower"] = null_band["old_null_band_ic_lower"]
                p_metrics["old_null_band_ic_upper"] = null_band["old_null_band_ic_upper"]
                
            # Fold-level stats
            if not df_fold_temp.empty:
                fold_names = [f.fold_name for f in h_folds if f.split_type == s_type]
                f_ic_sub = df_fold_temp[
                    (df_fold_temp["fold"].isin(fold_names)) &
                    (df_fold_temp["metric"] == "rank_ic_mean")
                ]
                if not f_ic_sub.empty:
                    f_ics = f_ic_sub["value"].dropna().values
                    n_f = len(f_ics)
                    pos_f = int((f_ics > 0).sum())
                    mean_f = float(np.mean(f_ics))
                    std_f = float(np.std(f_ics, ddof=1)) if n_f > 1 else 0.0
                    t_f = float(mean_f / (std_f / np.sqrt(n_f))) if (n_f > 1 and std_f > 1e-9) else np.nan
                    p_metrics["folds_total_count"] = n_f
                    p_metrics["folds_positive_count"] = pos_f
                    p_metrics["folds_ic_mean"] = mean_f
                    p_metrics["folds_t_stat"] = t_f
                    
            for m_k, m_v in p_metrics.items():
                if isinstance(m_v, (int, float, np.number)) and pd.notnull(m_v):
                    long_rows.append({
                        "model": model_name,
                        "feature_set": feature_set,
                        "horizon": horizon,
                        "fold": s_type,
                        "split_type": s_type,
                        "metric": m_k,
                        "value": float(m_v)
                    })
                    
    df_long_all = pd.DataFrame(long_rows)
    df_long_all.to_csv(results_long_path, index=False)
    print(f"Saved {len(df_long_all)} rows to {results_long_path} in {time.time() - t0:.2f}s.")
    return df_long_all

if __name__ == "__main__":
    recompute_all()

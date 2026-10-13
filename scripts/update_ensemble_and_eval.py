"""Update ensemble definition (Part B) and evaluate ens_rank4 and ens_rank4_bestlgbm.

- ens_rank4: uniform across ALL horizons [ridge_core, rf_reg_core, lgbm_reg_core, momentum_126]
- ens_rank4_bestlgbm: sensitivity analysis using per-horizon best LightGBM on dev IC
"""
import sys
import copy
import json
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.models.metrics import evaluate_predictions, compute_null_band_simulation

base_dir = Path(__file__).resolve().parents[1]
config = load_config()
oos_dir = base_dir / "reports" / "oos_predictions"
results_path = base_dir / "reports" / "results_long.csv"

def compute_ensemble_predictions():
    horizons = ["short", "mid", "long"]
    results_df = pd.read_csv(results_path)

    # 1. Uniform components from config.yaml: ens_components
    # ens_components = ["ridge_core", "rf_reg_core", "lgbm_reg_core", "momentum_126"]
    ens_comps = config.get("ens_components", ["ridge_core", "rf_reg_core", "lgbm_reg_core", "momentum_126"])
    print(f"Pre-specified uniform ensemble components: {ens_comps}")

    all_new_results = []
    comparison_summary = {}

    for h in horizons:
        h_days = config["horizons"][h]

        # Load component predictions
        df_ridge = pd.read_parquet(oos_dir / f"ridge_{h}_core.parquet")[
            ["Date", "series_id", "pred_score", "fold", "split_type", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]
        ].rename(columns={"pred_score": "score_ridge"})

        df_rf = pd.read_parquet(oos_dir / f"rf_reg_{h}_core.parquet")[
            ["Date", "series_id", "pred_score"]
        ].rename(columns={"pred_score": "score_rf"})

        df_lgbm_reg = pd.read_parquet(oos_dir / f"lgbm_reg_{h}_core.parquet")[
            ["Date", "series_id", "pred_score"]
        ].rename(columns={"pred_score": "score_lgbm_reg"})

        df_lgbm_clf = pd.read_parquet(oos_dir / f"lgbm_clf_{h}_core.parquet")[
            ["Date", "series_id", "pred_score"]
        ].rename(columns={"pred_score": "score_lgbm_clf"})

        df_mom = pd.read_parquet(oos_dir / f"momentum_126_{h}_core.parquet")[
            ["Date", "series_id", "pred_score"]
        ].rename(columns={"pred_score": "score_mom"})

        # Merge common rankable set
        m = df_ridge.merge(df_rf, on=["Date", "series_id"]) \
                    .merge(df_lgbm_reg, on=["Date", "series_id"]) \
                    .merge(df_lgbm_clf, on=["Date", "series_id"]) \
                    .merge(df_mom, on=["Date", "series_id"])

        # Compute cross-sectional percentile ranks per date
        m["rank_ridge"] = m.groupby("Date")["score_ridge"].rank(pct=True)
        m["rank_rf"] = m.groupby("Date")["score_rf"].rank(pct=True)
        m["rank_lgbm_reg"] = m.groupby("Date")["score_lgbm_reg"].rank(pct=True)
        m["rank_lgbm_clf"] = m.groupby("Date")["score_lgbm_clf"].rank(pct=True)
        m["rank_mom"] = m.groupby("Date")["score_mom"].rank(pct=True)

        # -------------------------------------------------------------
        # Model 1: ens_rank4 (Uniform 4 components across all horizons)
        # Formula: (rank_ridge + rank_rf + rank_lgbm_reg + rank_mom) / 4.0
        # -------------------------------------------------------------
        m["score_ens_rank4"] = (m["rank_ridge"] + m["rank_rf"] + m["rank_lgbm_reg"] + m["rank_mom"]) / 4.0

        # -------------------------------------------------------------
        # Model 2: ens_rank4_bestlgbm (Sensitivity analysis only)
        # On Short: best LightGBM dev IC is lgbm_clf_core (+0.0043 vs -0.0048)
        # On Mid: best LightGBM dev IC is lgbm_reg_core (+0.0179 vs +0.0046)
        # On Long: best LightGBM dev IC is lgbm_clf_core (+0.0932 vs +0.0451)
        # -------------------------------------------------------------
        if h in ["short", "long"]:
            m["score_ens_bestlgbm"] = (m["rank_ridge"] + m["rank_rf"] + m["rank_lgbm_clf"] + m["rank_mom"]) / 4.0
        else:
            m["score_ens_bestlgbm"] = (m["rank_ridge"] + m["rank_rf"] + m["rank_lgbm_reg"] + m["rank_mom"]) / 4.0

        # Save OOS predictions
        for m_name, col_name in [("ens_rank4", "score_ens_rank4"), ("ens_rank4_bestlgbm", "score_ens_bestlgbm")]:
            df_oos = m[["series_id", "Date", "fold", "split_type", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]].copy()
            df_oos["pred_score"] = m[col_name].values
            df_oos["pred_prob"] = np.nan

            out_pq = oos_dir / f"{m_name}_{h}_core.parquet"
            df_oos.to_parquet(out_pq, index=False)
            print(f"Saved {m_name} {h} to {out_pq} ({len(df_oos)} rows)")

            # Evaluate fold metrics
            fold_long_rows = []
            for f_name, f_df in df_oos.groupby("fold"):
                s_type = f_df["split_type"].iloc[0]
                f_metrics = evaluate_predictions(f_df, horizon=h, h_days=h_days, config=config)
                for mk, mv in f_metrics.items():
                    if isinstance(mv, (int, float, np.number)):
                        fold_long_rows.append({
                            "model": m_name,
                            "feature_set": "core",
                            "horizon": h,
                            "fold": f_name,
                            "split_type": s_type,
                            "metric": mk,
                            "value": float(mv) if pd.notnull(mv) else np.nan
                        })

            df_fold_long = pd.DataFrame(fold_long_rows)
            all_new_results.extend(fold_long_rows)

            # Evaluate partition metrics (dev, stress, holdout)
            for s_type in df_oos["split_type"].unique():
                part_df = df_oos[df_oos["split_type"] == s_type].copy()
                part_metrics = evaluate_predictions(part_df, horizon=h, h_days=h_days, config=config)
                n_eff = part_metrics.get("portfolio_num_periods", 0)
                part_metrics["n_eff"] = float(n_eff)

                null_band = compute_null_band_simulation(
                    part_df, horizon=h, h_days=h_days, split_type=s_type,
                    seed=config.get("random_seed", 42), k=config["validation"].get("top_k", 5),
                    config=config
                )
                part_ic = part_metrics.get("rank_ic_mean", np.nan)

                if n_eff < 5:
                    part_metrics["outside_null_band"] = np.nan
                    part_metrics["permutation_p_val"] = np.nan
                    part_metrics["evaluable_status_code"] = 0.0
                else:
                    if pd.notnull(part_ic):
                        outside = "yes" if (part_ic < null_band["null_band_ic_lower"] or part_ic > null_band["null_band_ic_upper"]) else "no"
                        p_val = float(np.mean(null_band["draw_ics"] >= part_ic))
                    else:
                        outside = "no"
                        p_val = np.nan
                    part_metrics["outside_null_band"] = 1.0 if outside == "yes" else 0.0
                    part_metrics["permutation_p_val"] = p_val
                    part_metrics["evaluable_status_code"] = 1.0 if n_eff < 12 else 2.0

                part_metrics["null_band_ic_lower"] = null_band["null_band_ic_lower"]
                part_metrics["null_band_ic_upper"] = null_band["null_band_ic_upper"]
                part_metrics["null_band_ic_mean"] = null_band["null_band_ic_mean"]
                part_metrics["block_null_band_ic_lower"] = null_band["block_null_band_ic_lower"]
                part_metrics["block_null_band_ic_upper"] = null_band["block_null_band_ic_upper"]
                part_metrics["linear_null_band_ic_lower"] = null_band["linear_null_band_ic_lower"]
                part_metrics["linear_null_band_ic_upper"] = null_band["linear_null_band_ic_upper"]
                part_metrics["old_null_band_ic_lower"] = null_band["old_null_band_ic_lower"]
                part_metrics["old_null_band_ic_upper"] = null_band["old_null_band_ic_upper"]

                fold_ic_rows = df_fold_long[
                    (df_fold_long["split_type"] == s_type) &
                    (df_fold_long["metric"] == "rank_ic_mean")
                ]
                if not fold_ic_rows.empty:
                    f_ics = fold_ic_rows["value"].dropna().values
                    n_f = len(f_ics)
                    pos_f = int((f_ics > 0).sum())
                    mean_f = float(np.mean(f_ics))
                    std_f = float(np.std(f_ics, ddof=1)) if n_f > 1 else 0.0
                    t_stat_f = float(mean_f / (std_f / np.sqrt(n_f))) if (n_f > 1 and std_f > 1e-9) else np.nan
                    part_metrics["folds_total_count"] = n_f
                    part_metrics["folds_positive_count"] = pos_f
                    part_metrics["folds_ic_mean"] = mean_f
                    part_metrics["folds_t_stat"] = t_stat_f

                for mk, mv in part_metrics.items():
                    if isinstance(mv, (int, float, np.number)):
                        all_new_results.append({
                            "model": m_name,
                            "feature_set": "core",
                            "horizon": h,
                            "fold": s_type,
                            "split_type": s_type,
                            "metric": mk,
                            "value": float(mv) if pd.notnull(mv) else np.nan
                        })

    # Update results_long.csv
    df_new = pd.DataFrame(all_new_results)
    curr_results = pd.read_csv(results_path)
    # Remove old ens_rank4 and ens_rank4_bestlgbm rows
    curr_results = curr_results[~curr_results["model"].isin(["ens_rank4", "ens_rank4_bestlgbm"])]
    merged_results = pd.concat([curr_results, df_new], ignore_index=True)
    merged_results.to_csv(results_path, index=False)
    print(f"Updated {results_path} with both ens_rank4 and ens_rank4_bestlgbm metrics ({len(merged_results)} rows).")

if __name__ == "__main__":
    compute_ensemble_predictions()

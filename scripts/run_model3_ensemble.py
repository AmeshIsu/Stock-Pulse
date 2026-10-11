"""Build and evaluate Model 3 Ensemble Preview: ens_rank4 across all horizons."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.metrics import evaluate_predictions, compute_null_band_simulation

def run_ensemble():
    config = load_config()
    horizons = list(config["horizons"].keys())
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    results_path = base_dir / config["paths"]["results_long_csv"]
    
    print("=== BUILDING ENSEMBLE PREVIEW: ens_rank4 ===")
    
    # Check results_long for best LightGBM variant on DEV
    df_results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    
    ensemble_results_rows = []
    
    for h in horizons:
        h_days = config["horizons"][h]
        # 1. Determine best LightGBM variant on dev for this horizon
        lgbm_dev = df_results[
            (df_results["horizon"] == h) &
            (df_results["model"].str.startswith("lgbm_")) &
            (df_results["fold"] == "dev") &
            (df_results["metric"] == "rank_ic_mean")
        ].copy()
        
        best_lgbm_model = "lgbm_reg"
        best_lgbm_fset = "core"
        if not lgbm_dev.empty:
            best_row = lgbm_dev.sort_values("value", ascending=False).iloc[0]
            best_lgbm_model = best_row["model"]
            best_lgbm_fset = best_row["feature_set"]
            print(f"Horizon {h}: Best LightGBM variant on Dev is {best_lgbm_model}_{best_lgbm_fset} (IC = {best_row['value']:.4f})")
            
        # 2. Load 4 models
        ridge_file = oos_dir / f"ridge_{h}_core.parquet"
        rf_file = oos_dir / f"rf_reg_{h}_core.parquet"
        lgbm_file = oos_dir / f"{best_lgbm_model}_{h}_{best_lgbm_fset}.parquet"
        mom_file = oos_dir / f"momentum_126_{h}_core.parquet"
        
        if not (ridge_file.exists() and rf_file.exists() and lgbm_file.exists() and mom_file.exists()):
            print(f"Skipping {h}: Missing one of the required OOS prediction files.")
            continue
            
        df_ridge = pd.read_parquet(ridge_file)[["Date", "series_id", "pred_score", "fold", "split_type", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]].rename(columns={"pred_score": "score_ridge"})
        df_rf = pd.read_parquet(rf_file)[["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_rf"})
        df_lgbm = pd.read_parquet(lgbm_file)[["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_lgbm"})
        df_mom = pd.read_parquet(mom_file)[["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_mom"})
        
        merged = df_ridge.merge(df_rf, on=["Date", "series_id"]).merge(df_lgbm, on=["Date", "series_id"]).merge(df_mom, on=["Date", "series_id"])
        
        # 3. Cross-sectional rank percentiles per date
        merged["rank_ridge"] = merged.groupby("Date")["score_ridge"].rank(pct=True)
        merged["rank_rf"] = merged.groupby("Date")["score_rf"].rank(pct=True)
        merged["rank_lgbm"] = merged.groupby("Date")["score_lgbm"].rank(pct=True)
        merged["rank_mom"] = merged.groupby("Date")["score_mom"].rank(pct=True)
        
        # Equal weights average rank
        merged["pred_score"] = (merged["rank_ridge"] + merged["rank_rf"] + merged["rank_lgbm"] + merged["rank_mom"]) / 4.0
        merged["pred_prob"] = np.nan
        
        ens_oos = merged[["series_id", "Date", "fold", "split_type", "pred_score", "pred_prob", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]].copy()
        
        # Save OOS predictions
        ens_oos_file = oos_dir / f"ens_rank4_{h}_core.parquet"
        ens_oos.to_parquet(ens_oos_file, index=False)
        print(f"Saved ens_rank4 OOS predictions to: {ens_oos_file} ({len(ens_oos)} rows)")
        
        # 4. Evaluate fold and partition metrics
        fold_long_rows = []
        for f_name, f_df in ens_oos.groupby("fold"):
            s_type = f_df["split_type"].iloc[0]
            f_metrics = evaluate_predictions(f_df, horizon=h, h_days=h_days, config=config)
            for m_k, m_v in f_metrics.items():
                if isinstance(m_v, (int, float, np.number)):
                    fold_long_rows.append({
                        "model": "ens_rank4",
                        "feature_set": "core",
                        "horizon": h,
                        "fold": f_name,
                        "split_type": s_type,
                        "metric": m_k,
                        "value": float(m_v) if pd.notnull(m_v) else np.nan
                    })
                    
        df_fold_long = pd.DataFrame(fold_long_rows)
        ensemble_results_rows.extend(fold_long_rows)
        
        # Partition aggregates
        for s_type in ens_oos["split_type"].unique():
            part_df = ens_oos[ens_oos["split_type"] == s_type].copy()
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
                
            for m_k, m_v in part_metrics.items():
                if isinstance(m_v, (int, float, np.number)):
                    ensemble_results_rows.append({
                        "model": "ens_rank4",
                        "feature_set": "core",
                        "horizon": h,
                        "fold": s_type,
                        "split_type": s_type,
                        "metric": m_k,
                        "value": float(m_v) if pd.notnull(m_v) else np.nan
                    })
                    
    # Save ensemble metrics to results_long.csv
    if ensemble_results_rows:
        df_ens_long = pd.DataFrame(ensemble_results_rows)
        if results_path.exists():
            df_curr = pd.read_csv(results_path)
            df_curr = df_curr[df_curr["model"] != "ens_rank4"]
            df_merged = pd.concat([df_curr, df_ens_long], ignore_index=True)
        else:
            df_merged = df_ens_long
        df_merged.to_csv(results_path, index=False)
        print(f"Updated results_long.csv with ens_rank4 metrics ({len(df_merged)} rows).")

if __name__ == "__main__":
    run_ensemble()

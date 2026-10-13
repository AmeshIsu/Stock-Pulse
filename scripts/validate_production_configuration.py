"""scripts/validate_production_configuration.py

Part 0.2: Production configuration walk-forward validation.
Runs walk-forward of the PRODUCTION configuration for ridge_core, rf_reg_core, and lgbm_reg_core
with fixed consensus hyperparameters from models/production/hyperparameters.json (no inner tuning).
Writes to reports/oos_predictions_prodcfg/ (frozen files untouched).
Builds ens_rank4_prodcfg with tie-handling (method='average') and evaluates metrics.
Updates data/app/model_registry.json with 'production_config_metrics'.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.production.train import train_model_on_window
from src.models.preprocessing import transform_preprocessor, apply_cross_sectional_rank
from src.models.metrics import (
    compute_rank_ic,
    moving_block_bootstrap_ic_ci,
    compute_top_k_metrics,
    compute_null_band_simulation
)

PRODCFG_DIR = base_dir / "reports" / "oos_predictions_prodcfg"
PRODCFG_DIR.mkdir(parents=True, exist_ok=True)

def run_prodcfg_walkforward():
    config = load_config()
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    hp_path = base_dir / "models" / "production" / "hyperparameters.json"
    
    with open(hp_path, "r", encoding="utf-8") as f:
        consensus_hp = json.load(f)["models"]
        
    df_feat = pd.read_parquet(features_path)
    df_mask = pd.read_parquet(mask_path)
    df_u = df_feat[df_feat["in_ranking_universe"] & df_mask["rankable"].values].copy()
    
    folds = build_folds(df_u, config)
    horizons = ["short", "mid", "long"]
    models_to_train = ["ridge_core", "rf_reg_core", "lgbm_reg_core"]
    feature_cols = list(config["feature_sets"]["core"])
    
    predictions = {m: {h: [] for h in horizons} for m in models_to_train}
    
    print("=== Running Production Configuration Walk-Forward ===", flush=True)
    for idx, fold in enumerate(folds):
        h = fold.horizon
        train_df = df_u.loc[fold.train_indices].copy()
        test_df = df_u.loc[fold.test_indices].copy()
        target_col = f"fwd_rankpct_{h}"
        print(f"[{idx+1}/{len(folds)}] Training fold {fold.fold_name} ({h})...", flush=True)
        
        for m_id in models_to_train:
            hp = consensus_hp[m_id][h]
            model, prep = train_model_on_window(
                m_id, h, train_df, target_col, feature_cols, hp
            )
            
            # Predict test_df using production code path
            if m_id == "ridge_core":
                tr_test, _ = transform_preprocessor(test_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="neutral")
                tr_test = apply_cross_sectional_rank(tr_test, prep["cand_cols"])
                preds = model.predict(tr_test[prep["active_cols"]].values)
            elif m_id == "rf_reg_core":
                tr_test, _ = transform_preprocessor(test_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="median")
                preds = model.predict(tr_test[prep["active_cols"]].values)
            elif m_id == "lgbm_reg_core":
                tr_test, _ = transform_preprocessor(test_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="none")
                preds = model.predict(tr_test[prep["active_cols"]].values)
            else:
                raise ValueError(m_id)
                
            pred_df = test_df[["series_id", "Date"]].copy()
            pred_df["fold"] = fold.fold_name
            pred_df["split_type"] = fold.split_type
            pred_df["pred_score"] = preds
            pred_df["pred_prob"] = np.nan
            pred_df[f"fwd_ret_{h}"] = test_df[f"fwd_ret_{h}"].values
            pred_df[f"fwd_excess_{h}"] = test_df[f"fwd_excess_{h}"].values
            pred_df[f"fwd_rankpct_{h}"] = test_df[f"fwd_rankpct_{h}"].values
            pred_df[f"outperform_{h}"] = test_df[f"outperform_{h}"].values
            predictions[m_id][h].append(pred_df)
            
    # Save OOS parquets to reports/oos_predictions_prodcfg/
    combined_dfs = {m: {} for m in models_to_train}
    for m_id in models_to_train:
        for h in horizons:
            comb = pd.concat(predictions[m_id][h], ignore_index=True)
            comb = comb.sort_values(["Date", "series_id"]).reset_index(drop=True)
            out_pq = PRODCFG_DIR / f"{m_id}_{h}_core.parquet"
            comb.to_parquet(out_pq, index=False)
            combined_dfs[m_id][h] = comb
            print(f"Saved {out_pq} ({len(comb)} rows)")
            
    # Also copy / load momentum_126 (identical benchmark) into prodcfg folder
    mom_dfs = {}
    for h in horizons:
        orig_mom = pd.read_parquet(base_dir / "reports" / "oos_predictions" / f"momentum_126_{h}_core.parquet")
        out_pq = PRODCFG_DIR / f"momentum_126_{h}_core.parquet"
        orig_mom.to_parquet(out_pq, index=False)
        mom_dfs[h] = orig_mom
        print(f"Saved {out_pq} ({len(orig_mom)} rows)")
        
    # Build ens_rank4_prodcfg using average rank for ties
    print("=== Building ens_rank4_prodcfg ===")
    ens_dfs = {}
    for h in horizons:
        df_ridge = combined_dfs["ridge_core"][h][["Date", "series_id", "pred_score", "fold", "split_type", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]].rename(columns={"pred_score": "score_ridge"})
        df_rf = combined_dfs["rf_reg_core"][h][["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_rf"})
        df_lgbm = combined_dfs["lgbm_reg_core"][h][["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_lgbm"})
        df_mom = mom_dfs[h][["Date", "series_id", "pred_score"]].rename(columns={"pred_score": "score_mom"})
        
        m = df_ridge.merge(df_rf, on=["Date", "series_id"]).merge(df_lgbm, on=["Date", "series_id"]).merge(df_mom, on=["Date", "series_id"])
        
        # Tie handling: use method='average'
        m["rank_ridge"] = m.groupby("Date")["score_ridge"].rank(pct=True, method="average")
        m["rank_rf"] = m.groupby("Date")["score_rf"].rank(pct=True, method="average")
        m["rank_lgbm"] = m.groupby("Date")["score_lgbm"].rank(pct=True, method="average")
        m["rank_mom"] = m.groupby("Date")["score_mom"].rank(pct=True, method="average")
        
        m["score_ens_rank4"] = (m["rank_ridge"] + m["rank_rf"] + m["rank_lgbm"] + m["rank_mom"]) / 4.0
        
        df_ens = m[["series_id", "Date", "fold", "split_type", f"fwd_ret_{h}", f"fwd_excess_{h}", f"fwd_rankpct_{h}", f"outperform_{h}"]].copy()
        df_ens["pred_score"] = m["score_ens_rank4"].values
        df_ens["pred_prob"] = np.nan
        
        out_pq = PRODCFG_DIR / f"ens_rank4_{h}_core.parquet"
        df_ens.to_parquet(out_pq, index=False)
        ens_dfs[h] = df_ens
        print(f"Saved {out_pq} ({len(df_ens)} rows)")
        
    # Evaluate all models in prodcfg
    all_models = {
        "ridge_core": combined_dfs["ridge_core"],
        "rf_reg_core": combined_dfs["rf_reg_core"],
        "lgbm_reg_core": combined_dfs["lgbm_reg_core"],
        "momentum_126": mom_dfs,
        "ens_rank4": ens_dfs
    }
    
    # Load model_registry.json to compare and update
    reg_path = base_dir / "data" / "app" / "model_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    print("\n==========================================================================================================")
    print(" PRODUCTION CONFIGURATION METRICS EVALUATION & COMPARISON")
    print("==========================================================================================================")
    
    comparison_table = []
    
    for m_id, h_dict in all_models.items():
        for h in horizons:
            df_pred = h_dict[h]
            target_col = f"fwd_excess_{h}"
            fwd_ret_col = f"fwd_ret_{h}"
            fwd_rank_col = f"fwd_rankpct_{h}"
            h_days = config["horizons"][h]
            
            # Dev fold evaluations
            dev_df = df_pred[df_pred["split_type"] == "dev"]
            dev_folds = sorted(dev_df["fold"].unique())
            n_folds = len(dev_folds)
            
            fold_ics = []
            for f_name in dev_folds:
                sub_f = dev_df[dev_df["fold"] == f_name]
                f_ic = compute_rank_ic(sub_f, target_col=target_col, min_stocks=8)
                if len(f_ic) > 0:
                    fold_ics.append(float(f_ic.mean()))
                    
            dev_mean_ic = float(np.mean(fold_ics)) if fold_ics else np.nan
            fold_std = float(np.std(fold_ics, ddof=1)) if len(fold_ics) > 1 else np.nan
            fold_t_stat = (dev_mean_ic / (fold_std / np.sqrt(n_folds))) if (fold_std and fold_std > 1e-9) else 0.0
            pos_folds = sum(1 for x in fold_ics if x > 0)
            
            # Top-tier hit rate on dev
            top_k_res = compute_top_k_metrics(
                dev_df,
                fwd_excess_col=target_col,
                fwd_ret_col=fwd_ret_col,
                fwd_rank_col=fwd_rank_col,
                k=5,
                min_stocks=8
            )
            top_tier_hit_rate = top_k_res.get("top_k_precision", np.nan)
            
            # Holdout fold evaluation
            hold_df = df_pred[df_pred["split_type"] == "holdout"]
            if not hold_df.empty:
                hold_ics = compute_rank_ic(hold_df, target_col=target_col, min_stocks=8)
                holdout_ic = float(hold_ics.mean()) if len(hold_ics) > 0 else np.nan
                n_hold_dates = hold_df["Date"].nunique()
                n_eff = float(n_hold_dates / h_days)
            else:
                holdout_ic = np.nan
                n_eff = 0.0
                
            holdout_status = "indicative" if n_eff >= 5 else f"not evaluable (n_eff < 5)"
            
            prod_metrics = {
                "dev_mean_ic": round(dev_mean_ic, 4),
                "fold_t_stat": round(fold_t_stat, 4),
                "folds_positive_count": pos_folds,
                "folds_total_count": n_folds,
                "folds_positive_ratio": f"{pos_folds}/{n_folds}",
                "top_tier_hit_rate": round(top_tier_hit_rate, 4) if pd.notnull(top_tier_hit_rate) else None,
                "holdout_ic": round(holdout_ic, 4) if pd.notnull(holdout_ic) else None,
                "holdout_n_eff": int(round(n_eff)),
                "holdout_status": holdout_status
            }
            
            # Compare with validated metrics from registry
            val_h = registry.get(m_id, {}).get("horizons", {}).get(h, {})
            val_ic = val_h.get("dev_mean_ic", np.nan)
            val_t = val_h.get("fold_t_stat", np.nan)
            val_pos = val_h.get("folds_positive_count", 0)
            val_tot = val_h.get("folds_total_count", n_folds)
            val_hit = val_h.get("top_5_precision", np.nan)
            
            diff_ic = dev_mean_ic - val_ic if pd.notnull(val_ic) else np.nan
            diff_t = fold_t_stat - val_t if pd.notnull(val_t) else np.nan
            diff_pos = pos_folds - val_pos
            diff_hit = top_tier_hit_rate - val_hit if (pd.notnull(top_tier_hit_rate) and pd.notnull(val_hit)) else np.nan
            
            prod_metrics["difference_vs_validated"] = {
                "diff_dev_mean_ic": round(diff_ic, 4),
                "diff_fold_t_stat": round(diff_t, 4),
                "diff_folds_positive_count": diff_pos,
                "diff_top_tier_hit_rate": round(diff_hit, 4) if pd.notnull(diff_hit) else None
            }
            
            # Store in registry
            if m_id in registry and "horizons" in registry[m_id] and h in registry[m_id]["horizons"]:
                registry[m_id]["horizons"][h]["production_config_metrics"] = prod_metrics
                
            comparison_table.append({
                "model_id": m_id,
                "horizon": h,
                "val_mean_ic": val_ic,
                "prod_mean_ic": round(dev_mean_ic, 4),
                "diff_ic": round(diff_ic, 4),
                "val_t_stat": val_t,
                "prod_t_stat": round(fold_t_stat, 4),
                "diff_t": round(diff_t, 4),
                "val_folds_pos": f"{val_pos}/{val_tot}",
                "prod_folds_pos": f"{pos_folds}/{n_folds}",
                "val_hit_rate": val_hit,
                "prod_hit_rate": round(top_tier_hit_rate, 4) if pd.notnull(top_tier_hit_rate) else np.nan,
                "diff_hit": round(diff_hit, 4) if pd.notnull(diff_hit) else np.nan
            })

    # Save updated model_registry.json
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved updated {reg_path}")
    
    df_comp = pd.DataFrame(comparison_table)
    print("\nComparison Table (Validated vs Production Config):")
    print(df_comp.to_string(index=False))
    
    # Check Auto rule implication under production config
    # Auto rule: higher dev fold t-stat between ens_rank4 and momentum_126 (margin > 0.10)
    print("\n=== Auto Choice Comparison (Information Only) ===")
    for h in horizons:
        # Validated
        v_ens_t = registry["ens_rank4"]["horizons"][h]["fold_t_stat"]
        v_mom_t = registry["momentum_126"]["horizons"][h]["fold_t_stat"]
        v_choice = "ens_rank4" if v_ens_t > v_mom_t + 0.10 else ("momentum_126" if v_mom_t > v_ens_t + 0.10 else "ens_rank4")
        
        # Production config
        p_ens_t = registry["ens_rank4"]["horizons"][h]["production_config_metrics"]["fold_t_stat"]
        p_mom_t = registry["momentum_126"]["horizons"][h]["production_config_metrics"]["fold_t_stat"]
        p_choice = "ens_rank4" if p_ens_t > p_mom_t + 0.10 else ("momentum_126" if p_mom_t > p_ens_t + 0.10 else "ens_rank4")
        
        status = "UNCHANGED" if v_choice == p_choice else f"WOULD CHANGE to {p_choice}"
        print(f"Horizon {h.upper()}:")
        print(f"  Validated: ens_rank4 t={v_ens_t:.3f}, mom_126 t={v_mom_t:.3f} -> Chosen: {v_choice}")
        print(f"  Production Config: ens_rank4 t={p_ens_t:.3f}, mom_126 t={p_mom_t:.3f} -> Choice: {p_choice} ({status})")
        
    # Check ens_rank4 material weakness: > 0.02 IC drop
    print("\n=== ens_rank4 Material Weakness Assessment ===")
    for h in horizons:
        diff_ic = registry["ens_rank4"]["horizons"][h]["production_config_metrics"]["difference_vs_validated"]["diff_dev_mean_ic"]
        material = "YES (materially weaker, IC drop > 0.02)" if diff_ic < -0.02 else "NO (within 0.02 IC)"
        print(f"  Horizon {h}: IC diff = {diff_ic:+.4f} -> Materially weaker? {material}")

    return df_comp

if __name__ == "__main__":
    run_prodcfg_walkforward()

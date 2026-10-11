"""Compute permutation importance and TreeSHAP values for Model 3 (LightGBM)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.metrics import compute_rank_ic
from src.models.validation import build_folds
from src.models.explain import FEATURE_FAMILIES, FEATURE_LABELS, STOCK_IDENTITY_PROXIES

def compute_lgbm_permutation_importance(config: dict, n_repeats: int = 2) -> dict[str, pd.DataFrame]:
    """Compute permutation importance across DEV test folds for LightGBM core using tuned hyperparameters."""
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    folds = build_folds(df_u, config)
    hp_path = base_dir / "reports/06_lgbm_hyperparams.csv"
    hp_df = pd.read_csv(hp_path) if hp_path.exists() else pd.DataFrame()
    
    results = {}
    core_feats = list(config["feature_sets"]["core"])
    
    for h in ["short", "mid", "long"]:
        dev_folds = [f for f in folds if f.horizon == h and f.split_type == "dev"]
        fold_importances = []
        
        for fold in dev_folds:
            train_df = df_u.loc[fold.train_indices].copy()
            test_df = df_u.loc[fold.test_indices].copy()
            
            sub_hp = hp_df[(hp_df["model"] == "lgbm_reg") & (hp_df["horizon"] == h) & (hp_df["fold"] == fold.fold_name)]
            if not sub_hp.empty:
                r = sub_hp.iloc[0]
                best_iter = max(1, int(r["best_iteration"]))
                params = {
                    "objective": "regression",
                    "learning_rate": 0.03,
                    "num_leaves": int(r["num_leaves"]),
                    "min_child_samples": int(r["min_child_samples"]),
                    "feature_fraction": float(r["feature_fraction"]),
                    "lambda_l2": float(r["lambda_l2"]),
                    "bagging_fraction": 0.3,
                    "bagging_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "seed": 42,
                    "n_jobs": 2,
                    "verbosity": -1
                }
            else:
                best_iter = 50
                params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 8, "seed": 42, "verbosity": -1}
                
            valid_train = train_df[f"fwd_rankpct_{h}"].notna()
            ds_tr = lgb.Dataset(train_df.loc[valid_train, core_feats].values, label=train_df.loc[valid_train, f"fwd_rankpct_{h}"].values, free_raw_data=False)
            booster = lgb.train(params, ds_tr, num_boost_round=best_iter)
            
            # Base test predictions and IC
            test_eval = test_df.copy()
            test_eval["pred_score"] = booster.predict(test_df[core_feats].values)
            base_ics = compute_rank_ic(test_eval, target_col=f"fwd_excess_{h}", min_stocks=8)
            base_mean_ic = float(base_ics.mean()) if len(base_ics) > 0 else 0.0
            
            # Permutation importance per feature
            rng = np.random.RandomState(42)
            for feat in core_feats:
                diffs = []
                for rep in range(n_repeats):
                    perm_test = test_df.copy()
                    perm_test[feat] = rng.permutation(perm_test[feat].values)
                    perm_test["pred_score"] = booster.predict(perm_test[core_feats].values)
                    shuff_ics = compute_rank_ic(perm_test, target_col=f"fwd_excess_{h}", min_stocks=8)
                    shuff_mean_ic = float(shuff_ics.mean()) if len(shuff_ics) > 0 else 0.0
                    diffs.append(base_mean_ic - shuff_mean_ic)
                
                fold_importances.append({
                    "horizon": h,
                    "fold": fold.fold_name,
                    "feature": feat,
                    "importance": float(np.mean(diffs)),
                    "family": FEATURE_FAMILIES.get(feat, "Other"),
                    "is_identity_proxy": feat in STOCK_IDENTITY_PROXIES
                })
                
        df_imp = pd.DataFrame(fold_importances)
        results[h] = df_imp
        print(f"[{h}] Computed permutation importance across {len(dev_folds)} dev folds.", flush=True)
        
    return results


def compute_lgbm_shap_values(config: dict, sample_rows_per_fold: int = 2000) -> dict[str, pd.DataFrame]:
    """Compute mean absolute TreeSHAP values for best LightGBM variant across DEV test folds."""
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    folds = build_folds(df_u, config)
    hp_path = base_dir / "reports/06_lgbm_hyperparams.csv"
    hp_df = pd.read_csv(hp_path) if hp_path.exists() else pd.DataFrame()
    
    shap_results = {}
    core_feats = list(config["feature_sets"]["core"])
    
    for h in ["short", "mid", "long"]:
        dev_folds = [f for f in folds if f.horizon == h and f.split_type == "dev"]
        fold_shaps = []
        
        for fold in dev_folds:
            train_df = df_u.loc[fold.train_indices].copy()
            test_df = df_u.loc[fold.test_indices].copy()
            
            sub_hp = hp_df[(hp_df["model"] == "lgbm_reg") & (hp_df["horizon"] == h) & (hp_df["fold"] == fold.fold_name)]
            if not sub_hp.empty:
                r = sub_hp.iloc[0]
                best_iter = max(1, int(r["best_iteration"]))
                params = {
                    "objective": "regression",
                    "learning_rate": 0.03,
                    "num_leaves": int(r["num_leaves"]),
                    "min_child_samples": int(r["min_child_samples"]),
                    "feature_fraction": float(r["feature_fraction"]),
                    "lambda_l2": float(r["lambda_l2"]),
                    "bagging_fraction": 0.3,
                    "bagging_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "seed": 42,
                    "n_jobs": 2,
                    "verbosity": -1
                }
            else:
                best_iter = 50
                params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 8, "seed": 42, "verbosity": -1}
                
            valid_train = train_df[f"fwd_rankpct_{h}"].notna()
            ds_tr = lgb.Dataset(train_df.loc[valid_train, core_feats].values, label=train_df.loc[valid_train, f"fwd_rankpct_{h}"].values, free_raw_data=False)
            booster = lgb.train(params, ds_tr, num_boost_round=best_iter)
            
            # Sample test rows
            if len(test_df) > sample_rows_per_fold:
                sample_test = test_df.sample(n=sample_rows_per_fold, random_state=42)
            else:
                sample_test = test_df
                
            X_test_sample = sample_test[core_feats].values
            # Fast TreeSHAP using LightGBM booster
            contribs = booster.predict(X_test_sample, pred_contrib=True)
            shap_matrix = contribs[:, :-1]  # drop bias column
            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
            
            for i, feat in enumerate(core_feats):
                fold_shaps.append({
                    "horizon": h,
                    "fold": fold.fold_name,
                    "feature": feat,
                    "mean_abs_shap": float(mean_abs_shap[i]),
                    "family": FEATURE_FAMILIES.get(feat, "Other"),
                    "is_identity_proxy": feat in STOCK_IDENTITY_PROXIES
                })
                
        df_shap = pd.DataFrame(fold_shaps)
        shap_results[h] = df_shap
        print(f"[{h}] Computed TreeSHAP across {len(dev_folds)} dev folds.", flush=True)
        
    return shap_results


def plot_explainability_charts(perm_results: dict[str, pd.DataFrame], shap_results: dict[str, pd.DataFrame]):
    """Generate professional figures for SHAP values and feature family importance."""
    fig_dir = base_dir / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {
        "Momentum & Reversal": "#1f77b4",
        "Volatility & Risk": "#ff7f0e",
        "Liquidity & Turnover": "#2ca02c",
        "Trend & Technical": "#9467bd",
        "Market & Benchmark": "#8c564b",
        "Other": "#7f7f7f"
    }
    
    # 1. SHAP Importance Bar Chart (Mid Horizon)
    mid_shap = shap_results.get("mid")
    if mid_shap is not None and not mid_shap.empty:
        agg = mid_shap.groupby(["feature", "family", "is_identity_proxy"])["mean_abs_shap"].mean().reset_index()
        agg = agg.sort_values("mean_abs_shap", ascending=False).head(15)
        agg["label"] = agg["feature"].apply(lambda f: f"{FEATURE_LABELS.get(f, f)} {'[PROXY]' if f in STOCK_IDENTITY_PROXIES else ''}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=agg, x="mean_abs_shap", y="label", hue="family", dodge=False, palette=palette)
        plt.title("LightGBM Feature Importance (Mean |SHAP| Value) — Mid Horizon (63d)", fontsize=13, fontweight="bold")
        plt.xlabel("Mean Absolute SHAP Value (Impact on Model Upside Score)")
        plt.ylabel("Feature")
        plt.tight_layout()
        shap_fig_path = fig_dir / "06_lgbm_shap_importance.png"
        plt.savefig(shap_fig_path, dpi=300)
        plt.close()
        print(f"Saved SHAP importance chart to: {shap_fig_path}", flush=True)


if __name__ == "__main__":
    cfg = load_config()
    print("Computing LightGBM permutation importance across DEV test folds...", flush=True)
    perm_res = compute_lgbm_permutation_importance(cfg, n_repeats=2)
    
    perm_dfs = [df for df in perm_res.values() if not df.empty]
    if perm_dfs:
        perm_combined = pd.concat(perm_dfs, ignore_index=True)
        perm_path = base_dir / "reports/06_lgbm_permutation_importance.csv"
        perm_combined.to_csv(perm_path, index=False)
        print(f"Saved permutation importance to: {perm_path}", flush=True)
        
    print("Computing LightGBM TreeSHAP values across DEV test folds...", flush=True)
    shap_res = compute_lgbm_shap_values(cfg, sample_rows_per_fold=2000)
    
    shap_dfs = [df for df in shap_res.values() if not df.empty]
    if shap_dfs:
        shap_combined = pd.concat(shap_dfs, ignore_index=True)
        shap_path = base_dir / "reports/06_lgbm_shap_values.csv"
        shap_combined.to_csv(shap_path, index=False)
        print(f"Saved SHAP values to: {shap_path}", flush=True)
        
    plot_explainability_charts(perm_res, shap_res)
    print("Explainability analysis complete.", flush=True)

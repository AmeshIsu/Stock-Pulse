"""Hyperparameter extraction for LightGBM models by executing inner 80/20 IC early stopping."""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.models.boosting_models import get_12_sampled_configs, get_lgbm_specs

def extract_for_horizon(h: str):
    config = load_config()
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    folds = build_folds(df_u, config)
    specs = get_lgbm_specs(config)
    sampled_configs = get_12_sampled_configs(seed=42)
    
    hp_file = base_dir / f"reports/06_lgbm_hyperparams_{h}.csv"
    existing_records = []
    done_keys = set()
    if hp_file.exists():
        existing_df = pd.read_csv(hp_file)
        existing_records = existing_df.to_dict("records")
        for r in existing_records:
            done_keys.add((r["model"], r["feature_set"]))
            
    records = list(existing_records)
    t_start = time.time()
    
    print(f"=== EXTRACTING LIGHTGBM HYPERPARAMETERS FOR HORIZON: {h.upper()} ===", flush=True)
    
    h_days = config["horizons"][h]
    h_folds = [f for f in folds if f.horizon == h]
    target_fwd = f"fwd_excess_{h}"
    
    for spec, f_set in specs:
        if (spec.name, f_set) in done_keys:
            print(f"[{h}] [SKIP] Already extracted {spec.name}_{f_set}", flush=True)
            continue
            
        t_spec = time.time()
        f_cols = list(config["feature_sets"][f_set])
        target_col = f"fwd_rankpct_{h}" if spec.kind == "regression" else f"outperform_{h}"
        obj = "regression" if spec.kind == "regression" else "binary"
        
        for fold in h_folds:
            t0 = time.time()
            train_df = df_u.loc[fold.train_indices].copy()
            valid_y = train_df[target_col].notna()
            train_df = train_df[valid_y].copy()
            
            dates = sorted(train_df["Date"].unique())
            n_dates = len(dates)
            split_idx = int(n_dates * 0.8)
            val_start = dates[split_idx]
            
            train_df["_t_end"] = train_df.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
            step_dates = set(dates[::5])
            df_sub = train_df[train_df["Date"].isin(step_dates)]
            
            inner_tr_mask = (df_sub["Date"] < val_start) & (df_sub["_t_end"] < val_start)
            inner_val_mask = (df_sub["Date"] >= val_start)
            
            inner_tr_df = df_sub[inner_tr_mask]
            inner_val_df = df_sub[inner_val_mask].copy().reset_index(drop=True)
            
            X_tr = inner_tr_df[f_cols].values
            y_tr = inner_tr_df[target_col].values if obj == "regression" else inner_tr_df[target_col].astype(int).values
            X_val = inner_val_df[f_cols].values
            y_val = inner_val_df[target_col].values if obj == "regression" else inner_val_df[target_col].astype(int).values
            
            unique_val_dates = inner_val_df["Date"].drop_duplicates().tolist()
            sample_val_dates = set(unique_val_dates[::4])
            
            date_groups = []
            for d, grp in inner_val_df.groupby("Date"):
                if d in sample_val_dates and len(grp) >= 8:
                    idx = grp.index.values
                    yr = rankdata(grp[target_fwd].values)
                    yr_c = yr - yr.mean()
                    yr_std = np.linalg.norm(yr_c)
                    if yr_std > 1e-9:
                        date_groups.append((idx, yr_c / yr_std))
                        
            def rank_ic_eval(preds, train_data):
                corrs = []
                for idx, y_norm in date_groups:
                    pr = rankdata(preds[idx])
                    pr_c = pr - pr.mean()
                    pr_std = np.linalg.norm(pr_c)
                    if pr_std > 1e-9:
                        corrs.append(np.dot(pr_c / pr_std, y_norm))
                mean_ic = float(np.mean(corrs)) if corrs else 0.0
                return "rank_ic", mean_ic, True
            
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=f_cols, free_raw_data=False)
            ds_val = lgb.Dataset(X_val, label=y_val, feature_name=f_cols, reference=ds_tr, free_raw_data=False)
            
            best_ic = -999.0
            best_cfg = sampled_configs[0]
            best_iter = 50
            
            for cfg in sampled_configs:
                params = {
                    "objective": obj,
                    "learning_rate": 0.03,
                    "num_leaves": cfg["num_leaves"],
                    "min_child_samples": cfg["min_child_samples"],
                    "feature_fraction": cfg["feature_fraction"],
                    "lambda_l2": cfg["lambda_l2"],
                    "bagging_fraction": 0.3,
                    "bagging_freq": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "seed": 42,
                    "n_jobs": 1,
                    "metric": "None",
                    "verbosity": -1
                }
                callbacks = [lgb.early_stopping(stopping_rounds=50, first_metric_only=False, verbose=False)]
                bst = lgb.train(
                    params,
                    ds_tr,
                    num_boost_round=600,
                    valid_sets=[ds_val],
                    feval=rank_ic_eval,
                    callbacks=callbacks
                )
                val_score = bst.best_score.get("valid_0", {}).get("rank_ic", -999.0)
                iter_count = max(1, bst.best_iteration)
                if val_score > best_ic:
                    best_ic = val_score
                    best_cfg = cfg
                    best_iter = iter_count
                    
            dt = float(time.time() - t0)
            records.append({
                "model": spec.name,
                "horizon": h,
                "feature_set": f_set,
                "fold": fold.fold_name,
                "split_type": fold.split_type,
                "num_leaves": best_cfg["num_leaves"],
                "min_child_samples": best_cfg["min_child_samples"],
                "feature_fraction": best_cfg["feature_fraction"],
                "lambda_l2": best_cfg["lambda_l2"],
                "best_iteration": best_iter,
                "best_val_ic": float(best_ic),
                "train_time_sec": dt
            })
            
        print(f"[{h}] Logged {spec.name}_{f_set} ({len(h_folds)} folds) in {time.time() - t_spec:.1f}s.", flush=True)
        pd.DataFrame(records).to_csv(hp_file, index=False)
        
    print(f"[{h}] Completed in {time.time() - t_start:.1f}s. Saved to: {hp_file}", flush=True)

def merge_all():
    records = []
    for h in ["short", "mid", "long"]:
        hp_file = base_dir / f"reports/06_lgbm_hyperparams_{h}.csv"
        if hp_file.exists():
            records.append(pd.read_csv(hp_file))
    if records:
        merged = pd.concat(records, ignore_index=True)
        main_file = base_dir / "reports/06_lgbm_hyperparams.csv"
        merged.to_csv(main_file, index=False)
        print(f"Merged {len(merged)} records into {main_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="all", choices=["short", "mid", "long", "all", "merge"])
    args = parser.parse_args()
    
    if args.horizon == "merge":
        merge_all()
    elif args.horizon in ["short", "mid", "long"]:
        extract_for_horizon(args.horizon)
    else:
        for h in ["short", "mid", "long"]:
            extract_for_horizon(h)
        merge_all()

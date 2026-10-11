"""Model specification contract and experiment execution engine for Stock Pulse."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.models.preprocessing import (
    fit_preprocessor,
    transform_preprocessor,
    apply_cross_sectional_rank,
    filter_linear_features,
    REDUNDANT_LINEAR_FEATURES
)
from src.models.metrics import evaluate_predictions, compute_null_band_simulation

@dataclass
class ModelSpec:
    """Standardized model specification."""
    name: str
    kind: str  # 'regression' | 'classification'
    build_fn: Callable[[], Any]  # Callable returning model instance
    preprocessing: str = "raw"   # 'raw' | 'cs_rank'
    target_column: str | None = None  # Custom target column override if any


def run_experiment(
    spec: ModelSpec,
    horizon: str,
    feature_set: str = "core",
    config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute model training and evaluation across all walk-forward folds for a horizon.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - oos_predictions: Out-of-sample predictions across all test blocks
            - results_long: Long-format evaluation metrics
    """
    if config is None:
        config = load_config()
        
    base_dir = Path(__file__).resolve().parents[2]
    paths = config["paths"]
    horizons = config["horizons"]
    h_days = horizons[horizon]
    
    # 1. Load data & filter by rankable mask
    features_path = base_dir / paths["features_targets"]
    mask_path = base_dir / paths.get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    if not mask_path.exists():
        from src.data.create_rankable_mask import build_rankable_mask
        mask_df = build_rankable_mask(config)
    else:
        mask_df = pd.read_parquet(mask_path)
        
    rankable_mask = mask_df["rankable"].values
    # Exact common evaluation set: only rankable rows in ranking universe are used
    df_u = df[df["in_ranking_universe"] & rankable_mask].copy()
    
    # Feature columns
    feature_cols = list(config["feature_sets"][feature_set])
    
    # Target column
    if spec.target_column is not None:
        target_col = spec.target_column
    elif spec.kind == "regression":
        target_col = f"fwd_rankpct_{horizon}"
    else:
        target_col = f"outperform_{horizon}"
        
    is_benchmark = spec.name in ["equal_weight_universe", "market_benchmark", "market_equal_weight"]
        
    # 2. Build folds for this horizon on the clean common set
    all_folds = build_folds(df_u, config)
    h_folds = [f for f in all_folds if f.horizon == horizon]
    
    oos_preds = []
    long_rows = []
    
    for fold in h_folds:
        train_df = df_u.loc[fold.train_indices].copy()
        test_df = df_u.loc[fold.test_indices].copy()
        
        # Determine preprocessing
        if spec.preprocessing == "cs_rank":
            # Linear models: neutral imputation 0.5 for cs_rank
            state = fit_preprocessor(train_df, feature_cols, target_col=target_col if spec.kind == "regression" else None)
            train_trans, _ = transform_preprocessor(train_df, state, feature_cols, target_col=target_col if spec.kind == "regression" else None, impute_mode="neutral")
            test_trans, _ = transform_preprocessor(test_df, state, feature_cols, target_col=None, impute_mode="neutral")
            
            candidate_cols = [c for c in feature_cols if c not in REDUNDANT_LINEAR_FEATURES]
            train_trans = apply_cross_sectional_rank(train_trans, candidate_cols)
            test_trans = apply_cross_sectional_rank(test_trans, candidate_cols)
            active_feature_cols, drop_log = filter_linear_features(train_trans, candidate_cols, max_corr=0.98)
        elif spec.preprocessing in ["raw_native", "raw_nan"] or (spec.preprocessing == "raw" and "lgbm" in spec.name):
            # Tree models with native NaN handling (LightGBM): raw features, keep NaNs as is
            state = fit_preprocessor(train_df, feature_cols, target_col=target_col if spec.kind == "regression" else None)
            train_trans, _ = transform_preprocessor(train_df, state, feature_cols, target_col=target_col if spec.kind == "regression" else None, impute_mode="none")
            test_trans, _ = transform_preprocessor(test_df, state, feature_cols, target_col=None, impute_mode="none")
            active_feature_cols = list(feature_cols)
        else:
            # Tree models / baselines: train-median imputation
            state = fit_preprocessor(train_df, feature_cols, target_col=target_col if spec.kind == "regression" else None)
            train_trans, _ = transform_preprocessor(train_df, state, feature_cols, target_col=target_col if spec.kind == "regression" else None, impute_mode="median")
            test_trans, _ = transform_preprocessor(test_df, state, feature_cols, target_col=None, impute_mode="median")
            active_feature_cols = list(feature_cols)
            
        # Instantiate estimator
        estimator = spec.build_fn()
        
        if hasattr(estimator, "feature_set"):
            estimator.feature_set = feature_set
            
        # Pass context if custom estimator supports it
        if hasattr(estimator, "set_context"):
            estimator.set_context(
                feature_names=active_feature_cols,
                train_df=train_trans,
                target_col=target_col,
                fold=fold
            )
            
        # Train model
        X_train = train_trans[active_feature_cols].values
        y_train = train_trans[target_col].values
        estimator.fit(X_train, y_train)
        
        # Check holdout filtering for full feature set (>= 90% non-null full features)
        if fold.split_type == "holdout" and feature_set == "full":
            non_null_rate = test_df[feature_cols].notna().mean(axis=1)
            valid_test_mask = (non_null_rate >= 0.90)
            n_valid_rows = int(valid_test_mask.sum())
            if n_valid_rows < 100 or test_df.loc[valid_test_mask, "Date"].nunique() < 5:
                print(f"Skipping holdout evaluation for {spec.name} {horizon} {feature_set}: only {n_valid_rows} rows have >=90% full features (marked not evaluable).")
                continue
            test_df = test_df[valid_test_mask].copy()
            test_trans = test_trans.loc[test_df.index].copy()
            
        # Pass test context if custom estimator supports it
        if hasattr(estimator, "set_test_context"):
            estimator.set_test_context(test_df=test_trans)
            
        # Predict on test
        X_test = test_trans[active_feature_cols].values
        pred_prob = None
        
        if spec.kind == "classification":
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X_test)
                pred_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
                pred_score = pred_prob
            else:
                pred_score = estimator.predict(X_test)
        else:
            pred_score = estimator.predict(X_test)
            
        # Format fold predictions
        fold_pred_df = pd.DataFrame({
            "series_id": test_df["series_id"].values,
            "Date": test_df["Date"].values,
            "fold": fold.fold_name,
            "split_type": fold.split_type,
            "pred_score": pred_score,
            "pred_prob": pred_prob if pred_prob is not None else np.nan,
            f"fwd_ret_{horizon}": test_df[f"fwd_ret_{horizon}"].values,
            f"fwd_excess_{horizon}": test_df[f"fwd_excess_{horizon}"].values,
            f"fwd_rankpct_{horizon}": test_df[f"fwd_rankpct_{horizon}"].values,
            f"outperform_{horizon}": test_df[f"outperform_{horizon}"].values
        })
        oos_preds.append(fold_pred_df)
        
        # Evaluate fold metrics
        metrics = evaluate_predictions(fold_pred_df, horizon=horizon, h_days=h_days, config=config, is_benchmark=is_benchmark)
        
        for m_name, m_val in metrics.items():
            if isinstance(m_val, (int, float, np.number)):
                val_float = float(m_val) if pd.notnull(m_val) else np.nan
            else:
                continue
            long_rows.append({
                "model": spec.name,
                "feature_set": feature_set,
                "horizon": horizon,
                "fold": fold.fold_name,
                "split_type": fold.split_type,
                "metric": m_name,
                "value": val_float
            })
            
    df_oos = pd.concat(oos_preds, ignore_index=True) if oos_preds else pd.DataFrame()
    
    # Evaluate partition-level aggregates (all periods across partition)
    if not df_oos.empty:
        df_long_folds = pd.DataFrame(long_rows)
        for s_type in df_oos["split_type"].unique():
            part_df = df_oos[df_oos["split_type"] == s_type].copy()
            if part_df.empty:
                continue
            part_metrics = evaluate_predictions(part_df, horizon=horizon, h_days=h_days, config=config, is_benchmark=is_benchmark)
            
            # Effective sample gating (A5)
            n_eff = part_metrics.get("portfolio_num_periods", 0)
            part_metrics["n_eff"] = float(n_eff)
            
            # Compute partition null band and permutation p-value
            if not is_benchmark:
                null_band = compute_null_band_simulation(
                    part_df, horizon=horizon, h_days=h_days, split_type=s_type,
                    seed=config.get("random_seed", 42), k=config["validation"].get("top_k", 5),
                    config=config
                )
                part_ic = part_metrics.get("rank_ic_mean", np.nan)
                
                if n_eff < 5:
                    # Gating: if n_eff < 5, suppress IC verdict and p-value
                    part_metrics["outside_null_band"] = np.nan
                    part_metrics["permutation_p_val"] = np.nan
                    part_metrics["evaluable_status_code"] = 0.0  # 0: not evaluable
                else:
                    if pd.notnull(part_ic):
                        outside = "yes" if (part_ic < null_band["null_band_ic_lower"] or part_ic > null_band["null_band_ic_upper"]) else "no"
                        p_val = float(np.mean(null_band["draw_ics"] >= part_ic))
                    else:
                        outside = "no"
                        p_val = np.nan
                    part_metrics["outside_null_band"] = 1.0 if outside == "yes" else 0.0
                    part_metrics["permutation_p_val"] = p_val
                    part_metrics["evaluable_status_code"] = 1.0 if n_eff < 12 else 2.0  # 1: indicative, 2: evaluable
                    
                part_metrics["null_band_ic_lower"] = null_band["null_band_ic_lower"]
                part_metrics["null_band_ic_upper"] = null_band["null_band_ic_upper"]
                part_metrics["null_band_ic_mean"] = null_band["null_band_ic_mean"]
                part_metrics["block_null_band_ic_lower"] = null_band["block_null_band_ic_lower"]
                part_metrics["block_null_band_ic_upper"] = null_band["block_null_band_ic_upper"]
                part_metrics["linear_null_band_ic_lower"] = null_band["linear_null_band_ic_lower"]
                part_metrics["linear_null_band_ic_upper"] = null_band["linear_null_band_ic_upper"]
                part_metrics["old_null_band_ic_lower"] = null_band["old_null_band_ic_lower"]
                part_metrics["old_null_band_ic_upper"] = null_band["old_null_band_ic_upper"]
            
            # Fold-level statistics across all folds in this partition
            fold_names = [f.fold_name for f in h_folds if f.split_type == s_type]
            fold_ic_rows = df_long_folds[
                (df_long_folds["fold"].isin(fold_names)) &
                (df_long_folds["metric"] == "rank_ic_mean")
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
                
            for m_name, m_val in part_metrics.items():
                if isinstance(m_val, (int, float, np.number)):
                    val_float = float(m_val) if pd.notnull(m_val) else np.nan
                else:
                    continue
                long_rows.append({
                    "model": spec.name,
                    "feature_set": feature_set,
                    "horizon": horizon,
                    "fold": s_type,
                    "split_type": s_type,
                    "metric": m_name,
                    "value": val_float
                })
                
    df_long = pd.DataFrame(long_rows)
    
    # 3. Save OOS predictions
    oos_dir = base_dir / paths["oos_predictions_dir"]
    oos_dir.mkdir(parents=True, exist_ok=True)
    oos_file = oos_dir / f"{spec.name}_{horizon}_{feature_set}.parquet"
    if not df_oos.empty:
        df_oos.to_parquet(oos_file, engine="pyarrow", index=False)
        print(f"Saved out-of-sample predictions to: {oos_file}")
        
    # 4. Append metrics to results_long.csv
    results_long_path = base_dir / paths["results_long_csv"]
    results_long_path.parent.mkdir(parents=True, exist_ok=True)
    if results_long_path.exists():
        existing_results = pd.read_csv(results_long_path)
        mask_existing = ~(
            (existing_results["model"] == spec.name) &
            (existing_results["feature_set"] == feature_set) &
            (existing_results["horizon"] == horizon)
        )
        existing_results = existing_results[mask_existing]
        combined_results = pd.concat([existing_results, df_long], ignore_index=True)
    else:
        combined_results = df_long
        
    combined_results.to_csv(results_long_path, index=False)
    print(f"Updated results in: {results_long_path}")
    
    return df_oos, df_long

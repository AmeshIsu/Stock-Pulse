"""Determinism check script for Stock Pulse.

Re-fits ridge_core, momentum_126, rf_reg_core, and lgbm_reg_core in temporary directories
without touching any frozen prediction files.
"""
import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.models.baselines import FeatureBaselineEstimator
from src.models.boosting_models import LightGBMRegressorEstimator
from src.models.contract import ModelSpec, run_experiment
from src.models.forest_models import RandomForestRegressorEstimator
from src.models.linear_models import RidgeModelEstimator
from src.models.validation import build_folds
from src.models.preprocessing import (
    fit_preprocessor,
    transform_preprocessor,
    apply_cross_sectional_rank,
    filter_linear_features,
    REDUNDANT_LINEAR_FEATURES,
)

def run_determinism_checks():
    base_dir = Path(__file__).resolve().parents[1]
    raw_config = load_config()

    scratch_dir = base_dir / "scratch" / "determinism_check"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ridge_core": {},
        "momentum_126": {},
        "rf_reg_core_dev_2015_short": {},
        "lgbm_reg_core_dev_2015_short": {},
        "all_passed": True,
    }

    # -------------------------------------------------------------
    # 1. RIDGE CORE (All horizons: short, mid, long)
    # -------------------------------------------------------------
    print("=== Checking RIDGE_CORE Determinism (All Horizons, All Folds) ===", flush=True)
    ridge_spec = ModelSpec(
        name="ridge",
        kind="regression",
        build_fn=lambda: RidgeModelEstimator(),
        preprocessing="cs_rank",
    )

    for h in ["short", "mid", "long"]:
        print(f"--- Ridge {h} ---", flush=True)
        # Setup temporary configs
        cfg1 = copy.deepcopy(raw_config)
        cfg1["paths"]["oos_predictions_dir"] = f"scratch/determinism_check/run1_oos"
        cfg1["paths"]["results_long_csv"] = f"scratch/determinism_check/run1_results.csv"

        cfg2 = copy.deepcopy(raw_config)
        cfg2["paths"]["oos_predictions_dir"] = f"scratch/determinism_check/run2_oos"
        cfg2["paths"]["results_long_csv"] = f"scratch/determinism_check/run2_results.csv"

        t0 = time.time()
        preds1, _ = run_experiment(ridge_spec, horizon=h, feature_set="core", config=cfg1)
        t1 = time.time()
        print(f"  Run 1 finished in {t1 - t0:.1f}s", flush=True)

        preds2, _ = run_experiment(ridge_spec, horizon=h, feature_set="core", config=cfg2)
        t2 = time.time()
        print(f"  Run 2 finished in {t2 - t1:.1f}s", flush=True)

        # Run 1 vs Run 2 diff
        diff_1_2 = float(np.max(np.abs(preds1["pred_score"].values - preds2["pred_score"].values)))

        # Run 1 vs Frozen diff
        frozen_file = base_dir / "reports" / "oos_predictions" / f"ridge_{h}_core.parquet"
        frozen_df = pd.read_parquet(frozen_file)

        m = pd.merge(
            preds1[["series_id", "Date", "fold", "pred_score"]],
            frozen_df[["series_id", "Date", "fold", "pred_score"]],
            on=["series_id", "Date", "fold"],
            suffixes=("_refit", "_frozen"),
        )

        assert len(m) == len(frozen_df), f"Row count mismatch: {len(m)} vs {len(frozen_df)}"
        diff_frozen = float(np.max(np.abs(m["pred_score_refit"].values - m["pred_score_frozen"].values)))
        corr_frozen = float(np.corrcoef(m["pred_score_refit"].values, m["pred_score_frozen"].values)[0, 1])

        print(f"  Max score diff (Run 1 vs Run 2): {diff_1_2:.2e}", flush=True)
        print(f"  Max score diff (Run 1 vs Frozen): {diff_frozen:.2e}", flush=True)
        print(f"  Correlation with Frozen: {corr_frozen:.10f}", flush=True)

        passed = diff_frozen < 1e-9 and diff_1_2 < 1e-9
        results["ridge_core"][h] = {
            "run1_vs_run2_max_diff": diff_1_2,
            "refit_vs_frozen_max_diff": diff_frozen,
            "correlation": corr_frozen,
            "passed": passed,
        }
        if not passed:
            results["all_passed"] = False

    # -------------------------------------------------------------
    # 2. MOMENTUM_126 (All horizons: short, mid, long)
    # -------------------------------------------------------------
    print("\n=== Checking MOMENTUM_126 Determinism (All Horizons) ===", flush=True)
    mom_spec = ModelSpec(
        name="momentum_126",
        kind="regression",
        build_fn=lambda: FeatureBaselineEstimator("log_ret_126", sign=1.0),
        preprocessing="raw",
    )

    for h in ["short", "mid", "long"]:
        print(f"--- Momentum_126 {h} ---", flush=True)
        cfg = copy.deepcopy(raw_config)
        cfg["paths"]["oos_predictions_dir"] = f"scratch/determinism_check/mom_oos"
        cfg["paths"]["results_long_csv"] = f"scratch/determinism_check/mom_results.csv"

        preds, _ = run_experiment(mom_spec, horizon=h, feature_set="core", config=cfg)
        frozen_file = base_dir / "reports" / "oos_predictions" / f"momentum_126_{h}_core.parquet"
        frozen_df = pd.read_parquet(frozen_file)

        m = pd.merge(
            preds[["series_id", "Date", "fold", "pred_score"]],
            frozen_df[["series_id", "Date", "fold", "pred_score"]],
            on=["series_id", "Date", "fold"],
            suffixes=("_refit", "_frozen"),
        )
        assert len(m) == len(frozen_df), f"Row count mismatch: {len(m)} vs {len(frozen_df)}"
        diff_frozen = float(np.max(np.abs(m["pred_score_refit"].values - m["pred_score_frozen"].values)))
        corr_frozen = float(np.corrcoef(m["pred_score_refit"].values, m["pred_score_frozen"].values)[0, 1])

        print(f"  Max score diff vs Frozen: {diff_frozen:.2e}", flush=True)
        print(f"  Correlation with Frozen: {corr_frozen:.10f}", flush=True)

        passed = diff_frozen < 1e-9
        results["momentum_126"][h] = {
            "refit_vs_frozen_max_diff": diff_frozen,
            "correlation": corr_frozen,
            "passed": passed,
        }
        if not passed:
            results["all_passed"] = False

    # -------------------------------------------------------------
    # 3. RF_REG_CORE & LGBM_REG_CORE (One Fold: dev_2015, short)
    # -------------------------------------------------------------
    print("\n=== Checking RF_REG_CORE & LGBM_REG_CORE (dev_2015, short) ===", flush=True)
    # Load dataset and prepare exact fold context for dev_2015
    features_path = base_dir / raw_config["paths"]["features_targets"]
    mask_path = base_dir / raw_config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    df = pd.read_parquet(features_path)
    mask_df = pd.read_parquet(mask_path)
    df_u = df[df["in_ranking_universe"] & mask_df["rankable"].values].copy()

    all_folds = build_folds(df_u, raw_config)
    fold_2015 = next(f for f in all_folds if f.horizon == "short" and f.fold_name == "dev_2015")
    feature_cols = list(raw_config["feature_sets"]["core"])
    target_col = "fwd_rankpct_short"

    train_df = df_u.loc[fold_2015.train_indices].copy()
    test_df = df_u.loc[fold_2015.test_indices].copy()

    # Preprocessing for RF and LGBM
    state = fit_preprocessor(train_df, feature_cols, target_col=target_col)
    train_trans, _ = transform_preprocessor(train_df, state, feature_cols, target_col=target_col, impute_mode="median")
    test_trans, _ = transform_preprocessor(test_df, state, feature_cols, target_col=None, impute_mode="median")

    X_train = train_trans[feature_cols].values
    y_train = train_trans[target_col].values
    X_test = test_trans[feature_cols].values

    # Check RF
    print("--- RF Reg Core dev_2015 short ---", flush=True)
    rf_est = RandomForestRegressorEstimator(random_state=42)
    rf_est.set_context(feature_cols, train_df, target_col, fold_2015)
    rf_est.fit(X_train, y_train)
    rf_preds = rf_est.predict(X_test)

    rf_frozen_file = base_dir / "reports" / "oos_predictions" / "rf_reg_short_core.parquet"
    rf_frozen_df = pd.read_parquet(rf_frozen_file)
    rf_sub = rf_frozen_df[rf_frozen_df["fold"] == "dev_2015"].copy()

    test_df["pred_score_refit"] = rf_preds
    m_rf = pd.merge(
        test_df[["series_id", "Date", "pred_score_refit"]],
        rf_sub[["series_id", "Date", "pred_score"]],
        on=["series_id", "Date"],
    )
    diff_rf = float(np.max(np.abs(m_rf["pred_score_refit"].values - m_rf["pred_score"].values)))
    corr_rf = float(np.corrcoef(m_rf["pred_score_refit"].values, m_rf["pred_score"].values)[0, 1])
    print(f"  RF Max score diff: {diff_rf:.2e}", flush=True)
    print(f"  RF Correlation: {corr_rf:.10f}", flush=True)

    results["rf_reg_core_dev_2015_short"] = {
        "max_diff": diff_rf,
        "correlation": corr_rf,
        "passed": diff_rf < 1e-6,
    }
    if diff_rf >= 1e-6:
        results["all_passed"] = False

    # Check LightGBM
    print("--- LGBM Reg Core dev_2015 short ---", flush=True)
    lgbm_est = LightGBMRegressorEstimator(random_state=42)
    lgbm_est.set_context(feature_cols, train_df, target_col, fold_2015)
    lgbm_est.fit(X_train, y_train)
    lgbm_preds = lgbm_est.predict(X_test)

    lgbm_frozen_file = base_dir / "reports" / "oos_predictions" / "lgbm_reg_short_core.parquet"
    lgbm_frozen_df = pd.read_parquet(lgbm_frozen_file)
    lgbm_sub = lgbm_frozen_df[lgbm_frozen_df["fold"] == "dev_2015"].copy()

    test_df["pred_score_refit"] = lgbm_preds
    m_lgbm = pd.merge(
        test_df[["series_id", "Date", "pred_score_refit"]],
        lgbm_sub[["series_id", "Date", "pred_score"]],
        on=["series_id", "Date"],
    )
    diff_lgbm = float(np.max(np.abs(m_lgbm["pred_score_refit"].values - m_lgbm["pred_score"].values)))
    corr_lgbm = float(np.corrcoef(m_lgbm["pred_score_refit"].values, m_lgbm["pred_score"].values)[0, 1])
    print(f"  LGBM Max score diff: {diff_lgbm:.2e}", flush=True)
    print(f"  LGBM Correlation: {corr_lgbm:.10f}", flush=True)

    results["lgbm_reg_core_dev_2015_short"] = {
        "max_diff": diff_lgbm,
        "correlation": corr_lgbm,
        "passed": diff_lgbm < 1e-6,
    }
    if diff_lgbm >= 1e-6:
        results["all_passed"] = False

    # Save results
    out_file = scratch_dir / "determinism_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll checks complete. Saved results to {out_file}", flush=True)
    print(f"Overall determinism pass: {results['all_passed']}", flush=True)

if __name__ == "__main__":
    run_determinism_checks()

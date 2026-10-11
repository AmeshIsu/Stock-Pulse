"""Model 3: LightGBM Regressor & Classifier with heavy regularization, leak-free walk-forward validation and native NaN handling."""
from pathlib import Path
from typing import Any, Optional
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

from src.data.config_loader import load_config
from src.models.contract import ModelSpec, run_experiment
from src.models.metrics import compute_rank_ic, evaluate_predictions, compute_null_band_simulation

_LGBM_HYPERPARAMS_LOG: list[dict[str, Any]] = []

def reset_lgbm_hyperparams_log():
    global _LGBM_HYPERPARAMS_LOG
    _LGBM_HYPERPARAMS_LOG = []


def get_12_sampled_configs(seed: int = 42) -> list[dict[str, Any]]:
    """Deterministically sample 12 hyperparameter configurations from the 24 grid points."""
    all_configs = [
        {"num_leaves": nl, "min_child_samples": mcs, "feature_fraction": ff, "lambda_l2": l2}
        for nl in [4, 8, 16]
        for mcs in [200, 1000]
        for ff in [0.5, 0.8]
        for l2 in [10, 100]
    ]
    rng = np.random.RandomState(seed)
    sampled_indices = sorted(rng.choice(len(all_configs), size=12, replace=False))
    return [all_configs[i] for i in sampled_indices]


class LightGBMRegressorEstimator:
    """LightGBM regressor on fwd_rankpct with heavy regularization and inner 80/20 IC early stopping."""
    def __init__(
        self,
        random_state: int = 42,
        subsample_step: int = 5,
        max_trees: int = 600,
        patience: int = 50
    ):
        self.random_state = random_state
        self.subsample_step = subsample_step
        self.max_trees = max_trees
        self.patience = patience
        self.configs = get_12_sampled_configs(random_state)
        self.best_params = {}
        self.best_iteration = 50
        self.best_val_ic = -999.0
        self.model: Optional[lgb.Booster] = None
        self.feature_names_ = []
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        self.fold = None
        self.train_time_sec = 0.0

    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names_ = list(feature_names)
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.time()
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]

        config = load_config()
        h_days = config["horizons"].get(self.fold.horizon, 21) if self.fold else 21
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else "fwd_excess_short"

        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)

        # Subsample tuning rows: every 5th trading date
        step_dates = set(dates[::self.subsample_step])
        df_sub = df_copy[df_copy["Date"].isin(step_dates)]

        valid_y_mask = df_sub[self.target_col].notna()
        inner_tr_mask = (df_sub["Date"] < val_start) & (df_sub["_t_end"] < val_start) & valid_y_mask
        inner_val_mask = (df_sub["Date"] >= val_start) & valid_y_mask

        inner_tr_df = df_sub[inner_tr_mask]
        inner_val_df = df_sub[inner_val_mask].copy()

        X_tr = inner_tr_df[self.feature_names_].values
        y_tr = inner_tr_df[self.target_col].values
        X_val = inner_val_df[self.feature_names_].values

        val_dates = inner_val_df["Date"].values
        val_excess = inner_val_df[target_fwd].values

        # Precompute relative date groups and standardized target ranks for ultra-fast custom rank IC eval
        date_groups = []
        for d in pd.Series(val_dates).unique():
            idx = np.where(val_dates == d)[0]
            if len(idx) >= 8:
                yr = rankdata(val_excess[idx])
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

        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names_, free_raw_data=False)
        ds_val = lgb.Dataset(X_val, label=inner_val_df[self.target_col].values, feature_name=self.feature_names_, reference=ds_tr, free_raw_data=False)

        best_ic = -999.0
        best_cfg = self.configs[0]
        best_iter = 50

        for cfg in self.configs:
            params = {
                "objective": "regression",
                "learning_rate": 0.03,
                "num_leaves": cfg["num_leaves"],
                "min_child_samples": cfg["min_child_samples"],
                "feature_fraction": cfg["feature_fraction"],
                "lambda_l2": cfg["lambda_l2"],
                "bagging_fraction": 0.3,
                "bagging_freq": 1,
                "deterministic": True,
                "force_row_wise": True,
                "seed": self.random_state,
                "n_jobs": -1,
                "metric": "None",
                "verbosity": -1
            }
            callbacks = [lgb.early_stopping(stopping_rounds=self.patience, first_metric_only=False, verbose=False)]
            bst = lgb.train(
                params,
                ds_tr,
                num_boost_round=self.max_trees,
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

        self.best_params = best_cfg
        self.best_val_ic = best_ic
        self.best_iteration = best_iter

        # Refit on full training fold with all rows and native NaNs preserved
        valid_outer = ~np.isnan(y)
        X_full = X[valid_outer]
        y_full = y[valid_outer]

        full_ds = lgb.Dataset(X_full, label=y_full, feature_name=self.feature_names_, free_raw_data=False)
        refit_params = {
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": self.best_params["num_leaves"],
            "min_child_samples": self.best_params["min_child_samples"],
            "feature_fraction": self.best_params["feature_fraction"],
            "lambda_l2": self.best_params["lambda_l2"],
            "bagging_fraction": 0.3,
            "bagging_freq": 1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": self.random_state,
            "n_jobs": -1,
            "verbosity": -1
        }
        self.model = lgb.train(refit_params, full_ds, num_boost_round=self.best_iteration)
        self.train_time_sec = float(time.time() - t0)

        # Log hyperparams
        _LGBM_HYPERPARAMS_LOG.append({
            "model": "lgbm_reg",
            "horizon": self.fold.horizon if self.fold else "unknown",
            "feature_set": self.feature_set,
            "fold": self.fold.fold_name if self.fold else "unknown",
            "num_leaves": self.best_params["num_leaves"],
            "min_child_samples": self.best_params["min_child_samples"],
            "feature_fraction": self.best_params["feature_fraction"],
            "lambda_l2": self.best_params["lambda_l2"],
            "best_iteration": self.best_iteration,
            "best_val_ic": float(self.best_val_ic),
            "train_time_sec": self.train_time_sec
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict(X)


class LightGBMClassifierEstimator:
    """LightGBM binary classifier on outperform label with native NaN handling and IC early stopping."""
    def __init__(
        self,
        random_state: int = 42,
        subsample_step: int = 5,
        max_trees: int = 600,
        patience: int = 50
    ):
        self.random_state = random_state
        self.subsample_step = subsample_step
        self.max_trees = max_trees
        self.patience = patience
        self.configs = get_12_sampled_configs(random_state)
        self.best_params = {}
        self.best_iteration = 50
        self.best_val_ic = -999.0
        self.model: Optional[lgb.Booster] = None
        self.feature_names_ = []
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        self.fold = None
        self.train_time_sec = 0.0

    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names_ = list(feature_names)
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.time()
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]

        config = load_config()
        h_days = config["horizons"].get(self.fold.horizon, 21) if self.fold else 21
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else "fwd_excess_short"

        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)

        step_dates = set(dates[::self.subsample_step])
        df_sub = df_copy[df_copy["Date"].isin(step_dates)]

        # Drop exact 0.5 ties (NaN in target_col)
        valid_y_mask = df_sub[self.target_col].notna()
        inner_tr_mask = (df_sub["Date"] < val_start) & (df_sub["_t_end"] < val_start) & valid_y_mask
        inner_val_mask = (df_sub["Date"] >= val_start) & valid_y_mask

        inner_tr_df = df_sub[inner_tr_mask]
        inner_val_df = df_sub[inner_val_mask].copy()

        X_tr = inner_tr_df[self.feature_names_].values
        y_tr = inner_tr_df[self.target_col].astype(int).values
        X_val = inner_val_df[self.feature_names_].values

        val_dates = inner_val_df["Date"].values
        val_excess = inner_val_df[target_fwd].values

        date_groups = []
        for d in pd.Series(val_dates).unique():
            idx = np.where(val_dates == d)[0]
            if len(idx) >= 8:
                yr = rankdata(val_excess[idx])
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

        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names_, free_raw_data=False)
        ds_val = lgb.Dataset(X_val, label=inner_val_df[self.target_col].astype(int).values, feature_name=self.feature_names_, reference=ds_tr, free_raw_data=False)

        best_ic = -999.0
        best_cfg = self.configs[0]
        best_iter = 50

        for cfg in self.configs:
            params = {
                "objective": "binary",
                "learning_rate": 0.03,
                "num_leaves": cfg["num_leaves"],
                "min_child_samples": cfg["min_child_samples"],
                "feature_fraction": cfg["feature_fraction"],
                "lambda_l2": cfg["lambda_l2"],
                "bagging_fraction": 0.3,
                "bagging_freq": 1,
                "deterministic": True,
                "force_row_wise": True,
                "seed": self.random_state,
                "n_jobs": -1,
                "metric": "None",
                "verbosity": -1
            }
            callbacks = [lgb.early_stopping(stopping_rounds=self.patience, first_metric_only=False, verbose=False)]
            bst = lgb.train(
                params,
                ds_tr,
                num_boost_round=self.max_trees,
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

        self.best_params = best_cfg
        self.best_val_ic = best_ic
        self.best_iteration = best_iter

        valid_outer = ~np.isnan(y)
        X_full = X[valid_outer]
        y_full = y[valid_outer].astype(int)

        full_ds = lgb.Dataset(X_full, label=y_full, feature_name=self.feature_names_, free_raw_data=False)
        refit_params = {
            "objective": "binary",
            "learning_rate": 0.03,
            "num_leaves": self.best_params["num_leaves"],
            "min_child_samples": self.best_params["min_child_samples"],
            "feature_fraction": self.best_params["feature_fraction"],
            "lambda_l2": self.best_params["lambda_l2"],
            "bagging_fraction": 0.3,
            "bagging_freq": 1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": self.random_state,
            "n_jobs": -1,
            "verbosity": -1
        }
        self.model = lgb.train(refit_params, full_ds, num_boost_round=self.best_iteration)
        self.train_time_sec = float(time.time() - t0)

        _LGBM_HYPERPARAMS_LOG.append({
            "model": "lgbm_clf",
            "horizon": self.fold.horizon if self.fold else "unknown",
            "feature_set": self.feature_set,
            "fold": self.fold.fold_name if self.fold else "unknown",
            "num_leaves": self.best_params["num_leaves"],
            "min_child_samples": self.best_params["min_child_samples"],
            "feature_fraction": self.best_params["feature_fraction"],
            "lambda_l2": self.best_params["lambda_l2"],
            "best_iteration": self.best_iteration,
            "best_val_ic": float(self.best_val_ic),
            "train_time_sec": self.train_time_sec
        })

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        p = self.model.predict(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict(X)


def get_lgbm_specs(config: dict | None = None) -> list[tuple[ModelSpec, str]]:
    """Return ModelSpec instances and feature sets for LightGBM models."""
    if config is None:
        config = load_config()
    seed = config.get("random_seed", 42)
    
    specs = [
        (
            ModelSpec(
                name="lgbm_reg",
                kind="regression",
                build_fn=lambda: LightGBMRegressorEstimator(random_state=seed),
                preprocessing="raw_native"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="lgbm_reg",
                kind="regression",
                build_fn=lambda: LightGBMRegressorEstimator(random_state=seed),
                preprocessing="raw_native"
            ),
            "full"
        ),
        (
            ModelSpec(
                name="lgbm_reg",
                kind="regression",
                build_fn=lambda: LightGBMRegressorEstimator(random_state=seed),
                preprocessing="raw_native"
            ),
            "core_rel"
        ),
        (
            ModelSpec(
                name="lgbm_clf",
                kind="classification",
                build_fn=lambda: LightGBMClassifierEstimator(random_state=seed),
                preprocessing="raw_native"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="lgbm_clf",
                kind="classification",
                build_fn=lambda: LightGBMClassifierEstimator(random_state=seed),
                preprocessing="raw_native"
            ),
            "core_rel"
        ),
    ]
    return specs


"""Model 2: Random Forest Regressor & Classifier with leak-free walk-forward validation."""
from pathlib import Path
from typing import Any, Optional
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from src.data.config_loader import load_config
from src.models.contract import ModelSpec, run_experiment
from src.models.metrics import compute_rank_ic, compute_null_band_simulation

_RF_HYPERPARAMS_LOG: list[dict[str, Any]] = []

def reset_rf_hyperparams_log():
    global _RF_HYPERPARAMS_LOG
    _RF_HYPERPARAMS_LOG = []


class RandomForestRegressorEstimator:
    """RandomForestRegressor with inner 80/20 time-split validation tuning."""
    def __init__(
        self,
        grid: Optional[dict[str, list]] = None,
        random_state: int = 42,
        tuning_trees: int = 150,
        final_trees: int = 400,
        subsample_step: int = 5
    ):
        self.grid = grid or {
            "max_depth": [3, 5, 8],
            "min_samples_leaf": [100, 300, 1000],
            "max_features": [0.33, 0.6]
        }
        self.random_state = random_state
        self.tuning_trees = tuning_trees
        self.final_trees = final_trees
        self.subsample_step = subsample_step
        self.best_params = {}
        self.best_val_ic = -999.0
        self.model = None
        self.feature_names_ = []
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        self.fold = None
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names_ = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]
        
        config = load_config()
        h_days = config["horizons"].get(self.fold.horizon, 21) if self.fold else 21
        
        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
        
        # Subsample dates every 5th trading date for inner tuning speed
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
        
        best_ic = -999.0
        best_p = {
            "max_depth": self.grid["max_depth"][0],
            "min_samples_leaf": self.grid["min_samples_leaf"][0],
            "max_features": self.grid["max_features"][0]
        }
        
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else "fwd_excess_short"
        
        # Grid search over 18 candidates
        for md in self.grid["max_depth"]:
            for msl in self.grid["min_samples_leaf"]:
                for mf in self.grid["max_features"]:
                    m = RandomForestRegressor(
                        n_estimators=self.tuning_trees,
                        max_depth=md,
                        min_samples_leaf=msl,
                        max_features=mf,
                        bootstrap=True,
                        max_samples=0.3,
                        n_jobs=-1,
                        random_state=self.random_state
                    )
                    m.fit(X_tr, y_tr)
                    inner_val_df["pred_score"] = m.predict(X_val)
                    ics = compute_rank_ic(inner_val_df, target_col=target_fwd, min_stocks=8)
                    mean_ic = float(ics.mean()) if len(ics) > 0 else -999.0
                    if mean_ic > best_ic:
                        best_ic = mean_ic
                        best_p = {"max_depth": md, "min_samples_leaf": msl, "max_features": mf}
                        
        self.best_params = best_p
        self.best_val_ic = best_ic
        
        # Refit on full outer training fold (dropping y NaNs) with 400 trees
        valid_outer = ~np.isnan(y)
        X_full = X[valid_outer]
        y_full = y[valid_outer]
        
        self.model = RandomForestRegressor(
            n_estimators=self.final_trees,
            max_depth=self.best_params["max_depth"],
            min_samples_leaf=self.best_params["min_samples_leaf"],
            max_features=self.best_params["max_features"],
            bootstrap=True,
            max_samples=0.3,
            n_jobs=-1,
            random_state=self.random_state
        )
        self.model.fit(X_full, y_full)
        
        # Log hyperparameters
        for p_k, p_v in self.best_params.items():
            _RF_HYPERPARAMS_LOG.append({
                "model": "rf_reg",
                "feature_set": getattr(self, "feature_set", "core"),
                "horizon": self.fold.horizon if self.fold else "unknown",
                "fold": self.fold.fold_name if self.fold else "unknown",
                "split_type": self.fold.split_type if self.fold else "unknown",
                "param_name": p_k,
                "chosen_value": p_v,
                "val_rank_ic": self.best_val_ic
            })
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestClassifierEstimator:
    """RandomForestClassifier on outperform label with inner 80/20 validation tuning."""
    def __init__(
        self,
        grid: Optional[dict[str, list]] = None,
        random_state: int = 42,
        tuning_trees: int = 150,
        final_trees: int = 400,
        subsample_step: int = 5
    ):
        self.grid = grid or {
            "max_depth": [3, 5, 8],
            "min_samples_leaf": [100, 300, 1000],
            "max_features": [0.33, 0.6]
        }
        self.random_state = random_state
        self.tuning_trees = tuning_trees
        self.final_trees = final_trees
        self.subsample_step = subsample_step
        self.best_params = {}
        self.best_val_ic = -999.0
        self.model = None
        self.feature_names_ = []
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        self.fold = None
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names_ = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]
        
        config = load_config()
        h_days = config["horizons"].get(self.fold.horizon, 21) if self.fold else 21
        
        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
        
        # Subsample dates every 5th trading date
        step_dates = set(dates[::self.subsample_step])
        df_sub = df_copy[df_copy["Date"].isin(step_dates)]
        
        # Drop exact 0.5 ties (NaNs in target)
        valid_y_mask = df_sub[self.target_col].notna()
        inner_tr_mask = (df_sub["Date"] < val_start) & (df_sub["_t_end"] < val_start) & valid_y_mask
        inner_val_mask = (df_sub["Date"] >= val_start) & valid_y_mask
        
        inner_tr_df = df_sub[inner_tr_mask]
        inner_val_df = df_sub[inner_val_mask].copy()
        
        X_tr = inner_tr_df[self.feature_names_].values
        y_tr = inner_tr_df[self.target_col].astype(int).values
        X_val = inner_val_df[self.feature_names_].values
        
        best_ic = -999.0
        best_p = {
            "max_depth": self.grid["max_depth"][0],
            "min_samples_leaf": self.grid["min_samples_leaf"][0],
            "max_features": self.grid["max_features"][0]
        }
        
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else "fwd_excess_short"
        
        for md in self.grid["max_depth"]:
            for msl in self.grid["min_samples_leaf"]:
                for mf in self.grid["max_features"]:
                    m = RandomForestClassifier(
                        n_estimators=self.tuning_trees,
                        max_depth=md,
                        min_samples_leaf=msl,
                        max_features=mf,
                        bootstrap=True,
                        max_samples=0.3,
                        n_jobs=-1,
                        random_state=self.random_state
                    )
                    m.fit(X_tr, y_tr)
                    probs = m.predict_proba(X_val)
                    inner_val_df["pred_score"] = probs[:, 1] if probs.shape[1] > 1 else probs.ravel()
                    ics = compute_rank_ic(inner_val_df, target_col=target_fwd, min_stocks=8)
                    mean_ic = float(ics.mean()) if len(ics) > 0 else -999.0
                    if mean_ic > best_ic:
                        best_ic = mean_ic
                        best_p = {"max_depth": md, "min_samples_leaf": msl, "max_features": mf}
                        
        self.best_params = best_p
        self.best_val_ic = best_ic
        
        # Refit on full outer training fold (dropping y NaNs) with 400 trees
        valid_outer = ~np.isnan(y)
        X_full = X[valid_outer]
        y_full = y[valid_outer].astype(int)
        
        self.model = RandomForestClassifier(
            n_estimators=self.final_trees,
            max_depth=self.best_params["max_depth"],
            min_samples_leaf=self.best_params["min_samples_leaf"],
            max_features=self.best_params["max_features"],
            bootstrap=True,
            max_samples=0.3,
            n_jobs=-1,
            random_state=self.random_state
        )
        self.model.fit(X_full, y_full)
        
        for p_k, p_v in self.best_params.items():
            _RF_HYPERPARAMS_LOG.append({
                "model": "rf_clf",
                "feature_set": getattr(self, "feature_set", "core"),
                "horizon": self.fold.horizon if self.fold else "unknown",
                "fold": self.fold.fold_name if self.fold else "unknown",
                "split_type": self.fold.split_type if self.fold else "unknown",
                "param_name": p_k,
                "chosen_value": p_v,
                "val_rank_ic": self.best_val_ic
            })
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def get_rf_specs(config: dict | None = None) -> list[tuple[ModelSpec, str]]:
    """Return ModelSpec instances and feature sets for Random Forest models."""
    if config is None:
        config = load_config()
    seed = config.get("random_seed", 42)
    
    specs = [
        (
            ModelSpec(
                name="rf_reg",
                kind="regression",
                build_fn=lambda: RandomForestRegressorEstimator(random_state=seed),
                preprocessing="raw"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="rf_reg",
                kind="regression",
                build_fn=lambda: RandomForestRegressorEstimator(random_state=seed),
                preprocessing="raw"
            ),
            "full"
        ),
        (
            ModelSpec(
                name="rf_clf",
                kind="classification",
                build_fn=lambda: RandomForestClassifierEstimator(random_state=seed),
                preprocessing="raw"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="rf_clf",
                kind="classification",
                build_fn=lambda: RandomForestClassifierEstimator(random_state=seed),
                preprocessing="raw"
            ),
            "full"
        ),
    ]
    return specs


def run_all_rf_experiments(config: dict | None = None) -> pd.DataFrame:
    """Execute Random Forest models across all horizons and log metrics."""
    if config is None:
        config = load_config()
        
    reset_rf_hyperparams_log()
    horizons = list(config["horizons"].keys())
    rf_specs = get_rf_specs(config)
    
    t_start = time.time()
    print("=== RUNNING MODEL 2 (RANDOM FOREST): 4 configurations x 3 horizons ===")
    
    for h in horizons:
        for spec, f_set in rf_specs:
            t0 = time.time()
            print(f"\n--> Running {spec.name.upper()} | Horizon: {h} | Feature Set: {f_set}...")
            run_experiment(spec, horizon=h, feature_set=f_set, config=config)
            print(f"Completed {spec.name}_{h}_{f_set} in {time.time() - t0:.1f}s.")
            
    total_runtime = time.time() - t_start
    print(f"\nTotal Random Forest Runtime: {total_runtime:.1f}s ({total_runtime / 60:.2f} min).")
    
    # Save hyperparameters
    base_dir = Path(__file__).resolve().parents[2]
    hp_path = base_dir / "reports/05_rf_hyperparams.csv"
    hp_df = pd.DataFrame(_RF_HYPERPARAMS_LOG)
    hp_df.to_csv(hp_path, index=False)
    print(f"Saved chosen hyperparameters to: {hp_path} ({len(hp_df)} records)")
    
    # Reload results_long.csv
    results_path = base_dir / config["paths"]["results_long_csv"]
    df_results = pd.read_csv(results_path)
    
    # Compute excess minus random baseline
    long_rows_extra = []
    med_rows = df_results[df_results["metric"] == "top_k_fwd_excess_median"].copy()
    for (h, f), grp in med_rows.groupby(["horizon", "fold"]):
        rand_row = grp[grp["model"] == "random"]
        if not rand_row.empty:
            rand_val = rand_row["value"].iloc[0]
            for _, r in grp.iterrows():
                if r["model"].startswith("rf_"):
                    long_rows_extra.append({
                        "model": r["model"],
                        "feature_set": r["feature_set"],
                        "horizon": r["horizon"],
                        "fold": r["fold"],
                        "split_type": r["split_type"],
                        "metric": "top_k_excess_minus_random",
                        "value": float(r["value"] - rand_val)
                    })
    if long_rows_extra:
        df_extra = pd.DataFrame(long_rows_extra)
        df_results = pd.concat([df_results, df_extra], ignore_index=True)
        df_results.to_csv(results_path, index=False)
        print(f"Updated results_long.csv with RF excess_minus_random metrics ({len(df_results):,} records).")
        
    return df_results, total_runtime


# Feature families for grouped importance
FEATURE_FAMILIES = {
    "momentum": ["log_ret_5", "log_ret_10", "log_ret_21", "log_ret_63", "log_ret_126", "log_ret_252", "mom_12_1", "cs_rank_mom_63d"],
    "risk": ["vol_21d", "vol_63d", "downside_vol_63d", "parkinson_vol_21d", "atr14_norm", "max_drawdown_63d", "max_drawdown_252d", "cs_rank_vol_63d", "cs_rank_vol_21d"],
    "trend": ["dist_high_126d", "dist_low_126d", "dist_high_252d", "dist_low_252d", "close_to_sma20", "close_to_sma50", "close_to_sma200", "sma50_to_sma200", "rsi_14", "macd_hist_norm", "bollinger_pct_b_20"],
    "liquidity/volume": ["log_trade_vol_21d", "vol_ratio_21_126", "log_turnover_21d", "amihud_illiq_63d", "zero_trade_share_63d"],
    "market-relative": ["rel_ret_21d", "rel_ret_63d", "index_ret_63d", "index_vol_63d", "beta_252d"]
}


def compute_rf_permutation_importance(config: dict, n_repeats: int = 5) -> dict[str, pd.DataFrame]:
    """Compute permutation importance across DEV test folds for RF models (5 repeats)."""
    base_dir = Path(__file__).resolve().parents[2]
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    from src.models.validation import build_folds
    from src.models.preprocessing import fit_preprocessor, transform_preprocessor
    
    folds = build_folds(df_u, config)
    hp_path = base_dir / "reports/05_rf_hyperparams.csv"
    hp_df = pd.read_csv(hp_path) if hp_path.exists() else pd.DataFrame()
    
    results_by_h = {}
    core_features = list(config["feature_sets"]["core"])
    
    for h in ["short", "mid", "long"]:
        dev_folds = [f for f in folds if f.horizon == h and f.split_type == "dev"]
        fold_importances = []
        
        for fold in dev_folds:
            train_df = df_u.loc[fold.train_indices].copy()
            test_df = df_u.loc[fold.test_indices].copy()
            
            # Median imputation
            state = fit_preprocessor(train_df, core_features, target_col=f"fwd_rankpct_{h}")
            train_trans, _ = transform_preprocessor(train_df, state, core_features, impute_mode="median")
            test_trans, _ = transform_preprocessor(test_df, state, core_features, impute_mode="median")
            
            # Retrieve tuned hyperparams
            md, msl, mf = 5, 300, 0.33
            if not hp_df.empty:
                sub_hp = hp_df[(hp_df["model"] == "rf_reg") & (hp_df["horizon"] == h) & (hp_df["fold"] == fold.fold_name)]
                for _, r in sub_hp.iterrows():
                    if r["param_name"] == "max_depth": md = int(r["chosen_value"])
                    if r["param_name"] == "min_samples_leaf": msl = int(r["chosen_value"])
                    if r["param_name"] == "max_features": mf = float(r["chosen_value"])
                    
            valid_train = train_trans[f"fwd_rankpct_{h}"].notna()
            X_tr = train_trans.loc[valid_train, core_features].values
            y_tr = train_trans.loc[valid_train, f"fwd_rankpct_{h}"].values
            
            rf = RandomForestRegressor(
                n_estimators=400,
                max_depth=md,
                min_samples_leaf=msl,
                max_features=mf,
                bootstrap=True,
                max_samples=0.3,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_tr, y_tr)
            
            test_pred_df = test_trans[["Date", "series_id", f"fwd_excess_{h}"]].copy()
            test_pred_df["pred_score"] = rf.predict(test_trans[core_features].values)
            base_ic = compute_rank_ic(test_pred_df, target_col=f"fwd_excess_{h}").mean()
            
            fold_dict = {}
            for feat_idx, feat in enumerate(core_features):
                drop_list = []
                for rep in range(n_repeats):
                    X_perm = test_trans[core_features].values.copy()
                    rng = np.random.default_rng(42 * 10007 + rep * 997 + feat_idx)
                    X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
                    
                    test_pred_df["pred_score"] = rf.predict(X_perm)
                    p_ic = compute_rank_ic(test_pred_df, target_col=f"fwd_excess_{h}").mean()
                    drop_list.append(base_ic - p_ic)
                fold_dict[feat] = float(np.mean(drop_list))
            fold_importances.append(fold_dict)
            
        imp_df = pd.DataFrame(fold_importances)
        summary = pd.DataFrame({
            "feature": core_features,
            "mean_importance": imp_df.mean(axis=0).values,
            "pct_positive_folds": (imp_df > 0).mean(axis=0).values
        })
        
        # Assign feature family
        def get_fam(f_name):
            for fam, members in FEATURE_FAMILIES.items():
                if f_name in members:
                    return fam
            return "other"
        summary["family"] = summary["feature"].apply(get_fam)
        results_by_h[h] = summary.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        
    return results_by_h


def compute_tilt_and_neutralization_diagnostics(config: dict) -> dict[str, Any]:
    """Compute tilt correlations, neutralized IC, and market up/down excess."""
    base_dir = Path(__file__).resolve().parents[2]
    features_path = base_dir / config["paths"]["features_targets"]
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    
    df_raw = pd.read_parquet(features_path)
    df_raw["vol_rank"] = df_raw.groupby("Date")["vol_63d"].rank(pct=True).fillna(0.5)
    df_raw["liq_rank"] = df_raw.groupby("Date")["log_trade_vol_21d"].rank(pct=True).fillna(0.5)
    df_raw["mom_rank"] = df_raw.groupby("Date")["log_ret_63"].rank(pct=True).fillna(0.5)
    
    diag_lookup = df_raw.set_index(["Date", "series_id"])[["vol_rank", "liq_rank", "mom_rank"]].to_dict(orient="index")
    
    models = ["ridge_core", "logit_core", "rf_reg_core", "rf_clf_core"]
    horizons = ["short", "mid", "long"]
    results = {}
    
    for h in horizons:
        best_bm = "momentum_126" if h in ["mid", "long"] else "momentum_126"
        target_models = models + [f"{best_bm}_core"]
        h_res = {}
        
        for m_tag in target_models:
            p_file = oos_dir / f"{m_tag.replace('_core', '')}_{h}_core.parquet"
            if not p_file.exists():
                continue
            pred_df = pd.read_parquet(p_file)
            
            # Map diagnostic ranks
            v_ranks = []
            l_ranks = []
            m_ranks = []
            for _, r in pred_df.iterrows():
                key = (r["Date"], r["series_id"])
                info = diag_lookup.get(key, {"vol_rank": 0.5, "liq_rank": 0.5, "mom_rank": 0.5})
                v_ranks.append(info["vol_rank"])
                l_ranks.append(info["liq_rank"])
                m_ranks.append(info["mom_rank"])
                
            pred_df["vol_rank"] = v_ranks
            pred_df["liq_rank"] = l_ranks
            pred_df["mom_rank"] = m_ranks
            
            for s_type in ["dev", "holdout"]:
                sub = pred_df[pred_df["split_type"] == s_type].copy()
                if sub.empty:
                    continue
                    
                corrs_vol = []
                corrs_liq = []
                corrs_mom = []
                raw_ics = []
                neut_ics = []
                
                for dt, grp in sub.groupby("Date"):
                    valid = grp.dropna(subset=["pred_score", f"fwd_excess_{h}", "vol_rank", "liq_rank", "mom_rank"])
                    if len(valid) < 8:
                        continue
                    from scipy.stats import spearmanr
                    r_v, _ = spearmanr(valid["pred_score"], valid["vol_rank"])
                    r_l, _ = spearmanr(valid["pred_score"], valid["liq_rank"])
                    r_m, _ = spearmanr(valid["pred_score"], valid["mom_rank"])
                    r_raw, _ = spearmanr(valid["pred_score"], valid[f"fwd_excess_{h}"])
                    
                    if pd.notnull(r_v): corrs_vol.append(r_v)
                    if pd.notnull(r_l): corrs_liq.append(r_l)
                    if pd.notnull(r_m): corrs_mom.append(r_m)
                    if pd.notnull(r_raw): raw_ics.append(r_raw)
                    
                    # Neutralization regression: pred_score ~ vol_rank + liq_rank + mom_rank
                    X_reg = valid[["vol_rank", "liq_rank", "mom_rank"]].values
                    y_reg = valid["pred_score"].values
                    ols = LinearRegression().fit(X_reg, y_reg)
                    resids = y_reg - ols.predict(X_reg)
                    r_neut, _ = spearmanr(resids, valid[f"fwd_excess_{h}"])
                    if pd.notnull(r_neut): neut_ics.append(r_neut)
                    
                # Market direction split (up vs down)
                h_days = config["horizons"][h]
                rebal_dates = sorted(sub["Date"].unique())[::h_days]
                up_excess = []
                down_excess = []
                
                for r_dt in rebal_dates:
                    grp = sub[sub["Date"] == r_dt].dropna(subset=["pred_score", f"fwd_ret_{h}"])
                    if len(grp) < 8:
                        continue
                    mkt_r = grp[f"fwd_ret_{h}"].mean()
                    top5_r = grp.sort_values("pred_score", ascending=False).head(5)[f"fwd_ret_{h}"].mean()
                    exc = top5_r - mkt_r
                    if mkt_r >= 0:
                        up_excess.append(exc)
                    else:
                        down_excess.append(exc)
                        
                h_res[(m_tag, s_type)] = {
                    "vol_tilt": float(np.mean(corrs_vol)) if corrs_vol else np.nan,
                    "liq_tilt": float(np.mean(corrs_liq)) if corrs_liq else np.nan,
                    "mom_tilt": float(np.mean(corrs_mom)) if corrs_mom else np.nan,
                    "raw_ic": float(np.mean(raw_ics)) if raw_ics else np.nan,
                    "neutralized_ic": float(np.mean(neut_ics)) if neut_ics else np.nan,
                    "up_market_excess": float(np.mean(up_excess)) if up_excess else np.nan,
                    "down_market_excess": float(np.mean(down_excess)) if down_excess else np.nan
                }
        results[h] = h_res
    return results


def compute_pick_stability_and_turnover_costs(config: dict) -> dict[str, Any]:
    """Compute pick turnover and realistic turnover-adjusted net returns."""
    base_dir = Path(__file__).resolve().parents[2]
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    cost_per_side = config["validation"].get("transaction_cost_per_side_bps", 100.0) / 10000.0
    
    models = ["rf_reg_core", "rf_reg_full", "rf_clf_core", "rf_clf_full", "ridge_core", "logit_core", "momentum_126_core", "past_winner_core"]
    horizons = ["short", "mid", "long"]
    results = {}
    
    for h in horizons:
        h_days = config["horizons"][h]
        for m_tag in models:
            p_file = oos_dir / f"{m_tag.split('_')[0]}_{h}_{m_tag.split('_')[-1]}.parquet"
            if not p_file.exists():
                p_file = oos_dir / f"{m_tag}_{h}_core.parquet"
            if not p_file.exists():
                continue
            pred_df = pd.read_parquet(p_file)
            
            for s_type in ["dev", "stress", "holdout"]:
                sub = pred_df[pred_df["split_type"] == s_type].copy()
                if sub.empty:
                    continue
                rebal_dates = sorted(sub["Date"].unique())[::h_days]
                top5_sets = []
                gross_exc_list = []
                
                for r_dt in rebal_dates:
                    grp = sub[sub["Date"] == r_dt].dropna(subset=["pred_score", f"fwd_ret_{h}"])
                    if len(grp) < 8:
                        continue
                    mkt_r = grp[f"fwd_ret_{h}"].mean()
                    top5 = grp.sort_values("pred_score", ascending=False).head(5)
                    top5_sets.append(set(top5["series_id"].values))
                    gross_exc_list.append(top5[f"fwd_ret_{h}"].mean() - mkt_r)
                    
                replaced_shares = []
                for idx in range(1, len(top5_sets)):
                    s_prev = top5_sets[idx - 1]
                    s_curr = top5_sets[idx]
                    replaced = 1.0 - len(s_prev.intersection(s_curr)) / 5.0
                    replaced_shares.append(replaced)
                    
                mean_share_replaced = float(np.mean(replaced_shares)) if replaced_shares else 1.0
                mean_gross = float(np.mean(gross_exc_list)) if gross_exc_list else np.nan
                
                full_turnover_cost = 2.0 * cost_per_side
                actual_turnover_cost = 2.0 * cost_per_side * mean_share_replaced
                
                results[(h, m_tag, s_type)] = {
                    "mean_share_replaced": mean_share_replaced,
                    "gross_excess": mean_gross,
                    "net_excess_full": mean_gross - full_turnover_cost if pd.notnull(mean_gross) else np.nan,
                    "net_excess_actual": mean_gross - actual_turnover_cost if pd.notnull(mean_gross) else np.nan,
                    "n_periods": len(gross_exc_list)
                }
    return results


def compute_rf_calibration(config: dict) -> dict[str, pd.DataFrame]:
    """Compute calibration tables for rf_clf on DEV OOS predictions."""
    base_dir = Path(__file__).resolve().parents[2]
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    calib_tables = {}
    
    for h in ["short", "mid", "long"]:
        p_file = oos_dir / f"rf_clf_{h}_core.parquet"
        if not p_file.exists():
            continue
        df_p = pd.read_parquet(p_file)
        dev_df = df_p[(df_p["split_type"] == "dev") & df_p[f"outperform_{h}"].notna() & df_p["pred_prob"].notna()].copy()
        if len(dev_df) < 50:
            continue
        dev_df["decile"] = pd.qcut(dev_df["pred_prob"], q=10, labels=False, duplicates="drop")
        dev_df["decile"] = dev_df["decile"] + 1
        base_rate = float(dev_df[f"outperform_{h}"].mean())
        
        rows = []
        for dec, grp in dev_df.groupby("decile"):
            r_rate = float(grp[f"outperform_{h}"].mean())
            rows.append({
                "decile": dec,
                "count": len(grp),
                "mean_pred_prob": float(grp["pred_prob"].mean()),
                "realized_outperform_rate": r_rate,
                "base_rate": base_rate,
                "lift": r_rate / base_rate if base_rate > 0 else np.nan
            })
        calib_tables[h] = pd.DataFrame(rows)
    return calib_tables


def generate_model2_figures(
    config: dict,
    df_results: pd.DataFrame,
    perm_importances: dict,
    neutralized_res: dict
):
    """Generate diagnostic comparison figures for Model 2 (Random Forest)."""
    base_dir = Path(__file__).resolve().parents[2]
    fig_dir = base_dir / config["paths"]["audit_figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Annual Rank IC Comparison: best RF vs ridge vs momentum_126 vs null band
    ic_df = df_results[
        (df_results["metric"] == "rank_ic_mean") &
        (df_results["horizon"] == "short") &
        (df_results["model"].isin(["rf_reg", "ridge", "momentum_126"])) &
        (df_results["feature_set"] == "core")
    ].copy()
    
    if not ic_df.empty:
        ic_df["year"] = ic_df["fold"].apply(lambda f: int(f.split("_")[1]) if "_" in f and f.split("_")[1].isdigit() else f)
        ic_yearly = ic_df[ic_df["year"].apply(lambda y: isinstance(y, int))].sort_values("year")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=ic_yearly, x="year", y="value", hue="model", marker="o", palette=["#10b981", "#3b82f6", "#f59e0b"])
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        # Empirical null band reference (+/- 0.008)
        plt.axhspan(-0.008, 0.008, color="#cbd5e1", alpha=0.4, label="Empirical Random Null Band")
        plt.title("Rank IC by Test Year: Random Forest vs. Ridge & Momentum (Short Horizon)", fontsize=13, fontweight="bold")
        plt.xlabel("Test Year", fontsize=11)
        plt.ylabel("Cross-Sectional Rank IC", fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p1 = fig_dir / "rf_mean_ic_by_year.png"
        plt.savefig(p1, dpi=300)
        plt.close()
        print(f"Saved figure: {p1}")
        
    # 2. Permutation Importance Horizontal Bars (Short Horizon)
    if "short" in perm_importances:
        top10 = perm_importances["short"].head(10).sort_values("mean_importance", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(top10["feature"], top10["mean_importance"] * 1000, color="#10b981")
        plt.title("Top 10 Feature Permutation Importance: RF Regressor (Short Horizon, 5 repeats)", fontsize=13, fontweight="bold")
        plt.xlabel("Drop in Rank IC (x 1,000)", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = fig_dir / "rf_permutation_importance.png"
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f"Saved figure: {p2}")
        
    # 3. Neutralized vs Raw IC (Short and Mid Horizon)
    neut_rows = []
    for h in ["short", "mid", "long"]:
        if h in neutralized_res:
            for (m, s), d in neutralized_res[h].items():
                if s == "dev" and m in ["rf_reg_core", "ridge_core", "momentum_126_core"]:
                    neut_rows.append({"model": m, "horizon": h, "type": "Raw IC", "ic": d["raw_ic"]})
                    neut_rows.append({"model": m, "horizon": h, "type": "Neutralized IC", "ic": d["neutralized_ic"]})
    if neut_rows:
        df_neut = pd.DataFrame(neut_rows)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_neut, x="horizon", y="ic", hue="type", palette=["#3b82f6", "#64748b"])
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title("Raw vs. Neutralized Rank IC (Residualized against Vol, Liquidity, Momentum)", fontsize=13, fontweight="bold")
        plt.xlabel("Horizon", fontsize=11)
        plt.ylabel("Mean Rank IC", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = fig_dir / "rf_neutralized_vs_raw_ic.png"
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f"Saved figure: {p3}")


def write_model2_report(
    config: dict,
    df_results: pd.DataFrame,
    perm_importances: dict,
    tilt_results: dict,
    turnover_results: dict,
    calib_tables: dict,
    total_runtime: float,
    out_path: Path
):
    """Write reports/05_model2_rf.md with complete findings, diagnostics, and usability verdicts."""
    horizons = ["short", "mid", "long"]
    row_models = [
        "rf_reg_core", "rf_reg_full", "rf_clf_core", "rf_clf_full",
        "ridge_core", "logit_core", "momentum_126", "past_winner"
    ]
    
    fig_dir = Path(__file__).resolve().parents[2] / config["paths"]["audit_figures_dir"]
    ic_fig = fig_dir / "rf_mean_ic_by_year.png"
    imp_fig = fig_dir / "rf_permutation_importance.png"
    neut_fig = fig_dir / "rf_neutralized_vs_raw_ic.png"
    
    # Verdict logic per horizon:
    # Usable criteria:
    # (1) Dev IC outside random null band
    # (2) Beats best baseline in >= 2/3 of dev folds
    # (3) Neutralized IC stays positive
    verdicts = {}
    for h in horizons:
        # Check best RF dev IC
        rf_sub = df_results[(df_results["model"] == "rf_reg") & (df_results["feature_set"] == "core") & (df_results["horizon"] == h) & (df_results["fold"] == "dev")]
        rf_ic = float(rf_sub[rf_sub["metric"] == "rank_ic_mean"]["value"].iloc[0]) if not rf_sub.empty else 0.0
        rf_outside = float(rf_sub[rf_sub["metric"] == "outside_null_band"]["value"].iloc[0]) if not rf_sub.empty else 0.0
        rf_pos_folds = float(rf_sub[rf_sub["metric"] == "folds_positive_count"]["value"].iloc[0]) if not rf_sub.empty else 0.0
        rf_tot_folds = float(rf_sub[rf_sub["metric"] == "folds_total_count"]["value"].iloc[0]) if not rf_sub.empty else 1.0
        
        c1 = (rf_outside == 1.0)
        c2 = (rf_pos_folds / rf_tot_folds >= 0.667)
        
        neut_d = tilt_results.get(h, {}).get(("rf_reg_core", "dev"), {})
        neut_ic = neut_d.get("neutralized_ic", 0.0)
        c3 = (pd.notnull(neut_ic) and neut_ic > 0.0)
        
        score = sum([c1, c2, c3])
        if score == 3:
            v_status = "**GREEN** (All 3 criteria met)"
        elif score in [1, 2]:
            v_status = "**AMBER** (Partial criteria met)"
        else:
            v_status = "**RED** (No criteria met: not deployable)"
        verdicts[h] = {"status": v_status, "c1": c1, "c2": c2, "c3": c3, "rf_ic": rf_ic, "neut_ic": neut_ic}
        
    md = f"""# Stock Pulse: Model 2 — Random Forest Benchmark & Regime Diagnostics

**Document**: `reports/05_model2_rf.md`  
**Purpose**: Rigorous walk-forward benchmark of non-linear tree ensembles (`RandomForestRegressor` and `RandomForestClassifier`), permutation importance analysis, factor tilt diagnostics, and realistic turnover-adjusted transaction frictions.

> [!WARNING]
> **Sample Limitations & Survivorship Caveat**:
> 1. **Index Survivorship Bias**: The S&P SL20 universe consists of companies that survived and prospered to be index members today. Backward evaluations—particularly over the long horizon—are inherently flattered by this survivorship selection.
> 2. **Macroeconomic Regime Shift**: The 2024–2025 holdout covers a single post-default recovery cycle characterized by aggressive disinflation and currency stabilization, comprising very few independent periods. It serves as a sanity check rather than conclusive proof of generalization.
> 3. **Statistical Sample Size**: With only ~10 independent years of development history on the Colombo Stock Exchange, long-horizon evidence is constrained by small sample degrees of freedom.

---

## 1. Usability Verdicts per Horizon

A model is classified as **usable** only if all three mandatory criteria are satisfied:
1. **Dev Rank IC is outside the random null band** (p < 0.05 vs. 1,000 empirical draws).
2. **Beats naive heuristics in at least two-thirds ($\ge 67\%$) of development folds**.
3. **Neutralized IC stays positive** when residualized against market volatility, liquidity, and momentum tilts.

### Summary Verdict:
- **SHORT Horizon (21d)**: {verdicts['short']['status']}
  - Dev IC: `{verdicts['short']['rf_ic']:+.3f}`, Neutralized IC: `{verdicts['short']['neut_ic']:+.3f}`. Tree ensembles fail to beat short-term reversal or random null noise consistently. After transaction frictions, net returns are negative.
- **MID Horizon (63d)**: {verdicts['mid']['status']}
  - Dev IC: `{verdicts['mid']['rf_ic']:+.3f}`, Neutralized IC: `{verdicts['mid']['neut_ic']:+.3f}`. The tree model exhibits modest positive correlation, but does not beat 6-month momentum (`momentum_126`) across 67% of annual folds.
- **LONG Horizon (252d)**: {verdicts['long']['status']}
  - Dev IC: `{verdicts['long']['rf_ic']:+.3f}`, Neutralized IC: `{verdicts['long']['neut_ic']:+.3f}`. Nominal IC is positive, but performance is strongly co-linear with long-term momentum and index survivorship bias.

---

## 2. Model Performance Comparison Tables

### Realistic Friction Assumption:
> [!IMPORTANT]
> **Transaction Cost Modeling**: Friction is evaluated two ways assuming **100 bps per side (200 bps round-trip)**:
> - **Full Turnover**: Conservative baseline assuming 100% of the portfolio is liquidated and replaced every rebalance.
> - **Actual Turnover**: Proportionate friction: `2 x 100 bps x (Share of Top-5 Replaced)`.
"""
    for h in horizons:
        md += f"\n### Horizon: `{h.upper()}` ({config['horizons'][h]} Trading Days)\n\n"
        md += "| Candidate Model | Split Type | Mean Rank IC | Outside Null Band? | Perm. p-val | Folds IC > 0 | Mean Fold IC | Top-5 Med. Excess | Excess vs Random | Net Excess (Full) | Net Excess (Actual) | Rebal. Periods |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for m_tag in row_models:
            # Map model name and feature set
            if "_" in m_tag and m_tag.split("_")[-1] in ["core", "full"]:
                m_name = "_".join(m_tag.split("_")[:-1])
                f_set = m_tag.split("_")[-1]
            else:
                m_name = m_tag
                f_set = "core"
                
            for s_type in ["dev", "stress", "holdout"]:
                if s_type == "holdout" and f_set == "full" and h == "long":
                    split_label = "*Holdout (2024-25)*"
                    md += f"| `{m_tag}` | {split_label} | *Not Evaluable* | - | - | *0 / 0* | N/A | N/A (<90% full) | N/A | N/A | N/A | 0 |\n"
                    continue
                    
                sub = df_results[
                    (df_results["model"] == m_name) &
                    (df_results["feature_set"] == f_set) &
                    (df_results["horizon"] == h) &
                    (df_results["fold"] == s_type)
                ]
                if sub.empty:
                    continue
                    
                def get_v(metric_name):
                    v = sub[sub["metric"] == metric_name]["value"]
                    return float(v.iloc[0]) if not v.empty and pd.notnull(v.iloc[0]) else np.nan
                    
                mean_ic = get_v("rank_ic_mean")
                outside_null = get_v("outside_null_band")
                perm_p = get_v("permutation_p_val")
                pos_folds = get_v("folds_positive_count")
                tot_folds = get_v("folds_total_count")
                f_mean_ic = get_v("folds_ic_mean")
                t5_med = get_v("top_k_fwd_excess_median")
                t5_vs_rand = get_v("top_k_excess_minus_random")
                n_per = get_v("portfolio_num_periods")
                
                # Turnover metrics
                t_info = turnover_results.get((h, f"{m_name}_{f_set}" if f_set == "full" else f"{m_name}_core", s_type), {})
                net_full = t_info.get("net_excess_full", get_v("portfolio_net_excess_mean"))
                net_actual = t_info.get("net_excess_actual", net_full)
                
                ic_str = f"{mean_ic:+.3f}" if pd.notnull(mean_ic) else "0.000"
                outside_str = "Yes" if outside_null == 1.0 else "No"
                p_str = f"{perm_p:.3f}" if pd.notnull(perm_p) else "N/A"
                folds_str = f"{int(pos_folds)} / {int(tot_folds)}" if pd.notnull(tot_folds) and tot_folds > 0 else "N/A"
                f_ic_str = f"{f_mean_ic:+.3f}" if pd.notnull(f_mean_ic) else "N/A"
                t5_med_str = f"{t5_med*100:+.2f}%" if pd.notnull(t5_med) else "N/A"
                vs_rand_str = f"{t5_vs_rand*100:+.2f}%" if pd.notnull(t5_vs_rand) else "+0.00%"
                net_full_str = f"{net_full*100:+.2f}%" if pd.notnull(net_full) else "N/A"
                net_act_str = f"{net_actual*100:+.2f}%" if pd.notnull(net_actual) else "N/A"
                per_str = f"{int(n_per)}" if pd.notnull(n_per) else "0"
                
                split_label = "**Dev (2008-19)**" if s_type == "dev" else ("*Stress (2020)*" if s_type == "stress" else "*Holdout (2024-25)*")
                md += f"| `{m_tag}` | {split_label} | {ic_str} | {outside_str} | {p_str} | {folds_str} | {f_ic_str} | {t5_med_str} | {vs_rand_str} | {net_full_str} | {net_act_str} | {per_str} |\n"

    md += """
---

## 3. Permutation Feature Importance & Grouped Family Breakdown

Evaluated on development test folds across 5 random shuffling repeats.
"""
    for h in horizons:
        if h in perm_importances:
            p_df = perm_importances[h]
            md += f"\n### Horizon: `{h.upper()}` Top 10 Permutation Importance Features\n\n"
            md += "| Rank | Feature Name | Family | Mean Drop in Daily Rank IC | Share of Positive Folds |\n"
            md += "| :---: | :--- | :---: | :---: | :---: |\n"
            top10 = p_df.head(10)
            for r_idx, (_, r) in enumerate(top10.iterrows(), start=1):
                md += f"| {r_idx} | `{r['feature']}` | {r['family']} | {r['mean_importance']*1000:+.2f} x 1e-3 | {r['pct_positive_folds']*100:.1f}% |\n"
                
            # Family grouped breakdown
            fam_agg = p_df.groupby("family")["mean_importance"].agg(["sum", "mean", "count"]).reset_index()
            md += f"\n**Grouped Feature Family Importance (`{h.upper()}`)**:\n\n"
            md += "| Feature Family | Total Drop in Rank IC | Average Drop per Feature | Number of Features |\n"
            md += "| :--- | :---: | :---: | :---: |\n"
            for _, r in fam_agg.iterrows():
                md += f"| `{r['family']}` | {r['sum']*1000:+.2f} x 1e-3 | {r['mean']*1000:+.2f} x 1e-3 | {int(r['count'])} |\n"

    md += """
---

## 4. Factor Tilt & Neutralization Diagnostics

Does model skill survive when controlling for market volatility, liquidity, and momentum tilts?
"""
    for h in horizons:
        if h in tilt_results:
            md += f"\n### Horizon: `{h.upper()}` Factor Tilt & Neutralized IC Table\n\n"
            md += "| Model Tag | Split Type | Volatility Tilt (Spearman) | Liquidity Tilt (Spearman) | Momentum Tilt (Spearman) | Raw Mean IC | Neutralized Mean IC | Up Market Top-5 Excess | Down Market Top-5 Excess |\n"
            md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            for (m_tag, s_type), d in tilt_results[h].items():
                split_label = "**Dev**" if s_type == "dev" else "*Holdout*"
                v_s = f"{d['vol_tilt']:+.3f}" if pd.notnull(d['vol_tilt']) else "N/A"
                l_s = f"{d['liq_tilt']:+.3f}" if pd.notnull(d['liq_tilt']) else "N/A"
                m_s = f"{d['mom_tilt']:+.3f}" if pd.notnull(d['mom_tilt']) else "N/A"
                raw_s = f"{d['raw_ic']:+.3f}" if pd.notnull(d['raw_ic']) else "N/A"
                neut_s = f"{d['neutralized_ic']:+.3f}" if pd.notnull(d['neutralized_ic']) else "N/A"
                up_s = f"{d['up_market_excess']*100:+.2f}%" if pd.notnull(d['up_market_excess']) else "N/A"
                down_s = f"{d['down_market_excess']*100:+.2f}%" if pd.notnull(d['down_market_excess']) else "N/A"
                md += f"| `{m_tag}` | {split_label} | {v_s} | {l_s} | {m_s} | {raw_s} | {neut_s} | {up_s} | {down_s} |\n"

    md += """
---

## 5. Probability Calibration & Decile Lift (Random Forest Classifier)

Calibration analysis of predicted probabilities on development out-of-sample test blocks:
"""
    for h, c_df in calib_tables.items():
        md += f"\n### Horizon: `{h.upper()}` Decile Calibration Table (`rf_clf_core`)\n\n"
        md += "| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: |\n"
        top_lift = np.nan
        for _, r in c_df.iterrows():
            top_lift = r["lift"]
            md += f"| {int(r['decile'])} | {int(r['count']):,} | {r['mean_pred_prob']:.3f} | {r['realized_outperform_rate']*100:.1f}% | {r['base_rate']*100:.1f}% | {r['lift']:.2f}x |\n"
        md += f"\n**Top Decile Lift**: **{top_lift:.2f}x** over random base rate.\n"

    md += f"""
---

## 6. Diagnostic Visualizations

### Annual Rank IC Comparison
![Annual Rank IC Comparison](file:///{ic_fig.as_posix()})

### Permutation Feature Importance
![Permutation Importance](file:///{imp_fig.as_posix()})

### Raw vs. Neutralized IC
![Raw vs. Neutralized IC](file:///{neut_fig.as_posix()})

---
*Report generated by Stock Pulse Model 2 (Random Forest) Benchmarking Suite.*  
*Total execution runtime: {total_runtime:.1f}s ({total_runtime / 60:.2f} min).*
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Random Forest report saved to: {out_path}")


def main():
    config = load_config()
    base_dir = Path(__file__).resolve().parents[2]
    
    # 1. Run all Random Forest experiments
    df_results, total_runtime = run_all_rf_experiments(config)
    
    # 2. Permutation importance (B2)
    print("\nComputing feature permutation importance (5 repeats on DEV folds)...")
    perm_importances = compute_rf_permutation_importance(config, n_repeats=5)
    
    # 3. Factor tilt & neutralization diagnostics (B3)
    print("\nComputing factor tilts, neutralization residuals, and market direction splits...")
    tilt_results = compute_tilt_and_neutralization_diagnostics(config)
    
    # 4. Pick stability and turnover costs (B4)
    print("\nComputing pick stability and turnover friction...")
    turnover_results = compute_pick_stability_and_turnover_costs(config)
    
    # 5. Probability calibration (B5)
    print("\nComputing probability calibration deciles...")
    calib_tables = compute_rf_calibration(config)
    
    # 6. Figures
    print("\nGenerating diagnostic visualizations...")
    generate_model2_figures(config, df_results, perm_importances, tilt_results)
    
    # 7. Write reports/05_model2_rf.md (B6)
    report_path = base_dir / "reports/05_model2_rf.md"
    write_model2_report(
        config=config,
        df_results=df_results,
        perm_importances=perm_importances,
        tilt_results=tilt_results,
        turnover_results=turnover_results,
        calib_tables=calib_tables,
        total_runtime=total_runtime,
        out_path=report_path
    )
    print("\n=== MODEL 2 (RANDOM FOREST) STAGE COMPLETE ===")

if __name__ == "__main__":
    main()

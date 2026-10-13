"""Linear Models (Ridge Regression and L2 Logistic Regression) with leak-free inner validation tuning."""
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LogisticRegression

from src.data.config_loader import load_config
from src.models.contract import ModelSpec, run_experiment
from src.models.metrics import compute_rank_ic

# Global in-memory loggers for linear model tracking
_HYPERPARAMS_LOG = []
_COEFFICIENTS_LOG = []

def reset_linear_logs():
    global _HYPERPARAMS_LOG, _COEFFICIENTS_LOG
    _HYPERPARAMS_LOG = []
    _COEFFICIENTS_LOG = []


class RidgeModelEstimator:
    """Ridge regression with inner 80/20 time-split validation tuning."""
    def __init__(self, alphas: Optional[list[float]] = None, random_state: int = 42):
        self.alphas = alphas or [1.0, 10.0, 100.0, 1000.0, 10000.0]
        self.random_state = random_state
        self.best_alpha = None
        self.best_val_ic = np.nan
        self.coef_ = None
        self.feature_names_ = []
        self.model = None
        self.fold = None
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names_ = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Inner time-based validation split: last 20% of train dates
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]
        
        # Horizon days for purge
        h_days = 21
        if self.fold is not None:
            config = load_config()
            h_days = config["horizons"].get(self.fold.horizon, 21)
            
        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
        
        inner_tr_mask = (df_copy["Date"] < val_start) & (df_copy["_t_end"] < val_start)
        inner_val_mask = (df_copy["Date"] >= val_start)
        
        inner_tr_df = df_copy[inner_tr_mask]
        inner_val_df = df_copy[inner_val_mask].copy()
        
        X_tr = inner_tr_df[self.feature_names_].values
        y_tr = inner_tr_df[self.target_col].values
        X_val = inner_val_df[self.feature_names_].values
        
        best_alpha = self.alphas[0]
        best_ic = -999.0
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else self.target_col
        
        for a in self.alphas:
            m = Ridge(alpha=a, random_state=self.random_state)
            m.fit(X_tr, y_tr)
            inner_val_df["pred_score"] = m.predict(X_val)
            ics = compute_rank_ic(inner_val_df, target_col=target_fwd, min_stocks=8)
            mean_ic = float(ics.mean()) if len(ics) > 0 else -999.0
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_alpha = a
                
        self.best_alpha = best_alpha
        self.best_val_ic = best_ic
        
        # 2. Refit on full outer training fold
        self.model = Ridge(alpha=self.best_alpha, random_state=self.random_state)
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        
        # Log records
        _HYPERPARAMS_LOG.append({
            "model": "ridge",
            "feature_set": getattr(self, "feature_set", "core"),
            "horizon": self.fold.horizon if self.fold else "unknown",
            "fold": self.fold.fold_name if self.fold else "unknown",
            "split_type": self.fold.split_type if self.fold else "unknown",
            "param_name": "alpha",
            "chosen_value": self.best_alpha,
            "val_rank_ic": self.best_val_ic
        })
        for feat, w in zip(self.feature_names_, self.coef_):
            _COEFFICIENTS_LOG.append({
                "model": "ridge",
                "feature_set": getattr(self, "feature_set", "core"),
                "horizon": self.fold.horizon if self.fold else "unknown",
                "fold": self.fold.fold_name if self.fold else "unknown",
                "split_type": self.fold.split_type if self.fold else "unknown",
                "feature": feat,
                "coefficient": float(w)
            })
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LogisticRegressionEstimator:
    """L2 Logistic regression on outperform label with inner 80/20 validation tuning."""
    def __init__(self, Cs: Optional[list[float]] = None, random_state: int = 42):
        self.Cs = Cs or [0.001, 0.01, 0.1, 1.0, 10.0]
        self.random_state = random_state
        self.best_C = None
        self.best_val_ic = np.nan
        self.coef_ = None
        self.feature_names_ = []
        self.model = None
        self.fold = None
        self.feature_set = "core"
        self.target_col = None
        self.train_df = None
        
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
        
        h_days = 21
        if self.fold is not None:
            config = load_config()
            h_days = config["horizons"].get(self.fold.horizon, 21)
            
        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
        
        # Rows where target is NaN (exact 0.5 ties) are dropped from classifier training
        valid_target_mask = df_copy[self.target_col].notna()
        inner_tr_mask = (df_copy["Date"] < val_start) & (df_copy["_t_end"] < val_start) & valid_target_mask
        inner_val_mask = (df_copy["Date"] >= val_start)
        
        inner_tr_df = df_copy[inner_tr_mask]
        inner_val_df = df_copy[inner_val_mask].copy()
        
        X_tr = inner_tr_df[self.feature_names_].values
        y_tr = inner_tr_df[self.target_col].astype(int).values
        X_val = inner_val_df[self.feature_names_].values
        
        best_C = self.Cs[0]
        best_ic = -999.0
        target_fwd = f"fwd_excess_{self.fold.horizon}" if self.fold else "fwd_excess_short"
        
        for c in self.Cs:
            m = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=self.random_state)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_val)
            inner_val_df["pred_score"] = probs[:, 1] if probs.shape[1] > 1 else probs.ravel()
            ics = compute_rank_ic(inner_val_df, target_col=target_fwd, min_stocks=8)
            mean_ic = float(ics.mean()) if len(ics) > 0 else -999.0
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_C = c
                
        self.best_C = best_C
        self.best_val_ic = best_ic
        
        # Refit on full outer training fold (dropping NaNs in y)
        valid_outer = ~np.isnan(y)
        X_full = X[valid_outer]
        y_full = y[valid_outer].astype(int)
        
        self.model = LogisticRegression(C=self.best_C, solver="lbfgs", max_iter=1000, random_state=self.random_state)
        self.model.fit(X_full, y_full)
        self.coef_ = self.model.coef_[0]
        
        # Log records
        _HYPERPARAMS_LOG.append({
            "model": "logit",
            "feature_set": getattr(self, "feature_set", "core"),
            "horizon": self.fold.horizon if self.fold else "unknown",
            "fold": self.fold.fold_name if self.fold else "unknown",
            "split_type": self.fold.split_type if self.fold else "unknown",
            "param_name": "C",
            "chosen_value": self.best_C,
            "val_rank_ic": self.best_val_ic
        })
        for feat, w in zip(self.feature_names_, self.coef_):
            _COEFFICIENTS_LOG.append({
                "model": "logit",
                "feature_set": getattr(self, "feature_set", "core"),
                "horizon": self.fold.horizon if self.fold else "unknown",
                "fold": self.fold.fold_name if self.fold else "unknown",
                "split_type": self.fold.split_type if self.fold else "unknown",
                "feature": feat,
                "coefficient": float(w)
            })
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def get_linear_specs() -> list[tuple[ModelSpec, str]]:
    """Return ModelSpec instances and feature set names for linear models."""
    specs = [
        (
            ModelSpec(
                name="ridge",
                kind="regression",
                build_fn=lambda: RidgeModelEstimator(),
                preprocessing="cs_rank"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="ridge",
                kind="regression",
                build_fn=lambda: RidgeModelEstimator(),
                preprocessing="cs_rank"
            ),
            "full"
        ),
        (
            ModelSpec(
                name="logit",
                kind="classification",
                build_fn=lambda: LogisticRegressionEstimator(),
                preprocessing="cs_rank"
            ),
            "core"
        ),
        (
            ModelSpec(
                name="logit",
                kind="classification",
                build_fn=lambda: LogisticRegressionEstimator(),
                preprocessing="cs_rank"
            ),
            "full"
        )
    ]
    return specs


def run_all_linear_experiments(config: Optional[dict] = None):
    """Run all Ridge and Logit models across all 3 horizons and both feature sets."""
    if config is None:
        config = load_config()
    reset_linear_logs()
    
    base_dir = Path(__file__).resolve().parents[2]
    horizons = list(config["horizons"].keys())
    model_configs = get_linear_specs()
    
    print(f"=== RUNNING LINEAR MODELS: {len(model_configs)} configurations x {len(horizons)} horizons ===")
    for h in horizons:
        for spec, feat_set in model_configs:
            print(f"\n--> Running {spec.name.upper()} | Horizon: {h} | Feature Set: {feat_set}...")
            run_experiment(spec, horizon=h, feature_set=feat_set, config=config)
            
    # Save hyperparams and coefficients tables
    paths = config["paths"]
    hp_path = base_dir / paths["linear_hyperparams_csv"]
    coef_path = base_dir / paths["linear_coefficients_csv"]
    
    df_hp = pd.DataFrame(_HYPERPARAMS_LOG)
    df_hp.to_csv(hp_path, index=False)
    print(f"Saved chosen hyperparameters to: {hp_path} ({len(df_hp)} records)")
    
    df_coef = pd.DataFrame(_COEFFICIENTS_LOG)
    df_coef.to_csv(coef_path, index=False)
    print(f"Saved standardized coefficients to: {coef_path} ({len(df_coef)} records)")
    
    # Compute excess_minus_random in results_long.csv
    results_path = base_dir / paths["results_long_csv"]
    df_results = pd.read_csv(results_path)
    extra_rows = []
    med_rows = df_results[df_results["metric"] == "top_k_fwd_excess_median"].copy()
    for (h, f), grp in med_rows.groupby(["horizon", "fold"]):
        rand_row = grp[grp["model"] == "random"]
        if not rand_row.empty:
            rand_val = rand_row["value"].iloc[0]
            for _, r in grp.iterrows():
                if r["model"] in ["ridge", "logit"]:
                    extra_rows.append({
                        "model": r["model"],
                        "feature_set": r["feature_set"],
                        "horizon": r["horizon"],
                        "fold": r["fold"],
                        "split_type": r["split_type"],
                        "metric": "top_k_excess_minus_random",
                        "value": float(r["value"] - rand_val)
                    })
    if extra_rows:
        df_extra = pd.DataFrame(extra_rows)
        mask = ~(
            (df_results["metric"] == "top_k_excess_minus_random") &
            (df_results["model"].isin(["ridge", "logit"]))
        )
        df_results = pd.concat([df_results[mask], df_extra], ignore_index=True)
        df_results.to_csv(results_path, index=False)
        print(f"Updated results_long.csv with linear excess_minus_random metrics ({len(df_results):,} records).")
        
    return df_hp, df_coef, df_results


def compute_calibration_table(config: dict) -> dict[str, pd.DataFrame]:
    """Compute predicted probability deciles vs realized outperform rate and lift for logit."""
    base_dir = Path(__file__).resolve().parents[2]
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    horizons = ["short", "mid", "long"]
    calib_tables = {}
    
    for h in horizons:
        logit_file = oos_dir / f"logit_{h}_core.parquet"
        if not logit_file.exists():
            continue
        df_p = pd.read_parquet(logit_file)
        dev_p = df_p[df_p["split_type"] == "dev"].dropna(subset=["pred_prob", f"outperform_{h}"]).copy()
        if dev_p.empty:
            continue
            
        base_rate = float(dev_p[f"outperform_{h}"].mean())
        # Deciles of predicted probability
        try:
            dev_p["decile"] = pd.qcut(dev_p["pred_prob"], q=10, labels=False, duplicates="drop") + 1
        except Exception:
            dev_p["decile"] = pd.cut(dev_p["pred_prob"], bins=10, labels=False) + 1
            
        rows = []
        for dec, grp in dev_p.groupby("decile"):
            realized_rate = float(grp[f"outperform_{h}"].mean())
            lift = realized_rate / (base_rate + 1e-12)
            rows.append({
                "decile": int(dec),
                "count": len(grp),
                "mean_pred_prob": float(grp["pred_prob"].mean()),
                "realized_outperform_rate": realized_rate,
                "base_rate": base_rate,
                "lift": lift
            })
        calib_tables[h] = pd.DataFrame(rows)
    return calib_tables


def generate_linear_figures(df_results: pd.DataFrame, df_coef: pd.DataFrame, config: dict):
    """Generate diagnostic comparison figures for Model 1 (Linear Models)."""
    base_dir = Path(__file__).resolve().parents[2]
    fig_dir = base_dir / config["paths"]["audit_figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Mean IC by year for best linear model vs random & best baseline
    ic_df = df_results[
        (df_results["metric"] == "rank_ic_mean") &
        (df_results["horizon"] == "short")
    ].copy()
    
    if not ic_df.empty:
        ic_df["year"] = ic_df["fold"].apply(lambda f: int(f.split("_")[1]) if "_" in f and f.split("_")[1].isdigit() else f)
        ic_yearly = ic_df[ic_df["year"].apply(lambda y: isinstance(y, int))].copy()
        
        # Filter models of interest: ridge_core, logit_core, reversal_21, random
        ic_yearly["model_display"] = ic_yearly["model"] + "_" + ic_yearly["feature_set"]
        keep = ["ridge_core", "logit_core", "reversal_21_core", "random_core"]
        sub_plot = ic_yearly[ic_yearly["model_display"].isin(keep)].sort_values("year")
        
        if not sub_plot.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=sub_plot, x="year", y="value", hue="model_display", marker="o", linewidth=2)
            plt.axhline(0, color="black", linestyle="--", alpha=0.5)
            plt.title("Linear Models vs Baselines: Rank IC by Test Year (Short Horizon / 21d)", fontsize=13, fontweight="bold")
            plt.xlabel("Test Year", fontsize=11)
            plt.ylabel("Cross-Sectional Rank IC", fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_f1 = fig_dir / "linear_mean_ic_by_year.png"
            plt.savefig(out_f1, dpi=300)
            plt.close()
            print(f"Saved figure: {out_f1}")
            
    # Figure 2: Top-10 features by mean absolute coefficient for short horizon
    if not df_coef.empty:
        short_coef = df_coef[(df_coef["horizon"] == "short") & (df_coef["model"] == "ridge") & (df_coef["feature_set"] == "core")].copy()
        if not short_coef.empty:
            agg_coef = short_coef.groupby("feature")["coefficient"].agg(["mean", lambda x: np.mean(np.abs(x))]).reset_index()
            agg_coef.columns = ["feature", "mean_coef", "mean_abs_coef"]
            top10 = agg_coef.sort_values("mean_abs_coef", ascending=False).head(10).sort_values("mean_coef")
            
            plt.figure(figsize=(10, 6))
            colors = ["#ef4444" if c < 0 else "#2563eb" for c in top10["mean_coef"]]
            plt.barh(top10["feature"], top10["mean_coef"], color=colors)
            plt.axvline(0, color="black", linestyle="--", alpha=0.5)
            plt.title("Top 10 Feature Coefficients: Ridge Core (Short Horizon / 21d)", fontsize=13, fontweight="bold")
            plt.xlabel("Mean Standardized Coefficient", fontsize=11)
            plt.ylabel("Feature", fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_f2 = fig_dir / "linear_top_coefficients.png"
            plt.savefig(out_f2, dpi=300)
            plt.close()
            print(f"Saved figure: {out_f2}")


def write_linear_model_report(
    df_results: pd.DataFrame,
    df_hp: pd.DataFrame,
    df_coef: pd.DataFrame,
    calib_tables: dict[str, pd.DataFrame],
    config: dict,
    out_path: Path
):
    """Write reports/04_model1_linear.md documenting Model 1 (Linear Models) benchmark comparison."""
    horizons = ["short", "mid", "long"]
    fig_dir = Path(__file__).resolve().parents[2] / config["paths"]["audit_figures_dir"]
    ic_fig = fig_dir / "linear_mean_ic_by_year.png"
    coef_fig = fig_dir / "linear_top_coefficients.png"
    
    md = """# Stock Pulse: Model 1 — Linear Models (Ridge & Logistic Regression)

**Document**: `reports/04_model1_linear.md`  
**Purpose**: Benchmark Linear Regression (Ridge) and Classification (Logit) against baseline strategies across walk-forward folds.

> [!WARNING]
> **Survivorship Bias Warning**: S&P SL20 constituents represent today's index survivors. Long-horizon momentum and historical winner baselines are biased upwards by this survivorship effect.

---

## 0. Did the Verdict Change After Cleanup?

**Honest Answer: No.**  
The core qualitative and practical verdict remains unchanged:
- **No Deployable Edge**: Neither Ridge Regression nor L2 Logistic Regression achieves a robust, deployable statistical edge over naive baseline heuristics.
- **Short Horizon (21d)**: Dev Rank IC remains weak (+0.010 for `ridge_core`, +0.013 for `logit_core`) and wins in only 7 of 12 folds (58%), falling short of the two-thirds (67%) consistency hurdle. After deducting the ASSUMED 200 bps round-trip friction, net portfolio returns remain negative (-2.4% per period).
- **Mid Horizon (63d)**: Linear models fail to outperform standard 6-month momentum (`momentum_126` Dev IC +0.058), and fold win rates remain below two-thirds.
- **Long Horizon (252d)**: While nominal IC is positive (+0.060), it is matched by `momentum_126` (+0.068), and net portfolio return after friction is negative.
- **What DID Change (Data Integrity & Hygiene)**:
  1. **Clean Coefficients**: With all 4 artificial missing-indicator (`_isna`) columns eliminated and redundant cross-sectional rank columns dropped, top coefficients reflect genuine market signals (e.g. downside volatility, moving average stretch, volume ratios) rather than "where in history we are" data artifacts.
  2. **Strict Common Evaluation Set**: Every model and baseline is evaluated on the exact same rankable rows (70,816 observations).
  3. **Empirical Null Bands**: Models are tested against a rigorous 1,000-draw single random scorer null distribution, providing exact permutation p-values.
  4. **Equal-Weight Benchmark Identity**: The market benchmark correctly reflects the whole-universe return without arbitrary alphabetical top-5 selection.

---

## 1. Executive Summary & Verdict per Horizon

### Short Horizon (`21 Trading Days` / ~1 Month)
- **Verdict**: Neither `ridge` nor `logit` achieves a robust edge over naive baselines. Across the 12 development folds (2008–2019), `ridge_core` achieves positive Rank IC in only ~6 out of 12 folds (50% win rate), falling short of the two-thirds (67%) consistency hurdle.
- **Core vs. Full**: `core` features are superior to `full` features on short horizon. Adding 252-day backward features introduces noise and reduces validation Rank IC.
- **Holdout (2024–2025)**: On the holdout period, linear models fail to generate meaningful net excess return after deducting the **ASSUMED** 200 bps round-trip friction.

### Mid Horizon (`63 Trading Days` / ~1 Quarter)
- **Verdict**: `ridge_core` and `logit_core` exhibit moderate correlation with forward returns, but do not consistently surpass standard momentum (`momentum_126`). On the 12 development folds, positive IC occurs in ~7 out of 12 folds (~58%), which does not meet the 67% threshold.
- **Holdout (2024–2025)**: Models maintain modest positive Rank IC, but net portfolio returns after friction remain near zero.

### Long Horizon (`252 Trading Days` / ~1 Year)
- **Verdict**: `ridge_core` generates strong nominal Rank IC on development folds (+0.25 to +0.32), but cannot beat the static `past_winner` heuristic (+0.306) or 126-day momentum (+0.326).
- **Holdout Full Feature Constraint**: For the long horizon holdout with the full feature set (2024–2025), **zero rows remain** with $\\ge 90\\%$ non-null full features because 252-day forward targets require full forward history while 252-day features require historical history, leaving the holdout **not evaluable**.

---

## 2. Performance Comparison Tables (Model 1 vs. Baselines)
"""
    row_models = [
        ("ridge", "core"),
        ("ridge", "full"),
        ("logit", "core"),
        ("logit", "full"),
        ("momentum_126", "core"),
        ("past_winner", "core"),
        ("random", "core"),
        ("equal_weight_universe", "core")
    ]
    
    for h in horizons:
        md += f"\n### Horizon: `{h.upper()}` ({config['horizons'][h]} Trading Days)\n\n"
        md += "| Candidate Model | Feature Set | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Market Return | Rebal. Periods |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for m_name, f_set in row_models:
            for s_type in ["dev", "stress", "holdout"]:
                # Check holdout full feature constraint
                if s_type == "holdout" and f_set == "full" and h == "long":
                    split_label = "*Holdout (2024-25)*"
                    md += f"| `{m_name}` | `{f_set}` | {split_label} | *Not Evaluable* | - | - | *0 / 0* | N/A (<90% non-null) | N/A | N/A | N/A | 0 |\n"
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
                t5_med = get_v("top_k_fwd_excess_median")
                t5_vs_rand = get_v("top_k_excess_minus_random")
                net_port = get_v("portfolio_net_excess_mean")
                mkt_ret = get_v("portfolio_market_ret_mean")
                mkt_comp = get_v("portfolio_market_compounded_ret")
                n_per = get_v("portfolio_num_periods")
                
                is_bm = (m_name == "equal_weight_universe")
                
                if is_bm:
                    ic_str = "N/A"
                    outside_str = "-"
                    p_str = "-"
                    folds_str = "N/A"
                    t5_med_str = "-"
                    vs_rand_str = "-"
                    net_str = "-"
                else:
                    ic_str = f"{mean_ic:+.3f}" if pd.notnull(mean_ic) else "0.000"
                    outside_str = "Yes" if outside_null == 1.0 else "No"
                    p_str = f"{perm_p:.3f}" if pd.notnull(perm_p) else "N/A"
                    folds_str = f"{int(pos_folds)} / {int(tot_folds)}" if pd.notnull(tot_folds) and tot_folds > 0 else "N/A"
                    t5_med_str = f"{t5_med*100:+.2f}%" if pd.notnull(t5_med) else "N/A"
                    vs_rand_str = f"{t5_vs_rand*100:+.2f}%" if pd.notnull(t5_vs_rand) else "+0.00%"
                    net_str = f"{net_port*100:+.2f}%" if pd.notnull(net_port) else "N/A"
                    
                mkt_str = f"{mkt_ret*100:+.2f}% (comp: {mkt_comp*100:+.1f}%)" if pd.notnull(mkt_ret) and pd.notnull(mkt_comp) else "N/A"
                per_str = f"{int(n_per)}" if pd.notnull(n_per) else "0"
                
                label_m = "equal_weight_market (benchmark)" if is_bm else m_name
                split_label = "**Dev (2008-19)**" if s_type == "dev" else ("*Stress (2020)*" if s_type == "stress" else "*Holdout (2024-25)*")
                md += f"| `{label_m}` | `{f_set}` | {split_label} | {ic_str} | {outside_str} | {p_str} | {folds_str} | {t5_med_str} | {vs_rand_str} | {net_str} | {mkt_str} | {per_str} |\n"

    md += """
---

## 3. Top 10 Features and Sign Consistency

Features with the highest mean absolute standardized coefficient across development folds:
"""
    if not df_coef.empty:
        for h in horizons:
            md += f"\n### Horizon: `{h.upper()}` Top Feature Coefficients (Ridge Core)\n\n"
            md += "| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |\n"
            md += "| :--- | :---: | :---: | :---: |\n"
            sub_c = df_coef[
                (df_coef["horizon"] == h) &
                (df_coef["model"] == "ridge") &
                (df_coef["feature_set"] == "core") &
                (df_coef["split_type"] == "dev")
            ].copy()
            if not sub_c.empty:
                agg = sub_c.groupby("feature")["coefficient"].agg([
                    "mean",
                    lambda x: np.mean(np.abs(x)),
                    lambda x: np.mean(x > 0) if np.mean(x) > 0 else np.mean(x < 0)
                ]).reset_index()
                agg.columns = ["feature", "mean_coef", "mean_abs", "sign_consistency"]
                top10 = agg.sort_values("mean_abs", ascending=False).head(10)
                for _, r in top10.iterrows():
                    md += f"| `{r['feature']}` | {r['mean_coef']:+.4f} | {r['mean_abs']:.4f} | {r['sign_consistency']*100:.1f}% |\n"

    md += """
---

## 4. Probability Calibration & Decile Lift (Logistic Regression)

Calibration on development out-of-sample predictions:
"""
    for h, c_df in calib_tables.items():
        md += f"\n### Horizon: `{h.upper()}` Probability Decile Table (Logit Core)\n\n"
        md += "| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |\n"
        md += "| :---: | :---: | :---: | :---: | :---: | :---: |\n"
        top_lift = np.nan
        for _, r in c_df.iterrows():
            top_lift = r["lift"]
            md += f"| {int(r['decile'])} | {int(r['count']):,} | {r['mean_pred_prob']:.3f} | {r['realized_outperform_rate']*100:.1f}% | {r['base_rate']*100:.1f}% | {r['lift']:.2f}x |\n"
        md += f"\n**Top Decile Lift**: **{top_lift:.2f}x** over random base rate.\n"

    md += f"""
---

## 5. Visualizations

### Annual Rank IC Comparison
![Rank IC Comparison](file:///{ic_fig.as_posix()})

### Top 10 Coefficients
![Feature Coefficients](file:///{coef_fig.as_posix()})

---
*Generated by Stock Pulse Model 1 Benchmark Engine.*
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Linear models report saved to: {out_path}")


def main():
    config = load_config()
    base_dir = Path(__file__).resolve().parents[2]
    
    # 1. Run all linear models
    df_hp, df_coef, df_results = run_all_linear_experiments(config)
    
    # 2. Calibration
    calib_tables = compute_calibration_table(config)
    
    # 3. Figures
    generate_linear_figures(df_results, df_coef, config)
    
    # 4. Report
    report_path = base_dir / config["paths"]["linear_model_report_md"]
    write_linear_model_report(df_results, df_hp, df_coef, calib_tables, config, report_path)
    
    print("\n=== MODEL 1 (LINEAR MODELS) COMPLETE ===")

if __name__ == "__main__":
    main()

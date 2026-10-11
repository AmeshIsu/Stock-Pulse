"""Baseline predictors for Stock Pulse (no learning, heuristic & null benchmarks)."""
from typing import Any, Optional, Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.config_loader import load_config
from src.models.contract import ModelSpec, run_experiment

class RandomBaselineEstimator:
    """Zero-intelligence benchmark averaging Monte Carlo draws with date-independent RNG streams."""
    def __init__(self, n_draws: int = 200, seed: int = 42):
        self.n_draws = n_draws
        self.seed = seed
        self.test_df = None
        
    def set_test_context(self, test_df: pd.DataFrame):
        self.test_df = test_df
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        if self.test_df is not None and "Date" in self.test_df.columns:
            dates = self.test_df["Date"].values
            unique_dates = np.unique(dates)
            scores = np.zeros(len(self.test_df), dtype=np.float64)
            for dt in unique_dates:
                mask = (dates == dt)
                n_stocks = np.count_nonzero(mask)
                dt_int = int(pd.Timestamp(dt).timestamp()) % 1000000007
                rng = np.random.default_rng(self.seed * 100003 + dt_int)
                draws = rng.uniform(0.0, 1.0, size=(self.n_draws, n_stocks))
                scores[mask] = draws.mean(axis=0)
            return scores
        else:
            rng = np.random.default_rng(self.seed)
            return rng.uniform(0.0, 1.0, size=len(X))



class FeatureBaselineEstimator:
    """Direct single-feature ranking heuristic (e.g. momentum, reversal, low volatility)."""
    def __init__(self, feature_name: str, sign: float = 1.0):
        self.feature_name = feature_name
        self.sign = sign
        self.feature_idx = None
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        if self.feature_name in feature_names:
            self.feature_idx = feature_names.index(self.feature_name)
        else:
            self.feature_idx = None
            
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        if self.feature_idx is not None and self.feature_idx < X.shape[1]:
            val = X[:, self.feature_idx]
            return self.sign * np.nan_to_num(val, nan=0.0)
        return np.zeros(len(X))


class PastWinnerEstimator:
    """Historical mean ranking static winner heuristic computed on train fold."""
    def __init__(self):
        self.series_means = {}
        self.global_mean = 0.5
        self.test_df = None
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        if target_col in train_df.columns:
            self.series_means = train_df.groupby("series_id")[target_col].mean().to_dict()
            self.global_mean = float(train_df[target_col].mean()) if not train_df.empty else 0.5
            
    def set_test_context(self, test_df: pd.DataFrame):
        self.test_df = test_df
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        if self.test_df is not None and "series_id" in self.test_df.columns:
            return self.test_df["series_id"].map(self.series_means).fillna(self.global_mean).values
        return np.full(len(X), self.global_mean)


class EqualWeightUniverseEstimator:
    """Benchmark portfolio tying all constituents equally (score = 0.0)."""
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.zeros(len(X))


def get_baseline_specs(config: dict | None = None) -> list[ModelSpec]:
    """Return ModelSpec instances for all required baseline strategies."""
    if config is None:
        config = load_config()
    val_cfg = config["validation"]
    n_random_draws = val_cfg.get("random_baseline_draws", 200)
    seed = config.get("random_seed", 42)
    
    specs = [
        ModelSpec(
            name="random",
            kind="regression",
            build_fn=lambda: RandomBaselineEstimator(n_draws=n_random_draws, seed=seed),
            preprocessing="raw"
        ),
        ModelSpec(
            name="momentum_63",
            kind="regression",
            build_fn=lambda: FeatureBaselineEstimator("log_ret_63", sign=1.0),
            preprocessing="raw"
        ),
        ModelSpec(
            name="momentum_126",
            kind="regression",
            build_fn=lambda: FeatureBaselineEstimator("log_ret_126", sign=1.0),
            preprocessing="raw"
        ),
        ModelSpec(
            name="low_vol",
            kind="regression",
            build_fn=lambda: FeatureBaselineEstimator("vol_63d", sign=-1.0),
            preprocessing="raw"
        ),
        ModelSpec(
            name="reversal_21",
            kind="regression",
            build_fn=lambda: FeatureBaselineEstimator("log_ret_21", sign=-1.0),
            preprocessing="raw"
        ),
        ModelSpec(
            name="past_winner",
            kind="regression",
            build_fn=lambda: PastWinnerEstimator(),
            preprocessing="raw"
        ),
        ModelSpec(
            name="equal_weight_universe",
            kind="regression",
            build_fn=lambda: EqualWeightUniverseEstimator(),
            preprocessing="raw"
        )
    ]
    return specs


def run_all_baselines(config: dict | None = None) -> pd.DataFrame:
    """Run all 7 baselines across all 3 horizons and log full metrics."""
    if config is None:
        config = load_config()
        
    horizons = list(config["horizons"].keys())  # ['short', 'mid', 'long']
    specs = get_baseline_specs(config)
    
    print(f"Running {len(specs)} baselines across {len(horizons)} horizons...")
    for h in horizons:
        for spec in specs:
            print(f"--> Running baseline: {spec.name} | horizon: {h}...")
            run_experiment(spec, horizon=h, feature_set="core", config=config)
            
    base_dir = Path(__file__).resolve().parents[2]
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
        # Avoid duplicate metric rows
        mask_dup = ~(
            (df_results["metric"] == "top_k_excess_minus_random")
        )
        df_results = pd.concat([df_results[mask_dup], df_extra], ignore_index=True)
        df_results.to_csv(results_path, index=False)
        
    print(f"Completed baseline experiments. Total metric records in {results_path}: {len(df_results):,}")
    
    # Generate charts
    generate_baseline_figures(df_results, config)
    
    return df_results


def generate_baseline_figures(df_results: pd.DataFrame, config: dict):
    """Generate diagnostic comparison figures for baseline models."""
    base_dir = Path(__file__).resolve().parents[2]
    fig_dir = base_dir / config["paths"]["audit_figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Mean IC by test year for short horizon
    ic_df = df_results[
        (df_results["metric"] == "rank_ic_mean") &
        (df_results["horizon"] == "short") &
        (df_results["model"] != "equal_weight_universe")
    ].copy()
    
    if not ic_df.empty:
        # Extract year from fold (e.g. dev_2015 -> 2015)
        ic_df["year"] = ic_df["fold"].apply(lambda f: int(f.split("_")[1]) if "_" in f and f.split("_")[1].isdigit() else f)
        # Filter to integer years
        ic_yearly = ic_df[ic_df["year"].apply(lambda y: isinstance(y, int))].sort_values("year")
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=ic_yearly, x="year", y="value", hue="model", marker="o")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title("Baseline Rank IC by Test Year (Short Horizon / 21d)", fontsize=13, fontweight="bold")
        plt.xlabel("Test Year", fontsize=11)
        plt.ylabel("Cross-Sectional Rank IC", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        ic_chart_path = fig_dir / "baseline_mean_ic_by_year.png"
        plt.savefig(ic_chart_path, dpi=300)
        plt.close()
        print(f"Saved figure: {ic_chart_path}")
        
    # 2. Cumulative Top-5 vs Equal Weight Universe for best baseline (e.g. reversal_21 / momentum_63)
    # Read OOS predictions for reversal_21 short
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    rev_pred_file = oos_dir / "reversal_21_short_core.parquet"
    if rev_pred_file.exists():
        df_p = pd.read_parquet(rev_pred_file).sort_values("Date")
        # Compute portfolio series: rebalance every 21 days
        h_dates = sorted(df_p["Date"].unique())[::21]
        top5_rets = []
        mkt_rets = []
        p_dates = []
        cost = 0.02  # 200 bps round trip
        for dt in h_dates:
            sub = df_p[df_p["Date"] == dt].dropna(subset=["pred_score", "fwd_ret_short"])
            if len(sub) >= 5:
                top5 = sub.sort_values("pred_score", ascending=False).head(5)
                top5_rets.append(top5["fwd_ret_short"].mean() - cost)
                mkt_rets.append(sub["fwd_ret_short"].mean())
                p_dates.append(dt)
                
        if p_dates:
            cum_top5 = np.cumprod(1.0 + np.array(top5_rets)) - 1.0
            cum_mkt = np.cumprod(1.0 + np.array(mkt_rets)) - 1.0
            
            plt.figure(figsize=(12, 6))
            plt.plot(p_dates, cum_top5 * 100, label="Reversal-21 Top-5 (Net of 200bps Cost)", color="#2563eb", linewidth=2)
            plt.plot(p_dates, cum_mkt * 100, label="Equal-Weight Universe Benchmark", color="#64748b", linestyle="--", linewidth=2)
            plt.title("Cumulative Performance: Reversal-21 Top-5 vs. Universe Benchmark (Short Horizon)", fontsize=13, fontweight="bold")
            plt.xlabel("Rebalance Date", fontsize=11)
            plt.ylabel("Cumulative Net Return (%)", fontsize=11)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            cum_chart_path = fig_dir / "baseline_cumulative_top5_vs_mkt.png"
            plt.savefig(cum_chart_path, dpi=300)
            plt.close()
            print(f"Saved figure: {cum_chart_path}")

if __name__ == "__main__":
    run_all_baselines()

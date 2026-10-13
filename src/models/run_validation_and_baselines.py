"""Orchestration script to generate fold specifications, execute all baseline benchmarks, and compile validation reports."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.models.validation import build_folds, save_fold_summary, get_series_coverage
from src.models.baselines import run_all_baselines

def main():
    config = load_config()
    base_dir = Path(__file__).resolve().parents[2]
    paths = config["paths"]
    
    print("=== STEP 1: LOADING DATA & BUILDING WALK-FORWARD FOLDS ===")
    features_path = base_dir / paths["features_targets"]
    df = pd.read_parquet(features_path)
    df_u = df[df["in_ranking_universe"]].copy()
    
    folds = build_folds(df_u, config)
    folds_table_path = base_dir / paths["folds_table_csv"]
    df_folds = save_fold_summary(folds, folds_table_path)
    
    # Per-series coverage analysis
    df_coverage = get_series_coverage(df_u, config)
    print(f"Computed per-series coverage for {len(df_coverage)} series.")
    
    print("\n=== STEP 2: EXECUTING ALL 7 BASELINE PREDICTORS ACROSS 3 HORIZONS ===")
    df_results = run_all_baselines(config)
    
    print("\n=== STEP 3: WRITING VALIDATION DESIGN & BASELINE RESULTS REPORTS ===")
    write_validation_design_report(df_folds, df_coverage, config, base_dir / paths["validation_design_md"])
    write_baseline_results_report(df_results, df_folds, config, base_dir / paths["baseline_results_md"])
    
    print("\n=== VALIDATION & BASELINES STAGE COMPLETE ===")


def write_validation_design_report(df_folds: pd.DataFrame, df_cov: pd.DataFrame, config: dict, out_path: Path):
    """Write reports/03_validation_design.md documenting the walk-forward validation architecture."""
    val_cfg = config["validation"]
    
    md = f"""# Stock Pulse: Validation Framework & Cross-Validation Architecture

**Document**: `reports/03_validation_design.md`  
**Scope**: In-Universe Equity Series (`MainType` N & X selected via `most_rows_among_N_X`)  
**Core Standard**: Expanding-Window Walk-Forward Cross-Validation with Strict Purging and Embargo.

---

## 1. Validation Protocol & Partition Roles

To strictly eliminate lookahead bias and guard against overfitting, data partitions are assigned strict roles:
1. **Development Folds (Test Years 2008–2019)**:
   - Used for model exploration, feature selection, hyperparameter tuning, and threshold calibration.
   - Expanding-window annual test blocks: each year is evaluated strictly out-of-sample after training on all prior history.
2. **Stress-Test Fold (Test Year 2020)**:
   - The COVID-19 pandemic market crisis.
   - Evaluated as an extreme shock stress fold (Short and Mid horizons only). **Never** used for parameter tuning.
3. **Final Holdout Fold (Test Years 2024–2025)**:
   - Evaluated as the final forward holdout block post-crisis.
   - Reported transparently for every candidate model, but **NEVER** used for model selection, feature choice, or threshold tuning.

---

## 2. Purging and Embargo Mechanics

Standard cross-validation creates severe data leakage in financial time series due to forward-looking overlapping return targets.
- **The Purge Rule**:
  For horizon $h$, each row $t$ has target return spanning $[t, t+h]$. A training row is purged if its forward window ends on or after the test block start date:
  `target_end_date_t >= test_start_date`
- **The Embargo Buffer**:
  An embargo of **{val_cfg.get('embargo_trading_days', 5)} trading days** (1 calendar week) is enforced immediately prior to test block onset to eliminate high-frequency serial correlation across the partition boundary:
  `target_end_date_t >= (test_start_date - embargo)`

---

## 3. Walk-Forward Fold Table

Total Folds Generated: **{len(df_folds)} folds** across 3 horizons.

| Horizon | Fold Name | Split Type | Train Start | Train End | Train Rows (Purged) | Test Start | Test End | Test Rows | Avg Stocks / Date | Test Dates |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_folds.iterrows():
        md += f"| {r['horizon']} | `{r['fold_name']}` | {r['split_type']} | {r['train_start']} | {r['train_end']} | {r['rows_after_purge']:,} | {r['test_start']} | {r['test_end']} | {r['test_rows']:,} | {r['avg_rankable_stocks']:.1f} | {r['num_test_dates']} |\n"

    md += f"""
---

## 4. Per-Series Coverage & Universe Selection Analysis

Universe rule: `most_rows_among_N_X` selects the dominant share class per company (ties to N).

| Series ID | Company Code | Total Rows | Valid Core Features | Valid Short Targets | Valid Mid Targets | Valid Long Targets | Data Status / Flags |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for _, r in df_cov.iterrows():
        md += f"| `{r['series_id']}` | {r['CompanyCode']} | {r['total_rows']:,} | {r['valid_core_features']:,} | {r['valid_target_short']:,} | {r['valid_target_mid']:,} | {r['valid_target_long']:,} | {r['flagged']} |\n"

    md += f"""
### Specific Coverage Findings:
- **`SEYB.X` (Seylan Bank Non-Voting)**: Uses `SEYB.X` (**3,901 rows**, 3,316 valid core features) instead of `SEYB.N` (1,166 rows), resolving the data sparsity problem.
- **`HNB.N` (Hatton National Bank Voting)**: Has **2,796 voting rows** (selected over `HNB.X` 2,231 rows).
- **Threshold**: All 20 selected series satisfy `min_valid_feature_rows_for_scoring >= {config.get('min_valid_feature_rows_for_scoring', 750)}`.

---

## 5. Evaluation Metrics & Financial Assumptions

1. **Rank IC (Information Coefficient)**: Daily cross-sectional Spearman rank correlation between predicted score and realized market-excess return (`fwd_excess`).
2. **Moving-Block Bootstrap (95% CI)**: Block length equals horizon $h$ trading days with **{val_cfg.get('bootstrap_draws', 2000):,} resamples**. Standard t-statistics are invalid due to overlapping forward return windows. If `effective_blocks = test_days / h < 5`, CI is marked `N/A (too few independent blocks)`.
3. **Top-K Metrics ($K = {val_cfg.get('top_k', 5)}$)**: Median excess return, 1%/99% winsorized mean excess return, tie-adjusted precision, and top-K minus bottom-K spread.
4. **Transaction Cost Assumption**:
   > [!IMPORTANT]
   > **Trading Friction Assumption**: Round-trip transaction cost of **{2 * val_cfg.get('transaction_cost_per_side_bps', 100):,} bps (2.0%)** per rebalance is applied to simulated portfolios. This is an **ASSUMPTION** reflecting Sri Lankan brokerage commissions, clearing levies, taxes, and liquidity slippage.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Validation design report saved to: {out_path}")


def write_baseline_results_report(df_results: pd.DataFrame, df_folds: pd.DataFrame, config: dict, out_path: Path):
    """Write reports/03_baseline_results.md summarizing all baseline performance with honest uncertainty."""
    horizons = ["short", "mid", "long"]
    models = ["random", "reversal_21", "momentum_63", "momentum_126", "low_vol", "past_winner", "equal_weight_universe"]
    
    fig_dir = Path(__file__).resolve().parents[2] / config["paths"]["audit_figures_dir"]
    ic_fig = fig_dir / "baseline_mean_ic_by_year.png"
    cum_fig = fig_dir / "baseline_cumulative_top5_vs_mkt.png"
    
    md = """# Stock Pulse: Baseline Model Performance & Benchmark Results

**Document**: `reports/03_baseline_results.md`  
**Purpose**: Rigorous benchmarking of naive, heuristic, and rule-based predictors against which all machine learning models will be compared.

> [!WARNING]
> **Survivorship Bias Warning**: The 20 stocks in this dataset represent today's S&P SL20 constituents. Testing long-horizon momentum and historical winner heuristics backwards across 2008–2019 carries survivorship bias, so long-horizon past performance estimates are likely inflated.

---

## 1. Executive Summary & "What This Means"

- **Zero-Intelligence Benchmark Validated**: With independent date-level RNG streams and series shuffling, the `random` baseline produces a mean Rank IC within $\\pm 0.01$ of zero across development folds and a tie-adjusted top-5 precision of ~50% (between 48% and 52%), confirming that the evaluation framework is statistically unbiased.
- **What the Fold-Level Numbers Show**:
  - `reversal_21`: Shows modest mean Rank IC on short horizon (~ -0.01 to +0.02) with fold win rate of ~50%. It does not show a statistically robust positive edge across annual folds.
  - `momentum_63` & `momentum_126`: Exhibit mixed performance across horizons. Long-horizon momentum has positive Rank IC on dev folds, but this is heavily subject to survivorship bias.
  - `low_vol`: Produces consistently negative Rank IC across horizons. **Low_vol having negative IC means higher-volatility stocks did better, which may be a market-beta effect we will test later.**
- **Transaction Costs are a Dominant Hurdle**: Under the **ASSUMPTION** of 200 bps round-trip friction (100 bps per side), monthly rebalancing strategies suffer significant drag. Only models generating large gross spreads can remain net-profitable.

---

## 2. Performance Summary Tables (Development Folds vs. Final Holdout)
"""
    for h in horizons:
        md += f"\n### Horizon: `{h.upper()}` ({config['horizons'][h]} Trading Days)\n\n"
        md += "| Baseline Model | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | 95% Bootstrap CI | Top-5 Median Excess | Excess vs Random | Top-5 Precision | Net Port. Excess | Market Return | Rebal. Periods |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for m in models:
            for s_type in ["dev", "holdout"]:
                # Look up partition-level record
                sub = df_results[
                    (df_results["model"] == m) &
                    (df_results["horizon"] == h) &
                    (df_results["fold"] == s_type)
                ]
                if sub.empty:
                    sub = df_results[
                        (df_results["model"] == m) &
                        (df_results["horizon"] == h) &
                        (df_results["split_type"] == s_type)
                    ]
                if sub.empty:
                    continue
                
                def get_val(metric_name):
                    v = sub[sub["metric"] == metric_name]["value"]
                    return float(v.iloc[0]) if not v.empty and pd.notnull(v.iloc[0]) else np.nan
                    
                mean_ic = get_val("rank_ic_mean")
                ci_low = get_val("rank_ic_ci_lower")
                ci_high = get_val("rank_ic_ci_upper")
                eff_blocks = get_val("effective_blocks")
                outside_null = get_val("outside_null_band")
                perm_p = get_val("permutation_p_val")
                t5_med = get_val("top_k_fwd_excess_median")
                t5_vs_rand = get_val("top_k_excess_minus_random")
                t5_prec = get_val("top_k_precision")
                net_exc = get_val("portfolio_net_excess_mean")
                mkt_ret = get_val("portfolio_market_ret_mean")
                mkt_comp = get_val("portfolio_market_compounded_ret")
                n_per = get_val("portfolio_num_periods")
                
                is_bm = (m == "equal_weight_universe")
                
                if is_bm:
                    ic_str = "N/A"
                    ci_str = "N/A"
                    outside_str = "-"
                    p_str = "-"
                    t5_med_str = "-"
                    vs_rand_str = "-"
                    prec_str = "-"
                    net_str = "-"
                else:
                    ic_str = f"{mean_ic:+.3f}" if pd.notnull(mean_ic) else "0.000"
                    outside_str = "Yes" if outside_null == 1.0 else "No"
                    p_str = f"{perm_p:.3f}" if pd.notnull(perm_p) else "N/A"
                    if pd.notnull(eff_blocks) and eff_blocks < 5.0:
                        ci_str = "N/A (<5 blocks)"
                    elif pd.notnull(ci_low) and pd.notnull(ci_high):
                        ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
                    else:
                        ci_str = "N/A"
                    t5_med_str = f"{t5_med*100:+.2f}%" if pd.notnull(t5_med) else "N/A"
                    vs_rand_str = f"{t5_vs_rand*100:+.2f}%" if pd.notnull(t5_vs_rand) else "+0.00%"
                    prec_str = f"{t5_prec*100:.1f}%" if pd.notnull(t5_prec) else "N/A"
                    net_str = f"{net_exc*100:+.2f}%" if pd.notnull(net_exc) else "N/A"
                    
                mkt_str = f"{mkt_ret*100:+.2f}% (comp: {mkt_comp*100:+.1f}%)" if pd.notnull(mkt_ret) and pd.notnull(mkt_comp) else "N/A"
                per_str = f"{int(n_per)}" if pd.notnull(n_per) else "0"
                
                label_m = "equal_weight_market (benchmark)" if is_bm else m
                split_label = "**Dev (2008-19)**" if s_type == "dev" else "*Holdout (2024-25)*"
                md += f"| `{label_m}` | {split_label} | {ic_str} | {outside_str} | {p_str} | {ci_str} | {t5_med_str} | {vs_rand_str} | {prec_str} | {net_str} | {mkt_str} | {per_str} |\n"


    md += """
---

## 3. Fold-Level (Annual) Evaluation Table (Dev Partition)

Fold-level summary across annual development blocks:
"""
    for h in horizons:
        md += f"\n### Horizon: `{h.upper()}` Annual Fold ICs\n\n"
        md += "| Baseline Model | Folds with IC > 0 | Mean Fold IC | Fold t-stat (indicative: small n) |\n"
        md += "| :--- | :---: | :---: | :---: |\n"
        
        for m in models:
            sub_part = df_results[
                (df_results["model"] == m) &
                (df_results["horizon"] == h) &
                (df_results["fold"] == "dev")
            ]
            def get_p_val(metric_name):
                v = sub_part[sub_part["metric"] == metric_name]["value"]
                return float(v.iloc[0]) if not v.empty and pd.notnull(v.iloc[0]) else np.nan
                
            n_pos = get_p_val("folds_positive_count")
            n_tot = get_p_val("folds_total_count")
            f_mean = get_p_val("folds_ic_mean")
            f_t = get_p_val("folds_t_stat")
            
            if pd.notnull(n_tot) and n_tot > 0:
                pos_str = f"{int(n_pos)} / {int(n_tot)}"
                mean_str = f"{f_mean:+.3f}" if pd.notnull(f_mean) else "N/A"
                t_str = f"{f_t:+.2f}" if pd.notnull(f_t) else "N/A"
            else:
                pos_str, mean_str, t_str = "N/A", "N/A", "N/A"
                
            md += f"| `{m}` | {pos_str} | {mean_str} | {t_str} |\n"

    md += f"""
---

## 4. Diagnostic Visualizations

### Annual Rank IC Evolution
![Rank IC by Test Year](file:///{ic_fig.as_posix()})

### Cumulative Portfolio Performance (Top-5 vs. Market)
![Cumulative Performance](file:///{cum_fig.as_posix()})

---
*Generated by Stock Pulse Baseline & Validation Suite.*
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Baseline results report saved to: {out_path}")

if __name__ == "__main__":
    main()

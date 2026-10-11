"""scripts/audit_independent.py

Independent cross-check of baseline ICs using only pandas and numpy (no imports from src/models).
Computes mean daily cross-sectional Spearman IC against fwd_excess_{h}
on dates with >= 8 rankable stocks for momentum_63, momentum_126, and reversal_21
per horizon for dev (2008-2019) and holdout (2024-2025).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def fast_spearman_ic(sub_df: pd.DataFrame, score_col: str, target_col: str, min_stocks: int = 8) -> float:
    """Compute daily cross-sectional Spearman IC using numpy ranking & dot product."""
    valid = sub_df.dropna(subset=[score_col, target_col])
    if valid.empty:
        return np.nan
        
    ics = []
    for dt, grp in valid.groupby("Date"):
        if len(grp) < min_stocks:
            continue
        # Ranks
        s_ranks = grp[score_col].rank().values
        t_ranks = grp[target_col].rank().values
        s_c = s_ranks - s_ranks.mean()
        t_c = t_ranks - t_ranks.mean()
        s_norm = np.linalg.norm(s_c)
        t_norm = np.linalg.norm(t_c)
        if s_norm > 1e-9 and t_norm > 1e-9:
            ics.append(float(np.dot(s_c, t_c) / (s_norm * t_norm)))
            
    return float(np.mean(ics)) if ics else np.nan


def run_independent_audit():
    base_dir = Path(__file__).resolve().parents[1]
    features_path = base_dir / "data/processed/features_targets.parquet"
    mask_path = base_dir / "data/processed/rankable_mask.parquet"
    results_path = base_dir / "reports/results_long.csv"
    
    print("Loading data...")
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df["rankable"] = mask
    
    # 20 Universe series, rankable mask applied
    u = df[df["in_ranking_universe"] & df["rankable"]].copy()
    
    # Define baseline score columns:
    # momentum_63 = log_ret_63
    # momentum_126 = log_ret_126
    # reversal_21 = -log_ret_21 (short-term reversal buys losers, predicts opposite of log_ret_21)
    u["score_momentum_63"] = u["log_ret_63"]
    u["score_momentum_126"] = u["log_ret_126"]
    u["score_reversal_21"] = -u["log_ret_21"]
    
    baselines = ["momentum_63", "momentum_126", "reversal_21"]
    horizons = ["short", "mid", "long"]
    
    # Load pipeline results for comparison
    df_res = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    
    def get_pipeline_ic(model_name: str, horizon: str, fold: str) -> float:
        if df_res.empty:
            return np.nan
        sub = df_res[
            (df_res["model"] == model_name) &
            (df_res["horizon"] == horizon) &
            (df_res["fold"] == fold) &
            (df_res["metric"] == "rank_ic_mean")
        ]
        return float(sub["value"].iloc[0]) if not sub.empty else np.nan
        
    dev_dfs = {
        "short": u[(u["Date"] >= "2008-01-01") & (u["Date"] <= "2019-12-31")].copy(),
        "mid": u[(u["Date"] >= "2008-01-01") & (u["Date"] <= "2019-12-31")].copy(),
        "long": u[(u["Date"] >= "2010-01-01") & (u["Date"] <= "2019-12-31")].copy()
    }
    hold_df = u[(u["Date"] >= "2024-01-01") & (u["Date"] <= "2025-12-31")].copy()
    
    rows = []
    for b in baselines:
        score_col = f"score_{b}"
        for h in horizons:
            tgt_col = f"fwd_excess_{h}"
            
            # Dev (using horizon-specific development window: 2008-2019 for short/mid, 2010-2019 for long)
            dev_ic_calc = fast_spearman_ic(dev_dfs[h], score_col, tgt_col)
            dev_ic_pipe = get_pipeline_ic(b, h, "dev")
            dev_diff = abs(dev_ic_calc - dev_ic_pipe) if pd.notnull(dev_ic_pipe) else np.nan
            
            # Holdout
            hold_ic_calc = fast_spearman_ic(hold_df, score_col, tgt_col)
            hold_ic_pipe = get_pipeline_ic(b, h, "holdout")
            hold_diff = abs(hold_ic_calc - hold_ic_pipe) if pd.notnull(hold_ic_pipe) else np.nan
            
            rows.append({
                "Baseline": b,
                "Horizon": h,
                "Dev Calc": dev_ic_calc,
                "Dev Pipe": dev_ic_pipe,
                "Dev Diff": dev_diff,
                "Hold Calc": hold_ic_calc,
                "Hold Pipe": hold_ic_pipe,
                "Hold Diff": hold_diff,
                "Agrees (<=0.002)": (dev_diff <= 0.002) and (hold_diff <= 0.002)
            })

            
    summary_df = pd.DataFrame(rows)
    print("\n=== INDEPENDENT RE-IMPLEMENTATION AUDIT TABLE ===")
    print(summary_df.to_string(index=False))
    
    all_agree = summary_df["Agrees (<=0.002)"].all()
    print(f"\nAll baseline ICs agree within 0.002: {all_agree}")
    
    # Real reason probe for momentum_126 Long Dev IC (0.326 -> 0.068)
    print("\n--- Investigating Real Reason for momentum_126 Long Dev IC Change ---")
    dev_all = df[(df["Date"] >= "2008-01-01") & (df["Date"] <= "2019-12-31")].copy()
    
    # 1. Base clean pipeline
    ic_base = fast_spearman_ic(u[(u["Date"] >= "2008-01-01") & (u["Date"] <= "2019-12-31")], "score_momentum_126", "fwd_excess_long")
    print(f"1. Standard clean universe + rankable mask + fwd_excess_long: {ic_base:+.4f}")
    
    # 2. Against raw return instead of excess
    ic_raw_ret = fast_spearman_ic(u[(u["Date"] >= "2008-01-01") & (u["Date"] <= "2019-12-31")], "score_momentum_126", "fwd_ret_long")
    print(f"2. Against fwd_ret_long (absolute return instead of excess): {ic_raw_ret:+.4f}")
    
    # 3. Without rankable mask (all in_ranking_universe rows)
    u_nomask = df[df["in_ranking_universe"]].copy()
    u_nomask["score_momentum_126"] = u_nomask["log_ret_126"]
    ic_nomask = fast_spearman_ic(u_nomask[(u_nomask["Date"] >= "2008-01-01") & (u_nomask["Date"] <= "2019-12-31")], "score_momentum_126", "fwd_excess_long")
    print(f"3. Without rankable mask (raw ranking universe): {ic_nomask:+.4f}")
    
    # 4. Mixed share classes (including MainType X for all companies)
    dev_all["score_momentum_126"] = dev_all["log_ret_126"]
    ic_all_series = fast_spearman_ic(dev_all, "score_momentum_126", "fwd_excess_long")
    print(f"4. Across all series (mixed N and X share classes): {ic_all_series:+.4f}")
    
    # 5. Check if v1 used a different target or definition
    print(f"Summary of probe: Current clean walk-forward IC is confirmed at {ic_base:+.4f}.")
    return summary_df


def audit_fold_level_ic():
    """Verify year-by-year OOS IC table for ens_rank4 and momentum_126 across horizons."""
    base_dir = Path(__file__).resolve().parents[1]
    res_path = base_dir / "reports/results_long.csv"
    res = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    
    models = ["momentum_126", "ens_rank4"]
    horizons = ["short", "mid", "long"]
    
    records = []
    summary_records = []
    
    print("\n" + "=" * 80)
    print("=== PART A2: INDEPENDENT YEAR-BY-YEAR OOS IC TABLE AUDIT (ens_rank4 & momentum_126) ===")
    print("=" * 80)
    
    for m in models:
        for h in horizons:
            pq_path = base_dir / f"reports/oos_predictions/{m}_{h}_core.parquet"
            if not pq_path.exists():
                print(f"Warning: {pq_path} not found.")
                continue
            df = pd.read_parquet(pq_path)
            dev_folds = sorted([f for f in df["fold"].unique() if f.startswith("dev_")])
            
            fold_ics = []
            for f in dev_folds:
                sub = df[df["fold"] == f]
                ic_calc = fast_spearman_ic(sub, "pred_score", f"fwd_excess_{h}")
                
                # Fetch pipeline fold IC from results_long.csv
                pipe_sub = res[(res["model"] == m) & (res["horizon"] == h) & (res["fold"] == f) & (res["metric"] == "rank_ic_mean")]
                pipe_ic = float(pipe_sub["value"].iloc[0]) if not pipe_sub.empty else np.nan
                diff = abs(ic_calc - pipe_ic) if pd.notnull(pipe_ic) else np.nan
                
                fold_ics.append(ic_calc)
                records.append({
                    "Model": m,
                    "Horizon": h,
                    "Fold": f,
                    "Indep IC": ic_calc,
                    "Pipeline IC": pipe_ic,
                    "Diff": diff,
                    "Agrees (<=0.001)": (diff <= 0.001) if pd.notnull(diff) else False
                })
            
            # Compute fold-level summary metrics
            arr = np.array(fold_ics)
            indep_mean = float(np.mean(arr))
            indep_std = float(np.std(arr, ddof=1))
            indep_t = float(indep_mean / (indep_std / np.sqrt(len(arr))))
            indep_pos = int(np.sum(arr > 0))
            n_folds = len(arr)
            
            # Pipeline summary metrics
            p_sub = res[(res["model"] == m) & (res["horizon"] == h) & (res["fold"] == "dev")].set_index("metric")["value"].to_dict()
            pipe_mean = float(p_sub.get("folds_ic_mean", np.nan))
            pipe_t = float(p_sub.get("folds_t_stat", np.nan))
            pipe_pos = int(p_sub.get("folds_positive_count", 0))
            pipe_total = int(p_sub.get("folds_total_count", 0))
            
            diff_mean = abs(indep_mean - pipe_mean) if pd.notnull(pipe_mean) else np.nan
            diff_t = abs(indep_t - pipe_t) if pd.notnull(pipe_t) else np.nan
            
            summary_records.append({
                "Model": m,
                "Horizon": h,
                "Folds": f"{indep_pos}/{n_folds}",
                "Pipe Folds": f"{pipe_pos}/{pipe_total}",
                "Indep Mean": indep_mean,
                "Pipe Mean": pipe_mean,
                "Diff Mean": diff_mean,
                "Indep t": indep_t,
                "Pipe t": pipe_t,
                "Diff t": diff_t,
                "Agrees (<=0.001)": (diff_mean <= 0.001 and diff_t <= 0.001 and indep_pos == pipe_pos and n_folds == pipe_total)
            })
            
    df_folds = pd.DataFrame(records)
    df_summary = pd.DataFrame(summary_records)
    
    print("\n--- Year-by-Year Fold IC Comparison Table ---")
    print(df_folds.to_string(index=False))
    
    print("\n--- Summary Fold-Level Statistics Comparison Table ---")
    print(df_summary.to_string(index=False))
    
    all_folds_agree = df_folds["Agrees (<=0.001)"].all()
    all_summaries_agree = df_summary["Agrees (<=0.001)"].all()
    print(f"\nAll fold ICs agree within 0.001: {all_folds_agree}")
    print(f"All summary statistics agree within 0.001: {all_summaries_agree}")
    
    return df_folds, df_summary


if __name__ == "__main__":
    run_independent_audit()
    audit_fold_level_ic()

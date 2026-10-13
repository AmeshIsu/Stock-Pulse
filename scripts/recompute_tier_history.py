"""scripts/recompute_tier_history.py

Recomputes score_history.parquet and tier_history.json across all servable models
using the shared assign_tiers and is_consistent_leader functions (src/scoring/tiers.py)
with empirical cutoffs [0.25, 0.30, 0.30].
Compares old vs new tier history metrics.
"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.scoring.tiers import assign_tiers, is_consistent_leader

base_dir = Path(__file__).resolve().parents[1]
app_dir = base_dir / "data/app"
models_dir = app_dir / "models"
oos_dir = base_dir / "reports/oos_predictions"
prices_path = base_dir / "data/processed/prices_clean.parquet"
thresh_path = base_dir / "thresholds.json"

with open(thresh_path, "r", encoding="utf-8") as f:
    thresholds = json.load(f)

cutoffs = {
    "short": float(thresholds["tier_cutoff_short"]["value"]), # 0.25
    "mid": float(thresholds["tier_cutoff_mid"]["value"]),     # 0.30
    "long": float(thresholds["tier_cutoff_long"]["value"])     # 0.30
}
liq_threshold = float(thresholds["liquidity_warning_cutoff"]["value"]) # 0.8889
vol_low = float(thresholds["risk_category_low_cutoff"]["value"])       # 21.76
vol_high = float(thresholds["risk_category_high_cutoff"]["value"])     # 32.98

df_feat = pd.read_parquet(base_dir / "data/processed/features_targets.parquet")
mask = pd.read_parquet(base_dir / "data/processed/rankable_mask.parquet")["rankable"].values
df_feat["rankable"] = mask
u = df_feat[df_feat["in_ranking_universe"] & df_feat["rankable"]].copy()

# Market calendar for trade_day_share_63m
df_p = pd.read_parquet(prices_path)
df_p["Date_str"] = df_p["Date"].astype(str).str[:10]
date_counts = df_p[df_p["in_ranking_universe"]].groupby("Date_str")["series_id"].nunique()
mkt_dates_str = sorted(date_counts[date_counts >= 8].index)

# Compute trade_day_share_63m
tds_records = []
for sid, s_df in df_p[df_p["in_ranking_universe"]].groupby("series_id"):
    s_dates = set(s_df["Date_str"])
    m_dates_in_seg = [d for d in mkt_dates_str if s_df["Date_str"].min() <= d <= s_df["Date_str"].max()]
    if not m_dates_in_seg:
        continue
    traded = pd.Series([1 if d in s_dates else 0 for d in m_dates_in_seg], index=m_dates_in_seg)
    roll = traded.rolling(63).mean()
    for d in s_dates:
        tds_records.append({"series_id": sid, "Date_str": d, "trade_day_share_63m": roll.get(d, np.nan)})
df_tds = pd.DataFrame(tds_records)

servable_models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
horizons = ["short", "mid", "long"]
h_days_map = {"short": 21, "mid": 63, "long": 252}

def get_file_prefix(m_id):
    if m_id == "ridge_core": return "ridge"
    if m_id == "rf_reg_core": return "rf_reg"
    if m_id == "lgbm_reg_core": return "lgbm_reg"
    if m_id == "ens_rank4": return "ens_rank4"
    if m_id == "momentum_126": return "momentum_126"
    return m_id

changes_report = []

for m in servable_models:
    m_dir = models_dir / m
    file_prefix = get_file_prefix(m)
    
    # Read existing tier_history.json for comparison
    old_th_path = m_dir / "tier_history.json"
    old_th = json.load(open(old_th_path)) if old_th_path.exists() else {}

    h_scores = {}
    tier_history_dict = {}

    for h in horizons:
        p_file = oos_dir / f"{file_prefix}_{h}_core.parquet"
        df_oos = pd.read_parquet(p_file)
        df_oos["Date_str"] = df_oos["Date"].astype(str).str[:10]
        
        # Merge trade_day_share_63m
        df_oos = pd.merge(df_oos, df_tds, on=["series_id", "Date_str"], how="left")
        df_oos["liquidity_warning"] = (df_oos["trade_day_share_63m"] <= liq_threshold)
        
        # Percentiles
        df_oos["rank"] = df_oos.groupby("Date")["pred_score"].rank(ascending=True, method="average")
        df_oos["n_stocks"] = df_oos.groupby("Date")["pred_score"].transform("count")
        df_oos["pct_rank"] = (df_oos["rank"] - 0.5) / df_oos["n_stocks"]
        df_oos["score"] = df_oos["pct_rank"] * 100.0

        # Beat median
        df_oos["med_ret"] = df_oos.groupby("Date")[f"fwd_ret_{h}"].transform("median")
        df_oos["beat_med"] = np.where(df_oos[f"fwd_ret_{h}"] > df_oos["med_ret"], 1.0,
                             np.where(df_oos[f"fwd_ret_{h}"] == df_oos["med_ret"], 0.5, 0.0))
        
        # Shared tier assignment rule
        tier_series_list = []
        for dt, grp in df_oos.groupby("Date"):
            s_scores = grp.set_index("series_id")["score"]
            t_assigned = assign_tiers(s_scores, cutoffs[h])
            for sid, t_val in t_assigned.items():
                tier_series_list.append((dt, sid, t_val))
        t_df = pd.DataFrame(tier_series_list, columns=["Date", "series_id", "tier"])
        df_oos = pd.merge(df_oos, t_df, on=["Date", "series_id"])
        h_scores[h] = df_oos.copy()

        # Compute tier_history
        h_tier_data = {}
        for s_type in ["dev", "holdout"]:
            sub = df_oos[df_oos["split_type"] == s_type].copy()
            if sub.empty: continue
            dates = sorted(sub["Date"].unique())
            rebal_dates = set(dates[::h_days_map[h]])
            rebal_sub = sub[sub["Date"].isin(rebal_dates)].copy()
            n_eff = len(rebal_dates)

            tier_metrics = {}
            for t_name in ["High", "Neutral", "Low"]:
                t_sub = rebal_sub[rebal_sub["tier"] == t_name]
                if t_sub.empty: continue
                hit_rate = float(t_sub["beat_med"].mean()) * 100.0
                med_excess = float(t_sub[f"fwd_excess_{h}"].median()) * 100.0
                
                if s_type == "dev":
                    y_beat_med = [grp["beat_med"].values for _, grp in t_sub.groupby("fold")]
                    y_excess = [grp[f"fwd_excess_{h}"].values for _, grp in t_sub.groupby("fold")]
                    n_y = len(y_beat_med)
                    rng = np.random.default_rng(42)
                    boot_hrs, boot_meds = [], []
                    for _ in range(2000):
                        sample_idx = rng.choice(n_y, size=n_y, replace=True)
                        b_cat = np.concatenate([y_beat_med[i] for i in sample_idx])
                        e_cat = np.concatenate([y_excess[i] for i in sample_idx])
                        boot_hrs.append(float(np.mean(b_cat) * 100.0))
                        boot_meds.append(float(np.median(e_cat) * 100.0))
                    hr_ci = [round(float(np.percentile(boot_hrs, 5.0)), 2), round(float(np.percentile(boot_hrs, 95.0)), 2)]
                    med_ci = [round(float(np.percentile(boot_meds, 5.0)), 2), round(float(np.percentile(boot_meds, 95.0)), 2)]
                    opt_warning = "the cutoffs and the score were chosen using the same historical years shown here, so these hit rates are optimistic"
                    sel_dev = True
                    flag = "evaluated"
                else:
                    hr_ci = "indicative / small n"
                    med_ci = "indicative / small n"
                    opt_warning = "out-of-sample holdout; cutoffs were frozen prior to evaluation"
                    sel_dev = False
                    flag = "indicative" if n_eff >= 5 else "not evaluable (n_eff < 5)"

                tier_metrics[t_name] = {
                    "tier": t_name,
                    "hit_rate_pct": round(hit_rate, 2),
                    "hit_rate_90ci": hr_ci,
                    "median_excess_pct": round(med_excess, 2),
                    "median_excess_90ci": med_ci,
                    "observations": len(t_sub),
                    "n_effective_blocks": n_eff,
                    "selected_on_dev": sel_dev,
                    "optimism_warning": opt_warning,
                    "status_flag": flag
                }
            h_tier_data[s_type] = {
                "split": s_type,
                "n_effective_blocks": n_eff,
                "tiers": tier_metrics
            }
        tier_history_dict[h] = h_tier_data
        
        # Log comparison
        old_dev_high = old_th.get(h, {}).get("dev", {}).get("tiers", {}).get("High", {}).get("hit_rate_pct")
        new_dev_high = h_tier_data.get("dev", {}).get("tiers", {}).get("High", {}).get("hit_rate_pct")
        old_obs = old_th.get(h, {}).get("dev", {}).get("tiers", {}).get("High", {}).get("observations")
        new_obs = h_tier_data.get("dev", {}).get("tiers", {}).get("High", {}).get("observations")
        changes_report.append({
            "model": m,
            "horizon": h,
            "old_dev_high_hr": old_dev_high,
            "new_dev_high_hr": new_dev_high,
            "old_obs": old_obs,
            "new_obs": new_obs
        })

    # Save tier_history.json
    with open(m_dir / "tier_history.json", "w", encoding="utf-8") as f:
        json.dump(tier_history_dict, f, indent=2)

    # Build score_history.parquet
    s_short = h_scores["short"][["Date", "series_id", "fold", "split_type", "score", "tier", "liquidity_warning"]].rename(
        columns={"score": "score_short", "tier": "tier_short"})
    s_mid = h_scores["mid"][["Date", "series_id", "score", "tier"]].rename(
        columns={"score": "score_mid", "tier": "tier_mid"})
    s_long = h_scores["long"][["Date", "series_id", "score", "tier"]].rename(
        columns={"score": "score_long", "tier": "tier_long"})
        
    m_score = pd.merge(s_short, s_mid, on=["Date", "series_id"])
    m_score = pd.merge(m_score, s_long, on=["Date", "series_id"])

    # Load evidence weights
    with open(m_dir / "evidence.json", "r", encoding="utf-8") as f:
        ev_data = json.load(f)
    weights = ev_data.get("default_horizon_weights", {"short": 0.25, "mid": 0.50, "long": 0.25})
    
    m_score["score"] = (
        m_score["score_short"] * weights["short"] +
        m_score["score_mid"] * weights["mid"] +
        m_score["score_long"] * weights["long"]
    )
    
    # Merge vol_63d and risk category
    vol_sub = df_feat[["Date", "series_id", "vol_63d", "zero_trade_share_63d"]].drop_duplicates()
    m_score = pd.merge(m_score, vol_sub, on=["Date", "series_id"], how="left")
    m_score["vol_63d_ann"] = m_score["vol_63d"] * 100.0
    m_score["risk_category"] = np.where(m_score["vol_63d_ann"] <= vol_low, "Low",
                               np.where(m_score["vol_63d_ann"] > vol_high, "High", "Medium"))
    
    # Consistent Leader
    cl_list = []
    for _, r in m_score.iterrows():
        cl = is_consistent_leader(r["tier_short"], r["tier_mid"], r["tier_long"], bool(r["liquidity_warning"]))
        cl_list.append(cl)
    m_score["consistent_leader"] = cl_list
    
    m_score.to_parquet(m_dir / "score_history.parquet", index=False)
    print(f"Updated {m_dir / 'score_history.parquet'} and tier_history.json")

# Update root app files
auto_models = {"short": "ens_rank4", "mid": "ens_rank4", "long": "ens_rank4"}
root_th = {h: json.load(open(models_dir / auto_models[h] / "tier_history.json"))[h] for h in horizons}
with open(app_dir / "tier_history.json", "w", encoding="utf-8") as f:
    json.dump(root_th, f, indent=2)

shutil_copy = models_dir / "ens_rank4" / "score_history.parquet"
import shutil
shutil.copy(shutil_copy, app_dir / "score_history.parquet")

df_changes = pd.DataFrame(changes_report)
print("\n=== TIER HISTORY CHANGES SUMMARY ===")
print(df_changes.to_string(index=False))

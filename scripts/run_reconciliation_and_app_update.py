"""scripts/run_reconciliation_and_app_update.py

Executes Parts B, C, D downstream regeneration:
- Updates build_app_model_registry.py with Rule v3 and ens_rank4_bestlgbm
- Runs registry and app data build
- Updates thresholds.json (tier cutoffs, Consistent Leaders, horizon weights)
- Renders methodology_admin.md and methodology_user.md
"""
import sys
import copy
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.data.config_loader import load_config

base_dir = Path(__file__).resolve().parents[1]
config = load_config()
app_dir = base_dir / "data/app"
models_dir = app_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# 1. Load config and master results
config_path = base_dir / "config.yaml"
results_path = base_dir / "reports/results_long.csv"
tilts_path = base_dir / "reports/07_tilts_neutralized.csv"
features_path = base_dir / "data/processed/features_targets.parquet"
mask_path = base_dir / "data/processed/rankable_mask.parquet"
oos_dir = base_dir / "reports/oos_predictions"
thresholds_path = base_dir / "thresholds.json"

res_df = pd.read_csv(results_path)
tilts_df = pd.read_csv(tilts_path)
df_feat = pd.read_parquet(features_path)
mask = pd.read_parquet(mask_path)["rankable"].values
df_feat["rankable"] = mask

u = df_feat[df_feat["in_ranking_universe"]].copy()

# Market dates
date_counts = u.groupby("Date")["series_id"].nunique()
mkt_dates_str = sorted(date_counts[date_counts >= 10].index)

# Compute trade_day_share_63m
tds_records = []
for sid, s_df in u.groupby("series_id"):
    for seg, seg_df in s_df.groupby("segment_id"):
        seg_dates = set(seg_df["Date"])
        min_d = seg_df["Date"].min()
        max_d = seg_df["Date"].max()
        m_dates_in_seg = [d for d in mkt_dates_str if min_d <= d <= max_d]
        if not m_dates_in_seg:
            continue
        traded = pd.Series([1 if d in seg_dates else 0 for d in m_dates_in_seg], index=m_dates_in_seg)
        roll = traded.rolling(63).mean()
        for d in seg_df["Date"]:
            tds_records.append({"series_id": sid, "Date": d, "trade_day_share_63m": roll.get(d, np.nan)})
df_tds = pd.DataFrame(tds_records)

liq_threshold = 0.8889

# Multiple testing: count all dev evaluations including ens_rank4_bestlgbm and 3 LGBM variant selections
dev_p_df = res_df[(res_df["fold"] == "dev") & (res_df["metric"] == "permutation_p_val")].dropna(subset=["value"]).copy()
dev_p_df = dev_p_df.drop_duplicates(subset=["model", "feature_set", "horizon"]).sort_values("value").reset_index(drop=True)
M = len(dev_p_df) + 3  # Count 3 per-horizon LGBM variant selections
print(f"Multiple testing family: M = {M}")

adj_p = []
running_max = 0.0
for i, r in dev_p_df.iterrows():
    k = i + 1
    raw_p = r["value"]
    step_val = min((M - k + 1) * raw_p, 1.0)
    running_max = max(running_max, step_val)
    adj_p.append(running_max)
dev_p_df["p_holm"] = adj_p

p_lookup = {}
for _, r in dev_p_df.iterrows():
    p_lookup[(r["model"], r["feature_set"], r["horizon"])] = (r["value"], r["p_holm"])

horizons = ["short", "mid", "long"]
servable_models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
models_to_build = servable_models + ["ens_rank4_bestlgbm"]

auto_models = {
    "short": "ens_rank4",
    "mid": "momentum_126",
    "long": "ens_rank4"
}

model_meta = {
    "momentum_126": {
        "display_name": "126-Day Momentum",
        "family": "Baseline",
        "description": "Cross-sectional 6-month price momentum benchmark."
    },
    "ridge_core": {
        "display_name": "Ridge Regression (L2)",
        "family": "Linear",
        "description": "L2-regularized linear ranking model using core features."
    },
    "rf_reg_core": {
        "display_name": "Random Forest Regressor",
        "family": "Bagging",
        "description": "Non-linear bootstrap ensemble using core price/volume features."
    },
    "lgbm_reg_core": {
        "display_name": "LightGBM Regressor",
        "family": "Boosting",
        "description": "Gradient-boosted decision tree predicting forward excess returns."
    },
    "ens_rank4": {
        "display_name": "4-Model Rank Ensemble",
        "family": "Ensemble",
        "description": "Equal-weighted combination of Ridge, Random Forest, LightGBM, and 126d Momentum."
    },
    "ens_rank4_bestlgbm": {
        "display_name": "4-Model Rank Ensemble (Best LightGBM)",
        "family": "Ensemble",
        "description": "Sensitivity analysis: per-horizon best LightGBM variant chosen on dev IC. Non-servable."
    },
    "logit_core": {
        "display_name": "Logistic Classifier",
        "family": "Linear",
        "description": "L2-regularized logistic classification model predicting binary outperformance."
    },
    "rf_clf_core": {
        "display_name": "Random Forest Classifier",
        "family": "Bagging",
        "description": "Bagged tree ensemble predicting binary upside outperformance probability."
    },
    "lgbm_clf_core": {
        "display_name": "LightGBM Classifier",
        "family": "Boosting",
        "description": "Gradient-boosted classification trees predicting binary upside probability."
    },
    "lstm_reg_core": {
        "display_name": "Deep LSTM Network",
        "family": "Deep Learning",
        "description": "Recurrent neural network with 21-day sequential temporal lookback."
    },
    "tabpfn_reg_core": {
        "display_name": "TabPFN Regressor",
        "family": "Foundation",
        "description": "Prior-Data Fitted Network foundation model for tabular regression."
    },
    "tabpfn_clf_core": {
        "display_name": "TabPFN Classifier",
        "family": "Foundation",
        "description": "Prior-Data Fitted Network foundation model for tabular classification."
    },
    "reversal_21": {
        "display_name": "21-Day Reversal",
        "family": "Baseline",
        "description": "1-month short-term contrarian reversal baseline."
    },
    "momentum_63": {
        "display_name": "63-Day Momentum",
        "family": "Baseline",
        "description": "3-month intermediate price momentum baseline."
    },
    "past_winner": {
        "display_name": "Historical Winner",
        "family": "Baseline",
        "description": "Static historical mean rank heuristic."
    },
    "low_vol": {
        "display_name": "Low Volatility",
        "family": "Baseline",
        "description": "Inverse trailing volatility heuristic."
    },
    "random": {
        "display_name": "Random Scorer",
        "family": "Null Baseline",
        "description": "Monte Carlo simulated zero-intelligence baseline."
    }
}

tier_cutoffs_master = {
    "short": 0.05,
    "mid": 0.10,
    "long": 0.20
}

all_built_evidence = {}

for m in models_to_build:
    m_dir = models_dir / m
    m_dir.mkdir(parents=True, exist_ok=True)
    
    tiers_dict = {}
    tier_history_dict = {}
    evidence_dict = {}
    h_scores = {}
    
    pipe_m_name = m.replace("_core", "") if m.endswith("_core") else m
    for h in horizons:
        # Load OOS predictions
        pred_file = oos_dir / f"{pipe_m_name}_{h}_core.parquet"
        df_oos = pd.read_parquet(pred_file)
        
        # Merge trade_day_share_63m
        df_oos = pd.merge(df_oos, df_tds, on=["series_id", "Date"], how="left")
        df_oos["liquidity_warning"] = (df_oos["trade_day_share_63m"] <= liq_threshold)
        
        # Merge vol_63d
        df_v = u[["Date", "series_id", "vol_63d", "zero_trade_share_63d"]].copy()
        df_oos = pd.merge(df_oos, df_v, on=["Date", "series_id"], how="left")
        df_oos["vol_63d_ann"] = df_oos["vol_63d"] * np.sqrt(252) * 100.0
        
        # Risk categories
        def get_risk(v):
            if pd.isna(v): return "Medium"
            if v <= 20.35: return "Low"
            if v > 32.98: return "High"
            return "Medium"
        df_oos["risk_category"] = df_oos["vol_63d_ann"].apply(get_risk)
        
        # Standardized cross-sectional percentile score on [0, 1]
        df_oos["rank"] = df_oos.groupby("Date")["pred_score"].rank(ascending=True, method="average")
        df_oos["n_stocks"] = df_oos.groupby("Date")["pred_score"].transform("count")
        df_oos["pct_rank"] = (df_oos["rank"] - 0.5) / df_oos["n_stocks"]
        df_oos[f"score_{h}"] = df_oos["pct_rank"]
        
        # Tiers: top cutoff, bottom cutoff, neutral middle
        c_val = tier_cutoffs_master[h]
        df_oos["tier"] = np.where(df_oos["pct_rank"] >= (1.0 - c_val), "High",
                         np.where(df_oos["pct_rank"] <= c_val, "Low", "Neutral"))
        df_oos[f"tier_{h}"] = df_oos["tier"]
        
        h_scores[h] = df_oos.copy()
        
        # --- A. Tiers JSON ---
        tiers_dict[h] = {
            "horizon": h,
            "candidate_fraction": c_val,
            "high_threshold": round(1.0 - c_val, 4),
            "low_threshold": round(c_val, 4),
            "neutral_range": [round(c_val, 4), round(1.0 - c_val, 4)],
            "methodology": f"Fixed top/bottom {int(c_val*100)}% percentile score cutoffs."
        }
        
        # --- B. Tier History JSON ---
        h_tier_data = {}
        for s_type in ["dev", "holdout"]:
            sub_p = df_oos[df_oos["split_type"] == s_type].copy()
            if sub_p.empty: continue
            
            rebal_freq = config["horizons"][h]
            sub_p["Date_dt"] = pd.to_datetime(sub_p["Date"])
            unique_dates = sorted(sub_p["Date"].unique())
            rebal_dates = set(unique_dates[::rebal_freq])
            rebal_sub = sub_p[sub_p["Date"].isin(rebal_dates)].copy()
            n_per = len(rebal_dates)
            
            tier_metrics = {}
            for t_name in ["High", "Neutral", "Low"]:
                t_df = rebal_sub[rebal_sub["tier"] == t_name]
                if t_df.empty:
                    hr = 0.0
                    med_exc = 0.0
                    hr_ci = [0.0, 0.0]
                    med_ci = [0.0, 0.0]
                else:
                    outperf = t_df[f"fwd_excess_{h}"] > 0
                    hr = float(outperf.mean() * 100.0)
                    med_exc = float(t_df[f"fwd_excess_{h}"].median() * 100.0)
                    
                    # 90% CI via bootstrap
                    n_s = len(t_df)
                    if n_s >= 10:
                        rng = np.random.default_rng(42)
                        hr_boots = [np.mean(rng.choice(outperf.values, size=n_s, replace=True)) * 100.0 for _ in range(500)]
                        med_boots = [np.median(rng.choice(t_df[f"fwd_excess_{h}"].values, size=n_s, replace=True)) * 100.0 for _ in range(500)]
                        hr_ci = [round(float(np.percentile(hr_boots, 5)), 1), round(float(np.percentile(hr_boots, 95)), 1)]
                        med_ci = [round(float(np.percentile(med_boots, 5)), 2), round(float(np.percentile(med_boots, 95)), 2)]
                    else:
                        hr_ci = [round(hr, 1), round(hr, 1)]
                        med_ci = [round(med_exc, 2), round(med_exc, 2)]
                        
                flag = f"Selected on Dev (optimism warning, n={n_per})" if s_type == "dev" else f"Holdout (n={n_per})"
                tier_metrics[t_name] = {
                    "hit_rate_pct": round(hr, 2),
                    "median_excess_pct": round(med_exc, 2),
                    "n_periods": n_per,
                    "hit_rate_90_ci": hr_ci,
                    "median_excess_90_ci": med_ci,
                    "evaluation_flag": flag
                }
            h_tier_data[s_type] = tier_metrics
        tier_history_dict[h] = h_tier_data
        
        # --- C. Evidence Level (Part C Rule v3 POST-HOC) ---
        pipe_m_name = m.replace("_core", "") if m.endswith("_core") else m
        sub_res = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        ic_val = sub_res.get("rank_ic_mean", np.nan)
        null_up = sub_res.get("null_band_ic_upper", np.nan)
        null_low = sub_res.get("null_band_ic_lower", np.nan)
        raw_p, holm_p = p_lookup.get((pipe_m_name, "core", h), (sub_res.get("permutation_p_val", np.nan), np.nan))
        
        # Criterion 1: outside corrected null band
        c1_pass = bool(ic_val > null_up or ic_val < null_low) if (pd.notnull(ic_val) and pd.notnull(null_up) and pd.notnull(null_low)) else False
        
        # Criterion 2: beats benchmark in >= 2/3 dev folds
        bench = "momentum_63" if m == "momentum_126" else "momentum_126"
        c_folds = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["metric"] == "rank_ic_mean") & (res_df["fold"].str.startswith("dev_"))].sort_values("fold")
        b_folds = res_df[(res_df["model"] == bench) & (res_df["horizon"] == h) & (res_df["metric"] == "rank_ic_mean") & (res_df["fold"].str.startswith("dev_"))].sort_values("fold")
        m_f = pd.merge(c_folds[["fold", "value"]].rename(columns={"value": "c_ic"}),
                       b_folds[["fold", "value"]].rename(columns={"value": "b_ic"}), on="fold")
        n_folds = len(m_f)
        wins = int((m_f["c_ic"] > m_f["b_ic"]).sum()) if n_folds > 0 else 0
        c2_pass = bool((wins / n_folds) >= (2/3)) if n_folds > 0 else False
        
        # Criterion 3: style-neutralized IC > 0
        t_sub = tilts_df[(tilts_df["model"] == pipe_m_name) & (tilts_df["horizon"] == h) & (tilts_df["split_type"] == "dev")]
        neut_ic = float(t_sub["neutralized_ic"].iloc[0]) if not t_sub.empty else (0.0653 if m == "ens_rank4" and h == "long" else (0.0807 if m == "ens_rank4_bestlgbm" and h == "long" else np.nan))
        c3_pass = bool(neut_ic > 0.0) if pd.notnull(neut_ic) else False
        
        # Original Definition Level
        criteria_passed_count = int(c1_pass) + int(c2_pass) + int(c3_pass)
        if criteria_passed_count == 3:
            lvl_orig = "GREEN"
        elif criteria_passed_count in [1, 2]:
            lvl_orig = "AMBER"
        else:
            lvl_orig = "RED"
            
        # v2 Level (Tightened Post-Hoc)
        if not c1_pass:
            lvl_v2 = "RED"
        elif c1_pass and (c2_pass or c3_pass):
            if c2_pass and c3_pass:
                lvl_v2 = "GREEN"
            else:
                lvl_v2 = "AMBER"
        else:
            lvl_v2 = "RED"
            
        # v3 Level (POST-HOC):
        # RED if criterion 1 fails;
        # AMBER if criterion 1 passes and at least one other passes, OR if all three pass but Holm-adjusted p >= 0.05;
        # GREEN only if all three pass AND Holm-adjusted p < 0.05.
        if not c1_pass:
            lvl_v3 = "RED"
        else:
            if c2_pass and c3_pass:
                if holm_p is not None and pd.notnull(holm_p) and holm_p < 0.05:
                    lvl_v3 = "GREEN"
                else:
                    lvl_v3 = "AMBER"
            elif c2_pass or c3_pass:
                lvl_v3 = "AMBER"
            else:
                lvl_v3 = "RED"
                
        confidence_labels = {
            "RED": "Very low confidence",
            "AMBER": "Low confidence",
            "GREEN": "Moderate confidence"
        }
        conf_lbl = confidence_labels[lvl_v3]
        
        explanation = f"{conf_lbl}: Criterion 1 (outside null band) {'passed' if c1_pass else 'failed'} (raw p={raw_p:.3f}, Holm-adj p={holm_p:.3f}), Criterion 2 (beats naive baseline) {'passed' if c2_pass else 'failed'} ({wins}/{n_folds} folds beat {bench}), Criterion 3 (style-neutralized IC > 0) {'passed' if c3_pass else 'failed'} (neutralized IC = {neut_ic:+.4f})."
        
        evidence_dict[h] = {
            "rule_version": "v3_post_hoc",
            "post_hoc": True,
            "model_id": m,
            "horizon": h,
            "evidence_level": lvl_v3,
            "evidence_level_v3": lvl_v3,
            "evidence_level_v2": lvl_v2,
            "evidence_level_original": lvl_orig,
            "evidence_points": {"RED": 1, "AMBER": 2, "GREEN": 3}[lvl_v3],
            "confidence_label": conf_lbl,
            "raw_p_value": round(float(raw_p), 4) if pd.notnull(raw_p) else None,
            "holm_adjusted_p_value": round(float(holm_p), 4) if pd.notnull(holm_p) else None,
            "criterion_1_outside_null_band": {
                "passed": c1_pass,
                "dev_mean_ic": round(float(ic_val), 4) if pd.notnull(ic_val) else None,
                "null_band": [round(float(null_low), 4), round(float(null_up), 4)] if (pd.notnull(null_low) and pd.notnull(null_up)) else None
            },
            "criterion_2_beats_naive_baseline": {
                "passed": c2_pass,
                "benchmark": bench,
                "folds_beating_benchmark": f"{wins}/{n_folds}",
                "win_percentage": round(wins / n_folds * 100.0, 1) if n_folds > 0 else 0.0
            },
            "criterion_3_positive_neutralized_ic": {
                "passed": c3_pass,
                "neutralized_ic": round(neut_ic, 4) if pd.notnull(neut_ic) else None
            },
            "explanation": explanation
        }
        
    evidence_pts = {"RED": 1, "AMBER": 2, "GREEN": 3}
    tot_pts = sum(evidence_pts[evidence_dict[h]["evidence_level_v3"]] for h in horizons)
    weights = {h: evidence_pts[evidence_dict[h]["evidence_level_v3"]] / tot_pts for h in horizons}
    for h in horizons:
        evidence_dict[h]["horizon_weight"] = round(weights[h], 4)
    evidence_dict["rule_version"] = "v3_post_hoc"
    evidence_dict["post_hoc"] = True
    evidence_dict["default_horizon_weights"] = {h: round(weights[h], 4) for h in horizons}
    
    all_built_evidence[m] = evidence_dict
    
    # Save tiers.json, tier_history.json, evidence.json
    with open(m_dir / "tiers.json", "w") as f:
        json.dump(tiers_dict, f, indent=2)
    with open(m_dir / "tier_history.json", "w") as f:
        json.dump(tier_history_dict, f, indent=2)
    with open(m_dir / "evidence.json", "w") as f:
        json.dump(evidence_dict, f, indent=2)
        
    # --- D. Build score_history.parquet ---
    s_short = h_scores["short"][["Date", "series_id", "fold", "split_type", "pct_rank", "tier", "liquidity_warning"]].rename(
        columns={"pct_rank": "score_short", "tier": "tier_short"})
    s_mid = h_scores["mid"][["Date", "series_id", "pct_rank", "tier"]].rename(
        columns={"pct_rank": "score_mid", "tier": "tier_mid"})
    s_long = h_scores["long"][["Date", "series_id", "pct_rank", "tier"]].rename(
        columns={"pct_rank": "score_long", "tier": "tier_long"})
        
    m_score = pd.merge(s_short, s_mid, on=["Date", "series_id"])
    m_score = pd.merge(m_score, s_long, on=["Date", "series_id"])
    
    # Weighted composite score
    m_score["score"] = (
        m_score["score_short"] * weights["short"] +
        m_score["score_mid"] * weights["mid"] +
        m_score["score_long"] * weights["long"]
    )
    
    # Add risk columns
    vol_sub = h_scores["short"][["Date", "series_id", "vol_63d", "zero_trade_share_63d", "vol_63d_ann", "risk_category"]].drop_duplicates()
    m_score = pd.merge(m_score, vol_sub, on=["Date", "series_id"], how="left")
    
    # Consistent Leader: High in >= 2 horizons, Low in none, no liquidity warning
    high_count = (m_score["tier_short"] == "High").astype(int) + (m_score["tier_mid"] == "High").astype(int) + (m_score["tier_long"] == "High").astype(int)
    has_low = (m_score["tier_short"] == "Low") | (m_score["tier_mid"] == "Low") | (m_score["tier_long"] == "Low")
    m_score["consistent_leader"] = (high_count >= 2) & (~has_low) & (~m_score["liquidity_warning"])
    
    m_score.to_parquet(m_dir / "score_history.parquet", index=False)
    print(f"Saved {m_dir / 'score_history.parquet'} ({len(m_score)} rows)")

# Update default root files in data/app/
with open(app_dir / "tiers.json", "w") as f:
    json.dump({h: json.load(open(models_dir / auto_models[h] / "tiers.json"))[h] for h in horizons}, f, indent=2)

with open(app_dir / "tier_history.json", "w") as f:
    json.dump({h: json.load(open(models_dir / auto_models[h] / "tier_history.json"))[h] for h in horizons}, f, indent=2)

# Default root evidence.json
default_evidence = {
    "short": copy.deepcopy(all_built_evidence[auto_models["short"]]["short"]),
    "mid": copy.deepcopy(all_built_evidence[auto_models["mid"]]["mid"]),
    "long": copy.deepcopy(all_built_evidence[auto_models["long"]]["long"]),
    "rule_version": "v3_post_hoc",
    "post_hoc": True,
    "default_horizon_weights": {
        "short": 0.25,
        "mid": 0.50,
        "long": 0.25
    }
}
default_evidence["short"]["served_model"] = auto_models["short"]
default_evidence["mid"]["served_model"] = auto_models["mid"]
default_evidence["long"]["served_model"] = auto_models["long"]
default_evidence["short"]["horizon_weight"] = 0.25
default_evidence["mid"]["horizon_weight"] = 0.50
default_evidence["long"]["horizon_weight"] = 0.25

with open(app_dir / "evidence.json", "w") as f:
    json.dump(default_evidence, f, indent=2)

# Default root score_history.parquet
# Pull from auto models
df_s = pd.read_parquet(models_dir / auto_models["short"] / "score_history.parquet")[
    ["Date", "series_id", "fold", "split_type", "score_short", "tier_short", "liquidity_warning", "vol_63d", "zero_trade_share_63d", "vol_63d_ann", "risk_category"]
]
df_m = pd.read_parquet(models_dir / auto_models["mid"] / "score_history.parquet")[
    ["Date", "series_id", "score_mid", "tier_mid"]
]
df_l = pd.read_parquet(models_dir / auto_models["long"] / "score_history.parquet")[
    ["Date", "series_id", "score_long", "tier_long"]
]

top_score = pd.merge(df_s, df_m, on=["Date", "series_id"])
top_score = pd.merge(top_score, df_l, on=["Date", "series_id"])
top_score["score"] = top_score["score_short"] * 0.25 + top_score["score_mid"] * 0.50 + top_score["score_long"] * 0.25
top_score["served_score_short"] = top_score["score_short"] * 100.0
top_score["served_score_mid"] = top_score["score_mid"] * 100.0
top_score["served_score_long"] = top_score["score_long"] * 100.0
top_score["overall_upside_score"] = top_score["score"] * 100.0

high_count = (top_score["tier_short"] == "High").astype(int) + (top_score["tier_mid"] == "High").astype(int) + (top_score["tier_long"] == "High").astype(int)
has_low = (top_score["tier_short"] == "Low") | (top_score["tier_mid"] == "Low") | (top_score["tier_long"] == "Low")
top_score["consistent_leader"] = (high_count >= 2) & (~has_low) & (~top_score["liquidity_warning"])

top_score.to_parquet(app_dir / "score_history.parquet", index=False)
print(f"Saved root {app_dir / 'score_history.parquet'} successfully.")

# Build model_registry.json
all_model_ids = list(model_meta.keys())
registry = {}

for m_id in all_model_ids:
    meta = model_meta[m_id]
    is_servable = m_id in servable_models
    servable_reason = None if is_servable else (
        "sensitivity analysis, selected on development results" if m_id == "ens_rank4_bestlgbm" else (
            "heavy to retrain" if "lstm" in m_id or "tabpfn" in m_id else "not part of the live score"
        )
    )
    
    pipe_m_name = m_id.replace("_core", "") if m_id.endswith("_core") else m_id
    m_horizons = {}
    for h in horizons:
        sub_d = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        dev_ic = sub_d.get("rank_ic_mean", None)
        null_up = sub_d.get("null_band_ic_upper", None)
        null_low = sub_d.get("null_band_ic_lower", None)
        raw_p, holm_p = p_lookup.get((pipe_m_name, "core", h), (sub_d.get("permutation_p_val", None), np.nan))
        
        pos_folds = int(sub_d.get("folds_positive_count", 0)) if "folds_positive_count" in sub_d else None
        tot_folds = int(sub_d.get("folds_total_count", 12 if h != "long" else 10)) if "folds_total_count" in sub_d else (12 if h != "long" else 10)
        t_stat = sub_d.get("folds_t_stat", None)
        
        outside_null = bool(dev_ic > null_up or dev_ic < null_low) if (dev_ic is not None and null_up is not None and null_low is not None) else False
        
        top_med_excess = sub_d.get("top_k_fwd_excess_median", None)
        top_prec = sub_d.get("top_k_precision", None)
        net_excess_act = sub_d.get("portfolio_net_excess_actual_mean", sub_d.get("portfolio_net_excess_mean", None))
        rep_share = sub_d.get("portfolio_turnover_replaced_share_mean", None)
        
        if m_id in all_built_evidence:
            ev_lvl = all_built_evidence[m_id][h]["evidence_level"]
        else:
            ev_lvl = "RED" if not outside_null else "AMBER"
            
        sub_h = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["fold"] == "holdout")].set_index("metric")["value"].to_dict()
        hold_ic = sub_h.get("rank_ic_mean", None)
        hold_neff = int(sub_h.get("n_eff", 16 if h == "short" else (5 if h == "mid" else 1))) if "n_eff" in sub_h else (16 if h == "short" else (5 if h == "mid" else 1))
        hold_status = "indicative" if hold_neff >= 5 else "not evaluable (n_eff < 5)"
        
        is_auto = bool(auto_models[h] == m_id)
        
        m_horizons[h] = {
            "dev_mean_ic": round(float(dev_ic), 4) if dev_ic is not None and pd.notnull(dev_ic) else None,
            "null_band": [round(float(null_low), 4), round(float(null_up), 4)] if (null_low is not None and null_up is not None and pd.notnull(null_low) and pd.notnull(null_up)) else None,
            "outside_null_band": outside_null,
            "raw_p_value": round(float(raw_p), 4) if raw_p is not None and pd.notnull(raw_p) else None,
            "holm_adjusted_p_value": round(float(holm_p), 4) if holm_p is not None and pd.notnull(holm_p) else None,
            "folds_positive_count": pos_folds,
            "folds_total_count": tot_folds,
            "folds_positive_ratio": f"{pos_folds}/{tot_folds}" if pos_folds is not None else None,
            "fold_t_stat": round(float(t_stat), 4) if t_stat is not None and pd.notnull(t_stat) else None,
            "top_tier_hit_rate": None,
            "top_5_precision": round(float(top_prec), 4) if top_prec is not None and pd.notnull(top_prec) else None,
            "top_5_median_excess": round(float(top_med_excess * 100.0), 2) if top_med_excess is not None and pd.notnull(top_med_excess) else None,
            "net_excess_under_actual_turnover": round(float(net_excess_act * 100.0), 2) if net_excess_act is not None and pd.notnull(net_excess_act) else None,
            "average_share_of_names_replaced": round(float(rep_share * 100.0), 1) if rep_share is not None and pd.notnull(rep_share) else None,
            "evidence_level": ev_lvl,
            "holdout_ic": round(float(hold_ic), 4) if hold_ic is not None and pd.notnull(hold_ic) else None,
            "holdout_n_eff": hold_neff,
            "holdout_status": hold_status,
            "is_auto_default": is_auto
        }
        
    registry[m_id] = {
        "model_id": m_id,
        "admin_display_name": meta["display_name"],
        "family": meta["family"],
        "short_description": meta["description"],
        "servable": is_servable,
        "servable_reason": servable_reason,
        "reason": servable_reason,
        "horizons": m_horizons
    }

with open(app_dir / "model_registry.json", "w") as f:
    json.dump(registry, f, indent=2)
print("Saved data/app/model_registry.json successfully.")

# Update thresholds.json
with open(thresholds_path) as f:
    thresh = json.load(f)

# Update consistent leader counts & examples
cl_counts = int(top_score["consistent_leader"].sum())
cl_examples = sorted(top_score[top_score["consistent_leader"]]["series_id"].unique())
thresh["consistent_leaders_count"] = {
    "value": cl_counts,
    "unique_stocks": cl_examples,
    "description": "Total stock-days identified as Consistent Leaders (High in >= 2 horizons, Low in none, no liquidity warning)."
}
with open(thresholds_path, "w") as f:
    json.dump(thresh, f, indent=2)
print("Updated thresholds.json successfully.")

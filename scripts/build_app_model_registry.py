"""scripts/build_app_model_registry.py

Builds per-model app directories under data/app/models/{model_id}/:
- evidence.json (Part B tightened rule + original definition)
- tiers.json (candidate fractions & cutoffs)
- tier_history.json (Part D labelling: n_periods, selected_on_dev, optimism_warning)
- score_history.parquet (out-of-sample scores, tiers, new liquidity warning, consistent leaders)

Also builds data/app/model_registry.json and updates default served files in data/app/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

base_dir = Path(__file__).resolve().parents[1]
app_dir = base_dir / "data/app"
models_dir = app_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# 1. Load configuration and master results
config_path = base_dir / "config.yaml"
results_path = base_dir / "reports/results_long.csv"
tilts_path = base_dir / "reports/07_tilts_neutralized.csv"
features_path = base_dir / "data/processed/features_targets.parquet"
mask_path = base_dir / "data/processed/rankable_mask.parquet"
oos_dir = base_dir / "reports/oos_predictions"

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

# Liquidity threshold: 0.8889
liq_threshold = 0.8889

horizons = ["short", "mid", "long"]
h_days_map = {"short": 21, "mid": 63, "long": 252}

servable_models = [
    "momentum_126",
    "ridge_core",
    "rf_reg_core",
    "lgbm_reg_core",
    "ens_rank4"
]

auto_models = {
    "short": "ens_rank4",
    "mid": "momentum_126",
    "long": "ens_rank4"
}

# Model registry descriptions
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
        "description": "3-month quarterly price momentum benchmark."
    },
    "low_vol": {
        "display_name": "Low Volatility",
        "family": "Baseline",
        "description": "Low-risk anomaly benchmark buying lowest realized volatility stocks."
    },
    "past_winner": {
        "display_name": "Past Long-Term Winner",
        "family": "Baseline",
        "description": "Historical expanding-window multi-year winner baseline."
    },
    "random": {
        "display_name": "Random Walk Scorer",
        "family": "Baseline",
        "description": "Zero-intelligence uniform random linear scorer null reference."
    },
    "equal_weight_universe": {
        "display_name": "Equal Weight Universe",
        "family": "Baseline",
        "description": "Passive equal-weighted ranking benchmark across all universe constituents."
    }
}

# Map model id to file prefix
def get_file_prefix(m_id):
    if m_id == "ridge_core": return "ridge"
    if m_id == "rf_reg_core": return "rf_reg"
    if m_id == "lgbm_reg_core": return "lgbm_reg"
    if m_id == "ens_rank4": return "ens_rank4"
    if m_id == "momentum_126": return "momentum_126"
    return m_id

print("Generating per-model app bundles for servable models...")

candidate_fractions = [0.10, 0.20, 0.25, 0.30]
evidence_pts = {"RED": 1, "AMBER": 2, "GREEN": 3}

for m in servable_models:
    m_dir = models_dir / m
    m_dir.mkdir(parents=True, exist_ok=True)
    
    file_prefix = get_file_prefix(m)
    
    # We will compute evidence, tiers, tier_history, and score_history
    evidence_dict = {}
    tiers_dict = {}
    tier_history_dict = {}
    
    # Store per-horizon score df
    h_scores = {}
    
    for h in horizons:
        # Load OOS predictions
        p_file = oos_dir / f"{file_prefix}_{h}_core.parquet"
        df_oos = pd.read_parquet(p_file)
        
        # Merge trade_day_share_63m
        df_oos = pd.merge(df_oos, df_tds, on=["series_id", "Date"], how="left")
        df_oos["liquidity_warning"] = (df_oos["trade_day_share_63m"] <= liq_threshold)
        
        # Ranks and Percentiles
        df_oos["rank"] = df_oos.groupby("Date")["pred_score"].rank(ascending=True, method="average")
        df_oos["n_stocks"] = df_oos.groupby("Date")["pred_score"].transform("count")
        df_oos["pct_rank"] = (df_oos["rank"] - 0.5) / df_oos["n_stocks"]
        df_oos["score"] = df_oos["pct_rank"] * 100.0
        
        # Median return and beat median
        df_oos["med_ret"] = df_oos.groupby("Date")[f"fwd_ret_{h}"].transform("median")
        df_oos["beat_med"] = np.where(df_oos[f"fwd_ret_{h}"] > df_oos["med_ret"], 1.0,
                             np.where(df_oos[f"fwd_ret_{h}"] == df_oos["med_ret"], 0.5, 0.0))
        
        df_dev = df_oos[df_oos["split_type"] == "dev"].copy()
        years = sorted(df_dev["fold"].unique())
        n_years = len(years)
        
        # --- A. Tiers derivation ---
        candidate_rows = []
        qualifying_fracs = []
        for frac in candidate_fractions:
            top_mask = df_dev["pct_rank"] >= (1.0 - frac)
            top_df = df_dev[top_mask]
            
            yearly_mean_lifts = []
            pos_years = 0
            yearly_lifts = {}
            for y in years:
                sub_y = top_df[top_df["fold"] == y]
                hr = sub_y["beat_med"].mean() if len(sub_y) > 0 else 0.5
                lft = (hr - 0.5) * 100.0
                yearly_lifts[y] = round(lft, 2)
                yearly_mean_lifts.append(lft)
                if lft > 0:
                    pos_years += 1
            pooled_hr = top_df["beat_med"].mean()
            pooled_lift = (pooled_hr - 0.5) * 100.0
            
            rng = np.random.default_rng(42)
            boot_lifts = [np.mean(rng.choice(yearly_mean_lifts, size=n_years, replace=True)) for _ in range(2000)]
            ci_low = float(np.percentile(boot_lifts, 5.0))
            ci_high = float(np.percentile(boot_lifts, 95.0))
            
            qualifies = (pos_years / n_years) >= (2/3)
            if qualifies:
                qualifying_fracs.append(frac)
            
            candidate_rows.append({
                "fraction": frac,
                "fraction_pct": f"{int(frac*100)}%",
                "pos_years": f"{pos_years}/{n_years}",
                "pos_years_pct": round(pos_years / n_years * 100.0, 1),
                "pooled_lift_pct": round(pooled_lift, 2),
                "ci_90": [round(ci_low, 2), round(ci_high, 2)],
                "qualifies": qualifies,
                "yearly_lifts": yearly_lifts
            })
            
        if qualifying_fracs:
            chosen_frac = max(qualifying_fracs)
            status_desc = "empirical"
            reason_desc = f"Selected largest qualifying top-tier fraction ({int(chosen_frac*100)}%) with positive lift in >= 2/3 dev years."
        else:
            chosen_frac = 0.25
            status_desc = "convention - no evidence-based cutoff"
            reason_desc = "No candidate fraction achieved positive lift in >= 2/3 dev years; default convention of 25% (top quartile) applied."
            
        tiers_dict[h] = {
            "chosen_fraction": chosen_frac,
            "status": status_desc,
            "reason": reason_desc,
            "candidates": candidate_rows
        }
        
        # --- B. Tier History with Part D labelling ---
        # Assign tiers
        df_oos["tier"] = np.where(df_oos["pct_rank"] >= (1.0 - chosen_frac), "High",
                         np.where(df_oos["pct_rank"] < chosen_frac, "Low", "Neutral"))
        h_scores[h] = df_oos
        
        h_tier_data = {}
        for s_type in ["dev", "holdout"]:
            sub = df_oos[df_oos["split_type"] == s_type].copy()
            dates = sorted(sub["Date"].unique())
            rebal_dates = set(dates[::h_days_map[h]])
            rebal_sub = sub[sub["Date"].isin(rebal_dates)].copy()
            n_eff = len(rebal_dates)
            
            tier_metrics = {}
            for t_name in ["High", "Neutral", "Low"]:
                t_df = rebal_sub[rebal_sub["tier"] == t_name]
                if t_df.empty:
                    continue
                hit_rate = float(t_df["beat_med"].mean()) * 100.0
                med_excess = float(t_df[f"fwd_excess_{h}"].median()) * 100.0
                
                if s_type == "dev":
                    y_beat_med = [grp["beat_med"].values for _, grp in t_df.groupby("fold")]
                    y_excess = [grp[f"fwd_excess_{h}"].values for _, grp in t_df.groupby("fold")]
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
                    "num_periods": n_eff,
                    "n_periods": n_eff,
                    "selected_on_dev": sel_dev,
                    "optimism_warning": opt_warning,
                    "hit_rate_pct": round(hit_rate, 2),
                    "hit_rate_90_ci": hr_ci,
                    "median_excess_pct": round(med_excess, 2),
                    "median_excess_90_ci": med_ci,
                    "evaluation_flag": flag
                }
            h_tier_data[s_type] = tier_metrics
        tier_history_dict[h] = h_tier_data
        
        # --- C. Evidence Level (Part B Rule) ---
        pipe_m_name = m.replace("_core", "") if m.endswith("_core") else m
        sub_res = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        ic_val = sub_res.get("rank_ic_mean", np.nan)
        null_up = sub_res.get("null_band_ic_upper", np.nan)
        null_low = sub_res.get("null_band_ic_lower", np.nan)
        raw_p = sub_res.get("permutation_p_val", np.nan)
        
        # Criterion 1
        c1_pass = bool(ic_val > null_up or ic_val < null_low)
        
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
        neut_ic = float(t_sub["neutralized_ic"].iloc[0]) if not t_sub.empty else np.nan
        c3_pass = bool(neut_ic > 0.0)
        
        # Original Definition Level
        criteria_passed_count = int(c1_pass) + int(c2_pass) + int(c3_pass)
        if criteria_passed_count == 3:
            lvl_orig = "GREEN"
        elif criteria_passed_count in [1, 2]:
            lvl_orig = "AMBER"
        else:
            lvl_orig = "RED"
            
        # Tightened Rule (Post-Hoc)
        if not c1_pass:
            lvl_tight = "RED"
        elif c1_pass and (c2_pass or c3_pass):
            if c2_pass and c3_pass:
                lvl_tight = "GREEN"
            else:
                lvl_tight = "AMBER"
        else:
            lvl_tight = "RED"
            
        confidence_labels = {
            "RED": "Very low confidence",
            "AMBER": "Low confidence",
            "GREEN": "Moderate confidence"
        }
        conf_lbl = confidence_labels[lvl_tight]
        
        # Holm p-value approximation from master tables
        holm_p = min(1.0, raw_p * 69) if pd.notnull(raw_p) else np.nan
        
        explanation = f"{conf_lbl}: Criterion 1 (outside null band) {'passed' if c1_pass else 'failed'} (raw p={raw_p:.3f}), Criterion 2 (beats naive baseline) {'passed' if c2_pass else 'failed'} ({wins}/{n_folds} folds beat {bench}), Criterion 3 (style-neutralized IC > 0) {'passed' if c3_pass else 'failed'} (neutralized IC = {neut_ic:+.4f})."
        
        evidence_dict[h] = {
            "rule_version": "v2_tightened_post_hoc",
            "post_hoc": True,
            "model_id": m,
            "horizon": h,
            "evidence_level": lvl_tight,
            "evidence_level_tightened": lvl_tight,
            "evidence_level_original": lvl_orig,
            "evidence_points": evidence_pts[lvl_tight],
            "confidence_label": conf_lbl,
            "raw_p_value": round(float(raw_p), 4) if pd.notnull(raw_p) else None,
            "holm_adjusted_p_value": round(float(holm_p), 4) if pd.notnull(holm_p) else None,
            "criterion_1_outside_null_band": {
                "passed": c1_pass,
                "dev_mean_ic": round(float(ic_val), 4),
                "null_band": [round(float(null_low), 4), round(float(null_up), 4)]
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
        
    # Default horizon weights per model
    evidence_pts = {"RED": 1, "AMBER": 2, "GREEN": 3}
    tot_pts = sum(evidence_pts[evidence_dict[h]["evidence_level_tightened"]] for h in horizons)
    weights = {h: evidence_pts[evidence_dict[h]["evidence_level_tightened"]] / tot_pts for h in horizons}
    for h in horizons:
        evidence_dict[h]["horizon_weight"] = round(weights[h], 4)
    evidence_dict["rule_version"] = "v2_tightened_post_hoc"
    evidence_dict["post_hoc"] = True
    evidence_dict["default_horizon_weights"] = {h: round(weights[h], 4) for h in horizons}
    
    # Save tiers.json, tier_history.json, evidence.json
    with open(m_dir / "tiers.json", "w") as f:
        json.dump(tiers_dict, f, indent=2)
    with open(m_dir / "tier_history.json", "w") as f:
        json.dump(tier_history_dict, f, indent=2)
    with open(m_dir / "evidence.json", "w") as f:
        json.dump(evidence_dict, f, indent=2)
        
    # --- D. Build score_history.parquet ---
    s_short = h_scores["short"][["Date", "series_id", "fold", "split_type", "score", "tier", "liquidity_warning"]].rename(
        columns={"score": "score_short", "tier": "tier_short"})
    s_mid = h_scores["mid"][["Date", "series_id", "score", "tier"]].rename(
        columns={"score": "score_mid", "tier": "tier_mid"})
    s_long = h_scores["long"][["Date", "series_id", "score", "tier"]].rename(
        columns={"score": "score_long", "tier": "tier_long"})
        
    m_score = pd.merge(s_short, s_mid, on=["Date", "series_id"])
    m_score = pd.merge(m_score, s_long, on=["Date", "series_id"])
    
    # Merge vol_63d and risk categories
    vol_p33 = float(df_feat[df_feat["in_ranking_universe"] & df_feat["rankable"] & (df_feat["Date"] >= "2008-01-01") & (df_feat["Date"] <= "2019-12-31")]["vol_63d"].quantile(0.333))
    vol_p67 = float(df_feat[df_feat["in_ranking_universe"] & df_feat["rankable"] & (df_feat["Date"] >= "2008-01-01") & (df_feat["Date"] <= "2019-12-31")]["vol_63d"].quantile(0.667))
    m_score = m_score.merge(df_feat[["Date", "series_id", "vol_63d", "zero_trade_share_63d"]], on=["Date", "series_id"], how="left")
    m_score["vol_63d_ann"] = m_score["vol_63d"]
    m_score["risk_category"] = np.where(m_score["vol_63d"] <= vol_p33, "Low",
                               np.where(m_score["vol_63d"] <= vol_p67, "Medium", "High"))
    
    w_s = weights["short"]
    w_m = weights["mid"]
    w_l = weights["long"]
    m_score["served_score_short"] = m_score["score_short"]
    m_score["served_score_mid"] = m_score["score_mid"]
    m_score["served_score_long"] = m_score["score_long"]
    m_score["overall_upside_score"] = (w_s * m_score["score_short"] +
                                       w_m * m_score["score_mid"] +
                                       w_l * m_score["score_long"])
                                       
    high_count = ((m_score["tier_short"] == "High").astype(int) +
                  (m_score["tier_mid"] == "High").astype(int) +
                  (m_score["tier_long"] == "High").astype(int))
    has_low = ((m_score["tier_short"] == "Low") |
               (m_score["tier_mid"] == "Low") |
               (m_score["tier_long"] == "Low"))
    m_score["consistent_leader"] = (high_count >= 2) & (~has_low) & (~m_score["liquidity_warning"])
    
    m_score.to_parquet(m_dir / "score_history.parquet", index=False)
    print(f"Saved {m_dir} app bundle files successfully.")

# -------------------------------------------------------------
# 2. Build Top-Level Served Files for Default Auto Selection
# -------------------------------------------------------------
print("\nUpdating top-level data/app files for Auto served ensemble...")
# Auto selection per horizon: Short = ens_rank4, Mid = momentum_126, Long = ens_rank4
auto_evidence = {}
auto_tiers = {}
auto_tier_history = {}

# Load from respective models
for h, m_chosen in auto_models.items():
    with open(models_dir / m_chosen / "evidence.json", "r") as f:
        ev = json.load(f)
        auto_evidence[h] = ev[h]
        auto_evidence[h]["served_model"] = m_chosen
    with open(models_dir / m_chosen / "tiers.json", "r") as f:
        tr = json.load(f)
        auto_tiers[h] = tr[h]
    with open(models_dir / m_chosen / "tier_history.json", "r") as f:
        th = json.load(f)
        auto_tier_history[h] = th[h]

# Default weights for Auto served model
auto_pts = sum(evidence_pts[auto_evidence[h]["evidence_level_tightened"]] for h in horizons)
auto_weights = {h: evidence_pts[auto_evidence[h]["evidence_level_tightened"]] / auto_pts for h in horizons}
for h in horizons:
    auto_evidence[h]["horizon_weight"] = round(auto_weights[h], 4)
auto_evidence["rule_version"] = "v2_tightened_post_hoc"
auto_evidence["post_hoc"] = True
auto_evidence["default_horizon_weights"] = {h: round(auto_weights[h], 4) for h in horizons}

with open(app_dir / "evidence.json", "w") as f:
    json.dump(auto_evidence, f, indent=2)
with open(app_dir / "tiers.json", "w") as f:
    json.dump(auto_tiers, f, indent=2)
with open(app_dir / "tier_history.json", "w") as f:
    json.dump(auto_tier_history, f, indent=2)

# Build top-level score_history.parquet using chosen auto models per horizon
df_short = pd.read_parquet(models_dir / auto_models["short"] / "score_history.parquet")[["Date", "series_id", "fold", "split_type", "score_short", "tier_short", "liquidity_warning", "vol_63d", "zero_trade_share_63d", "vol_63d_ann", "risk_category"]]
df_mid = pd.read_parquet(models_dir / auto_models["mid"] / "score_history.parquet")[["Date", "series_id", "score_mid", "tier_mid"]]
df_long = pd.read_parquet(models_dir / auto_models["long"] / "score_history.parquet")[["Date", "series_id", "score_long", "tier_long"]]

top_score = pd.merge(df_short, df_mid, on=["Date", "series_id"])
top_score = pd.merge(top_score, df_long, on=["Date", "series_id"])

top_score["served_score_short"] = top_score["score_short"]
top_score["served_score_mid"] = top_score["score_mid"]
top_score["served_score_long"] = top_score["score_long"]
top_score["overall_upside_score"] = (auto_weights["short"] * top_score["score_short"] +
                                     auto_weights["mid"] * top_score["score_mid"] +
                                     auto_weights["long"] * top_score["score_long"])

high_count = ((top_score["tier_short"] == "High").astype(int) +
              (top_score["tier_mid"] == "High").astype(int) +
              (top_score["tier_long"] == "High").astype(int))
has_low = ((top_score["tier_short"] == "Low") |
           (top_score["tier_mid"] == "Low") |
           (top_score["tier_long"] == "Low"))
top_score["consistent_leader"] = (high_count >= 2) & (~has_low) & (~top_score["liquidity_warning"])

top_score.to_parquet(app_dir / "score_history.parquet", index=False)
print("Saved top-level data/app served files.")

# -------------------------------------------------------------
# 3. Build data/app/model_registry.json
# -------------------------------------------------------------
print("\nBuilding data/app/model_registry.json...")

all_model_ids = list(model_meta.keys())
registry = {}

for m_id in all_model_ids:
    meta = model_meta[m_id]
    is_servable = m_id in servable_models
    servable_reason = None if is_servable else ("heavy to retrain" if "lstm" in m_id or "tabpfn" in m_id else "not part of the live score")
    
    pipe_m_name = m_id.replace("_core", "") if m_id.endswith("_core") else m_id
    
    m_horizons = {}
    for h in horizons:
        # Dev metrics
        sub_d = res_df[(res_df["model"] == pipe_m_name) & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        dev_ic = sub_d.get("rank_ic_mean", None)
        null_up = sub_d.get("null_band_ic_upper", None)
        null_low = sub_d.get("null_band_ic_lower", None)
        raw_p = sub_d.get("permutation_p_val", None)
        holm_p = min(1.0, raw_p * 69) if raw_p is not None else None
        
        pos_folds = int(sub_d.get("folds_positive_count", 0)) if "folds_positive_count" in sub_d else None
        tot_folds = int(sub_d.get("folds_total_count", 12 if h != "long" else 10)) if "folds_total_count" in sub_d else (12 if h != "long" else 10)
        t_stat = sub_d.get("folds_t_stat", None)
        
        # Outside null band
        outside_null = bool(dev_ic > null_up or dev_ic < null_low) if (dev_ic is not None and null_up is not None and null_low is not None) else False
        
        # Portfolio metrics
        top_med_excess = sub_d.get("top_k_fwd_excess_median", None)
        top_prec = sub_d.get("top_k_precision", None)
        net_excess_act = sub_d.get("portfolio_net_excess_actual_mean", sub_d.get("portfolio_net_excess_mean", None))
        rep_share = sub_d.get("portfolio_turnover_replaced_share_mean", None)
        
        # Top-tier hit rate
        if is_servable:
            th_data = tier_history_dict[h]["dev"]["High"] if m_id in servable_models else None
            top_hr = th_data["hit_rate_pct"] if th_data else None
            top_hr_ci = th_data["hit_rate_90_ci"] if th_data else None
            ev_lvl = evidence_dict[h]["evidence_level_tightened"] if m_id in servable_models else None
        else:
            top_hr = None
            top_hr_ci = None
            ev_lvl = "RED" if not outside_null else "AMBER"
            
        # Holdout metrics
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
            "top_tier_hit_rate": round(float(top_hr), 2) if top_hr is not None and pd.notnull(top_hr) else None,
            "top_tier_hit_rate_90_ci": top_hr_ci,
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
        "horizons": m_horizons
    }

with open(app_dir / "model_registry.json", "w") as f:
    json.dump(registry, f, indent=2)
print("Saved data/app/model_registry.json successfully.")

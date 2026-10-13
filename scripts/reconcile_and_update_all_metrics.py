"""scripts/reconcile_and_update_all_metrics.py
Recomputes and reconciles:
1. Standardized top-tier hit rate across all models using src.scoring.hit_rate.
2. Per-fold annual IC for admin charts.
3. True validated ens_rank4 metrics from frozen OOS parquets vs production config.
4. Updates model_registry.json, tier_history.json, methodology files, and thresholds.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.scoring.hit_rate import compute_top_tier_hit_rate
APP_DIR = ROOT / "data/app"
OOS_FROZEN_DIR = ROOT / "reports/oos_predictions"
OOS_PRODCFG_DIR = ROOT / "reports/oos_predictions_prodcfg"
REG_PATH = APP_DIR / "model_registry.json"
THRESH_PATH = ROOT / "thresholds.json"

with open(REG_PATH, "r", encoding="utf-8") as f:
    registry = json.load(f)

with open(THRESH_PATH, "r", encoding="utf-8") as f:
    thresholds = json.load(f)

cutoffs = {
    "short": float(thresholds["tier_cutoff_short"]["value"]),
    "mid": float(thresholds["tier_cutoff_mid"]["value"]),
    "long": float(thresholds["tier_cutoff_long"]["value"])
}

horizons = ["short", "mid", "long"]
h_days_map = {"short": 21, "mid": 63, "long": 252}

# 1. Update hit rates and per_fold_ic for all models in registry
for m_id, m_info in registry.items():
    for h in horizons:
        # Determine frozen parquet path
        p_path = OOS_FROZEN_DIR / f"{m_id}_{h}_core.parquet"
        if not p_path.exists():
            continue

        df = pd.read_parquet(p_path)
        h_col = f"fwd_ret_{h}"

        # Per-fold IC
        fold_ics = {}
        for f_name, f_grp in df.groupby("fold"):
            d_ics = [spearmanr(g["pred_score"], g[h_col])[0] for _, g in f_grp.groupby("Date") if len(g) >= 8]
            if d_ics:
                fold_ics[f_name] = round(float(np.mean(d_ics)), 4)

        # Standardized top-tier hit rate on dev split
        hr_data = compute_top_tier_hit_rate(df, h, cutoff=cutoffs[h])

        # Update in registry
        h_reg = m_info["horizons"][h]
        h_reg["per_fold_ic"] = fold_ics
        if hr_data["hit_rate_pct"] is not None:
            h_reg["top_tier_hit_rate"] = hr_data["hit_rate_pct"]
            h_reg["top_tier_hit_rate_90_ci"] = hr_data["hit_rate_90_ci"]

# 2. Update production configuration metrics for production models
prod_models = ["ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
for m_id in prod_models:
    for h in horizons:
        p_path = OOS_PRODCFG_DIR / f"{m_id}_{h}_core.parquet"
        if not p_path.exists():
            continue

        df_prod = pd.read_parquet(p_path)
        dev_prod = df_prod[df_prod["split_type"] == "dev"]
        h_col = f"fwd_ret_{h}"

        # Dev fold ICs
        prod_fold_ics = []
        for f_name, f_grp in dev_prod.groupby("fold"):
            d_ics = [spearmanr(g["pred_score"], g[h_col])[0] for _, g in f_grp.groupby("Date") if len(g) >= 8]
            if d_ics:
                prod_fold_ics.append(np.mean(d_ics))

        n_folds = len(prod_fold_ics)
        dev_mean_ic = float(np.mean(prod_fold_ics))
        dev_std_ic = float(np.std(prod_fold_ics, ddof=1)) if n_folds > 1 else 0.0
        fold_t_stat = (dev_mean_ic / (dev_std_ic / np.sqrt(n_folds))) if dev_std_ic > 1e-9 else 0.0
        pos_folds = sum(1 for x in prod_fold_ics if x > 0)

        # Top-tier hit rate
        prod_hr = compute_top_tier_hit_rate(df_prod, h, cutoff=cutoffs[h])

        # Holdout metrics
        hold_df = df_prod[df_prod["split_type"] == "holdout"]
        if not hold_df.empty:
            hold_ics = [spearmanr(g["pred_score"], g[h_col])[0] for _, g in hold_df.groupby("Date") if len(g) >= 8]
            hold_ic = float(np.mean(hold_ics)) if hold_ics else None
            n_eff = int(round(hold_df["Date"].nunique() / h_days_map[h]))
        else:
            hold_ic = None
            n_eff = 0

        val_h = registry[m_id]["horizons"][h]
        val_mean = val_h["dev_mean_ic"]
        val_t = val_h["fold_t_stat"]
        val_pos = val_h["folds_positive_count"]
        val_hr = val_h.get("top_tier_hit_rate")

        diff_ic = dev_mean_ic - val_mean
        diff_t = fold_t_stat - val_t
        diff_pos = pos_folds - val_pos
        diff_hr = (prod_hr["hit_rate_pct"] - val_hr) if (prod_hr["hit_rate_pct"] is not None and val_hr is not None) else None

        prod_cfg_dict = {
            "dev_mean_ic": round(dev_mean_ic, 4),
            "fold_t_stat": round(fold_t_stat, 4),
            "folds_positive_count": pos_folds,
            "folds_total_count": n_folds,
            "folds_positive_ratio": f"{pos_folds}/{n_folds}",
            "top_tier_hit_rate": prod_hr["hit_rate_pct"],
            "top_tier_hit_rate_90_ci": prod_hr["hit_rate_90_ci"],
            "holdout_ic": round(hold_ic, 4) if hold_ic is not None else None,
            "holdout_n_eff": n_eff,
            "holdout_status": "indicative" if n_eff >= 5 else "not evaluable (n_eff < 5)",
            "difference_vs_validated": {
                "diff_dev_mean_ic": round(diff_ic, 4),
                "diff_fold_t_stat": round(diff_t, 4),
                "diff_folds_positive_count": diff_pos,
                "diff_top_tier_hit_rate": round(diff_hr, 4) if diff_hr is not None else None
            }
        }
        val_h["production_config_metrics"] = prod_cfg_dict

# Save updated registry
with open(REG_PATH, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)
print(f"Updated {REG_PATH}")

# 3. Synchronize tier_history.json for ens_rank4 and other models
for m_id in ["ens_rank4", "momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core"]:
    th_path = APP_DIR / "models" / m_id / "tier_history.json"
    if th_path.exists():
        with open(th_path, "r", encoding="utf-8") as f:
            th_data = json.load(f)
        for h in horizons:
            reg_h = registry[m_id]["horizons"][h]
            if "dev" in th_data.get(h, {}) and "High" in th_data[h]["dev"].get("tiers", {}):
                th_data[h]["dev"]["tiers"]["High"]["hit_rate_pct"] = reg_h["top_tier_hit_rate"]
                if reg_h.get("top_tier_hit_rate_90_ci"):
                    th_data[h]["dev"]["tiers"]["High"]["hit_rate_90ci"] = reg_h["top_tier_hit_rate_90_ci"]
        with open(th_path, "w", encoding="utf-8") as f:
            json.dump(th_data, f, indent=2)

# Also sync root data/app/tier_history.json (ens_rank4)
root_th_path = APP_DIR / "tier_history.json"
with open(root_th_path, "r", encoding="utf-8") as f:
    root_th = json.load(f)
for h in horizons:
    ens_h = registry["ens_rank4"]["horizons"][h]
    if "dev" in root_th.get(h, {}) and "High" in root_th[h]["dev"].get("tiers", {}):
        root_th[h]["dev"]["tiers"]["High"]["hit_rate_pct"] = ens_h["top_tier_hit_rate"]
        if ens_h.get("top_tier_hit_rate_90_ci"):
            root_th[h]["dev"]["tiers"]["High"]["hit_rate_90ci"] = ens_h["top_tier_hit_rate_90_ci"]
with open(root_th_path, "w", encoding="utf-8") as f:
    json.dump(root_th, f, indent=2)
print("Synchronized tier_history.json files.")

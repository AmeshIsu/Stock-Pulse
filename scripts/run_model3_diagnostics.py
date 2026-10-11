"""Compute Model 3 (LightGBM) and Ensemble diagnostics: factor tilts, neutralized IC, market up/down splits, calibration deciles."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.metrics import compute_rank_ic, simulate_rebalancing_portfolio

def run_diagnostics():
    config = load_config()
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df_feat = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df_feat[df_feat["in_ranking_universe"] & mask].copy()
    
    # Pre-merge style factors: volatility, liquidity, momentum
    df_u["cs_vol"] = df_u.groupby("Date")["vol_63d"].rank(pct=True)
    df_u["cs_liq"] = df_u.groupby("Date")["log_trade_vol_21d"].rank(pct=True)
    df_u["cs_mom"] = df_u.groupby("Date")["log_ret_63"].rank(pct=True)
    
    style_cols = ["Date", "series_id", "cs_vol", "cs_liq", "cs_mom"]
    df_style = df_u[style_cols].drop_duplicates()
    
    candidate_models = [
        "lgbm_reg_core",
        "lgbm_reg_full",
        "lgbm_reg_core_rel",
        "lgbm_clf_core",
        "lgbm_clf_core_rel",
        "ens_rank4",
        "rf_reg_core",
        "ridge_core",
        "momentum_126_core"
    ]
    
    tilt_rows = []
    updown_rows = []
    calib_rows = []
    
    for h in ["short", "mid", "long"]:
        h_days = config["horizons"][h]
        for m_name in candidate_models:
            # Parse model and fset
            parts = m_name.rsplit("_", 1)
            model_base, fset = parts[0], parts[1]
            if model_base.startswith("momentum_"):
                oos_file = oos_dir / f"{model_base}_{h}_{fset}.parquet"
            else:
                oos_file = oos_dir / f"{model_base}_{h}_{fset}.parquet"
                
            if not oos_file.exists():
                continue
                
            df_oos = pd.read_parquet(oos_file)
            merged = df_oos.merge(df_style, on=["Date", "series_id"], how="inner")
            
            # Tilts & Neutralized IC for DEV and HOLDOUT
            for s_type in ["dev", "holdout"]:
                sub = merged[merged["split_type"] == s_type].copy()
                if sub.empty:
                    continue
                    
                # 1. Tilts
                vol_tilts, liq_tilts, mom_tilts = [], [], []
                raw_ics, neut_ics = [], []
                
                for dt, grp in sub.groupby("Date"):
                    valid = grp.dropna(subset=["pred_score", f"fwd_excess_{h}", "cs_vol", "cs_liq", "cs_mom"])
                    if len(valid) < 8:
                        continue
                    
                    scores = valid["pred_score"].values
                    # Spearman correlations
                    r_vol, _ = spearmanr(scores, valid["cs_vol"].values)
                    r_liq, _ = spearmanr(scores, valid["cs_liq"].values)
                    r_mom, _ = spearmanr(scores, valid["cs_mom"].values)
                    if pd.notnull(r_vol): vol_tilts.append(r_vol)
                    if pd.notnull(r_liq): liq_tilts.append(r_liq)
                    if pd.notnull(r_mom): mom_tilts.append(r_mom)
                    
                    # Raw IC
                    r_raw, _ = spearmanr(scores, valid[f"fwd_excess_{h}"].values)
                    if pd.notnull(r_raw): raw_ics.append(r_raw)
                    
                    # Neutralized IC: residualize score against styles
                    X_mat = valid[["cs_vol", "cs_liq", "cs_mom"]].values
                    y_score = scores
                    reg = LinearRegression().fit(X_mat, y_score)
                    resid = y_score - reg.predict(X_mat)
                    
                    r_neut, _ = spearmanr(resid, valid[f"fwd_excess_{h}"].values)
                    if pd.notnull(r_neut): neut_ics.append(r_neut)
                    
                tilt_rows.append({
                    "horizon": h,
                    "model": m_name,
                    "split_type": s_type,
                    "raw_ic": float(np.mean(raw_ics)) if raw_ics else np.nan,
                    "neutralized_ic": float(np.mean(neut_ics)) if neut_ics else np.nan,
                    "tilt_vol": float(np.mean(vol_tilts)) if vol_tilts else np.nan,
                    "tilt_liq": float(np.mean(liq_tilts)) if liq_tilts else np.nan,
                    "tilt_mom": float(np.mean(mom_tilts)) if mom_tilts else np.nan
                })
                
                # 2. Market Up/Down Split on DEV
                if s_type == "dev":
                    mkt_rets = sub.groupby("Date")[f"fwd_ret_{h}"].mean()
                    up_dates = set(mkt_rets[mkt_rets > 0].index)
                    down_dates = set(mkt_rets[mkt_rets <= 0].index)
                    
                    ic_up = compute_rank_ic(sub[sub["Date"].isin(up_dates)], target_col=f"fwd_excess_{h}", min_stocks=8)
                    ic_down = compute_rank_ic(sub[sub["Date"].isin(down_dates)], target_col=f"fwd_excess_{h}", min_stocks=8)
                    
                    updown_rows.append({
                        "horizon": h,
                        "model": m_name,
                        "ic_up_market": float(ic_up.mean()) if len(ic_up) > 0 else np.nan,
                        "ic_down_market": float(ic_down.mean()) if len(ic_down) > 0 else np.nan,
                        "n_dates_up": len(ic_up),
                        "n_dates_down": len(ic_down)
                    })
                    
            # 3. Calibration deciles for classifiers
            if "clf" in m_name:
                for s_type in ["dev", "holdout"]:
                    sub_clf = df_oos[df_oos["split_type"] == s_type].dropna(subset=["pred_prob", f"outperform_{h}"]).copy()
                    if not sub_clf.empty and len(sub_clf) >= 50:
                        try:
                            sub_clf["prob_bin"] = pd.qcut(sub_clf["pred_prob"], q=5, duplicates="drop")
                            cal_grp = sub_clf.groupby("prob_bin", observed=True).agg(
                                mean_prob=("pred_prob", "mean"),
                                actual_rate=(f"outperform_{h}", "mean"),
                                count=("pred_prob", "count")
                            ).reset_index()
                            for _, r in cal_grp.iterrows():
                                calib_rows.append({
                                    "horizon": h,
                                    "model": m_name,
                                    "split_type": s_type,
                                    "bin": str(r["prob_bin"]),
                                    "mean_prob": float(r["mean_prob"]),
                                    "actual_rate": float(r["actual_rate"]),
                                    "count": int(r["count"])
                                })
                        except Exception as e:
                            pass

    df_tilts = pd.DataFrame(tilt_rows)
    df_updown = pd.DataFrame(updown_rows)
    df_calib = pd.DataFrame(calib_rows)
    
    df_tilts.to_csv(base_dir / "reports/06_tilts_neutralized.csv", index=False)
    df_updown.to_csv(base_dir / "reports/06_updown_market_split.csv", index=False)
    df_calib.to_csv(base_dir / "reports/06_classifier_calibration.csv", index=False)
    print("Diagnostics computed and saved successfully.")

if __name__ == "__main__":
    run_diagnostics()

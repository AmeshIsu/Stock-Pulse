"""src/scoring/score_latest.py

Institutional live scoring engine for Stock Pulse.
Computes cross-sectional upside percentiles, tiers, risk categories,
liquidity warnings, drivers, and technical chart indicators as of a target date.
Outputs serving artifacts under data/serving/{as_of}/.
"""
import sys
import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.production.train import reasons, MODELS_DIR
from src.models.preprocessing import transform_preprocessor, apply_cross_sectional_rank, REDUNDANT_LINEAR_FEATURES
from src.scoring.tiers import assign_tiers, is_consistent_leader

SERVING_DIR = base_dir / "data" / "serving"


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(prices: pd.Series, span_fast: int = 12, span_slow: int = 26, span_signal: int = 9):
    ema_fast = prices.ewm(span=span_fast, adjust=False).mean()
    ema_slow = prices.ewm(span=span_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = prices.rolling(window=window, min_periods=window).mean()
    rstd = prices.rolling(window=window, min_periods=window).std()
    upper = sma + (rstd * num_std)
    lower = sma - (rstd * num_std)
    return upper, sma, lower


def compute_percentiles(raw_s: pd.Series) -> pd.Series:
    """Compute percentile rankings in [0, 100] using midpoints with average rank for ties: ((rank - 0.5) / N) * 100."""
    if raw_s.empty:
        return pd.Series(dtype=float)
    N = len(raw_s)
    rk = raw_s.rank(method="average", ascending=True)
    return ((rk - 0.5) / N) * 100.0


def compute_ens_rank4_from_components(df_comp: pd.DataFrame) -> pd.Series:
    """Compute ens_rank4 scores from components using average ranks for ties."""
    rk_ridge = df_comp["score_ridge"].rank(pct=True, method="average")
    rk_rf = df_comp["score_rf"].rank(pct=True, method="average")
    rk_lgbm = df_comp["score_lgbm"].rank(pct=True, method="average")
    rk_mom = df_comp["score_mom"].rank(pct=True, method="average")
    raw_ens = (rk_ridge + rk_rf + rk_lgbm + rk_mom) / 4.0
    return compute_percentiles(raw_ens)


def load_active_models(serving_dir: Path | str | None = None) -> dict:
    """Load active models per horizon from data/serving/active_model_defaults.json."""
    if serving_dir is None:
        p = SERVING_DIR / "active_model_defaults.json"
    else:
        p = Path(serving_dir) / "active_model_defaults.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("active_models", {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"})
    return {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"}


def evaluate_volatility_note(median_vol: float | pd.Series, cutoff: float, note_text: str = "Market-wide volatility is elevated.") -> str | None:
    """Evaluate conditional volatility note based on 66.7th percentile cutoff."""
    if isinstance(median_vol, pd.Series):
        val = float(median_vol.median())
    else:
        val = float(median_vol)
    if val > cutoff:
        return note_text
    return None



def score_as_of(
    as_of_date: str | None = None,
    custom_weights: dict | None = None,
    config: dict | None = None,
    out_serving_dir: Path | str | None = None
) -> dict:
    """Generate complete live scoring artifacts for an as-of date.
    Default as-of date: last date in data/processed/features_targets.parquet.
    """
    if config is None:
        config = load_config()
        
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    prices_path = base_dir / config["paths"]["clean_prices"]
    thresholds_path = base_dir / "thresholds.json"
    
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    df_feat = pd.read_parquet(features_path)
    df_mask = pd.read_parquet(mask_path)
    df_feat["rankable"] = df_mask["rankable"].values
    
    df_u = df_feat[df_feat["in_ranking_universe"]].copy()
    all_dates = sorted(df_u["Date"].astype(str).str[:10].unique())
    
    if as_of_date is None:
        target_as_of = all_dates[-1]
    else:
        target_as_of = str(as_of_date)[:10]
        if target_as_of not in all_dates:
            # Pick closest previous date
            prev_dates = [d for d in all_dates if d <= target_as_of]
            if not prev_dates:
                raise ValueError(f"As-of date {target_as_of} precedes dataset history.")
            target_as_of = prev_dates[-1]
            
    print(f"Executing Scoring Engine: Data as of {target_as_of}")
    
    # 21 market days earlier for rank change comparison
    curr_idx = all_dates.index(target_as_of)
    prev_21_idx = max(0, curr_idx - 21)
    prev_21_date = all_dates[prev_21_idx]
    
    data_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()
    data_version = f"v_{target_as_of.replace('-', '')}_{data_hash[:8]}"
    
    if out_serving_dir is None:
        out_dir = SERVING_DIR / target_as_of
    else:
        out_dir = Path(out_serving_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_out_dir = out_dir / "prices"
    prices_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check universe series
    universe_series = config.get("universe_series", sorted(df_u["series_id"].unique()))
    
    # Load thresholds
    vol_low = thresholds["risk_category_low_cutoff"]["value"]
    vol_high = thresholds["risk_category_high_cutoff"]["value"]
    liq_cutoff = thresholds["liquidity_warning_cutoff"]["value"]
    stretched_series = thresholds.get("thin_trading_flag", {}).get("flagged_series", ["HNB.N"])
    
    # Load evidence & active defaults
    with open(base_dir / "data/app/evidence.json", "r", encoding="utf-8") as f:
        root_evidence = json.load(f)
        
    def_weights = root_evidence.get("default_horizon_weights", {"short": 0.25, "mid": 0.50, "long": 0.25})
    if custom_weights:
        w_sum = sum(custom_weights.values())
        active_weights = {k: v / w_sum for k, v in custom_weights.items()}
    else:
        active_weights = def_weights
        
    horizons = ["short", "mid", "long"]
    servable_models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    feature_cols = list(config["feature_sets"]["core"])
    
    # Load trained production models
    loaded_models = {}
    for m_id in ["ridge_core", "rf_reg_core", "lgbm_reg_core"]:
        loaded_models[m_id] = {}
        for h in horizons:
            m_path = MODELS_DIR / m_id / h / "model.pkl"
            p_path = MODELS_DIR / m_id / h / "preprocessor.pkl"
            if m_path.exists() and p_path.exists():
                with open(m_path, "rb") as f:
                    m_obj = pickle.load(f)
                with open(p_path, "rb") as f:
                    p_obj = pickle.load(f)
                loaded_models[m_id][h] = {"model": m_obj, "prep": p_obj}
                
    # Load tier definitions per model
    model_tiers = {}
    model_evidence = {}
    for m_id in servable_models:
        t_path = base_dir / "data/app/models" / m_id / "tiers.json"
        e_path = base_dir / "data/app/models" / m_id / "evidence.json"
        if t_path.exists():
            with open(t_path, "r", encoding="utf-8") as f:
                model_tiers[m_id] = json.load(f)
        if e_path.exists():
            with open(e_path, "r", encoding="utf-8") as f:
                model_evidence[m_id] = json.load(f)

    # 1. Market calendar and trailing 63 market dates for liquidity auditing
    df_p_raw = pd.read_parquet(prices_path)
    df_p_raw["Date_str"] = df_p_raw["Date"].astype(str).str[:10]
    market_calendar_dates = sorted(df_p_raw.groupby("Date_str")["series_id"].count()[lambda c: c >= 8].index)
    if target_as_of in market_calendar_dates:
        curr_mkt_idx = market_calendar_dates.index(target_as_of)
        last_63_mkt_dates = set(market_calendar_dates[max(0, curr_mkt_idx - 62): curr_mkt_idx + 1])
    else:
        last_63_mkt_dates = set(market_calendar_dates[-63:])
        
    stale_max_days = int(config.get("stale_max_market_days", 10))
    min_feature_rows = int(config.get("min_valid_feature_rows_for_scoring", 750))
    df_u["Date_str"] = df_u["Date"].astype(str).str[:10]
    
    # Classify stock status across the universe
    stock_status_map = {}
    cross_section_rows = []
    
    for sid in universe_series:
        sub_all = df_u[(df_u["series_id"] == sid) & (df_u["Date_str"] <= target_as_of)].sort_values("Date")
        if sub_all.empty or len(sub_all) < min_feature_rows or not sub_all["rankable"].iloc[-1]:
            stock_status_map[sid] = {
                "status": "not_enough_history",
                "last_traded_date": str(sub_all["Date_str"].iloc[-1]) if not sub_all.empty else "N/A",
                "row": None,
                "rankable": False
            }
        else:
            last_row = sub_all.iloc[-1]
            last_d = str(last_row["Date_str"])
            if last_d == target_as_of:
                stock_status_map[sid] = {
                    "status": "ranked",
                    "last_traded_date": target_as_of,
                    "row": last_row,
                    "rankable": True
                }
                cross_section_rows.append(last_row)
            else:
                mkt_gap = len([d for d in market_calendar_dates if last_d < d <= target_as_of])
                if mkt_gap <= stale_max_days:
                    stock_status_map[sid] = {
                        "status": "stale",
                        "last_traded_date": last_d,
                        "row": last_row,
                        "rankable": True
                    }
                    cross_section_rows.append(last_row)
                else:
                    stock_status_map[sid] = {
                        "status": "not_recently_traded",
                        "last_traded_date": last_d,
                        "row": None,
                        "rankable": False
                    }

    if cross_section_rows:
        cross_section_df = pd.DataFrame(cross_section_rows).set_index("series_id")
    else:
        cross_section_df = pd.DataFrame()
        
    n_rankable = len(cross_section_df)
    has_enough_data = (n_rankable >= 8)

    # 21-day previous date cross section for rank changes
    prev_cross_rows = []
    for sid in universe_series:
        sub_p = df_u[(df_u["series_id"] == sid) & (df_u["Date_str"] <= prev_21_date)].sort_values("Date")
        if not sub_p.empty and len(sub_p) >= min_feature_rows and sub_p["rankable"].iloc[-1]:
            prev_cross_rows.append(sub_p.iloc[-1])
    prev_cross_df = pd.DataFrame(prev_cross_rows).set_index("series_id") if prev_cross_rows else pd.DataFrame()

    tier_cutoffs = {
        "short": float(thresholds["tier_cutoff_short"]["value"]),
        "mid": float(thresholds["tier_cutoff_mid"]["value"]),
        "long": float(thresholds["tier_cutoff_long"]["value"])
    }

    # Scoring for every servable model
    for m_id in servable_models:
        score_rows = []
        
        def predict_model_raw(sub_df, horizon):
            if sub_df.empty:
                return pd.Series(dtype=float)
            if m_id == "momentum_126":
                return sub_df["log_ret_126"].copy()
            elif m_id == "ridge_core":
                prep = loaded_models["ridge_core"][horizon]["prep"]
                model = loaded_models["ridge_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="neutral")
                trans = apply_cross_sectional_rank(trans, prep["cand_cols"])
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            elif m_id == "rf_reg_core":
                prep = loaded_models["rf_reg_core"][horizon]["prep"]
                model = loaded_models["rf_reg_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="median")
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            elif m_id == "lgbm_reg_core":
                prep = loaded_models["lgbm_reg_core"][horizon]["prep"]
                model = loaded_models["lgbm_reg_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="none")
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            elif m_id == "ens_rank4":
                r_ridge = predict_model_raw_named("ridge_core", sub_df, horizon)
                r_rf = predict_model_raw_named("rf_reg_core", sub_df, horizon)
                r_lgbm = predict_model_raw_named("lgbm_reg_core", sub_df, horizon)
                r_mom = sub_df["log_ret_126"].copy()
                
                rnk_ridge = r_ridge.rank(pct=True, method="average")
                rnk_rf = r_rf.rank(pct=True, method="average")
                rnk_lgbm = r_lgbm.rank(pct=True, method="average")
                rnk_mom = r_mom.rank(pct=True, method="average")
                return (rnk_ridge + rnk_rf + rnk_lgbm + rnk_mom) / 4.0
            return pd.Series(dtype=float)

        def predict_model_raw_named(name, sub_df, horizon):
            if name == "ridge_core":
                prep = loaded_models["ridge_core"][horizon]["prep"]
                model = loaded_models["ridge_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="neutral")
                trans = apply_cross_sectional_rank(trans, prep["cand_cols"])
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            elif name == "rf_reg_core":
                prep = loaded_models["rf_reg_core"][horizon]["prep"]
                model = loaded_models["rf_reg_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="median")
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            elif name == "lgbm_reg_core":
                prep = loaded_models["lgbm_reg_core"][horizon]["prep"]
                model = loaded_models["lgbm_reg_core"][horizon]["model"]
                trans, _ = transform_preprocessor(sub_df, prep["preprocessor_state"], prep["feature_cols"], impute_mode="none")
                return pd.Series(model.predict(trans[prep["active_cols"]].values), index=sub_df.index)
            return pd.Series(dtype=float)

        # Compute raw scores and percentiles for cross section with average rank for ties
        raw_cur = {}
        pct_cur = {}
        tier_cur = {}
        pct_prev = {}
        
        def compute_percentiles(raw_s: pd.Series) -> pd.Series:
            if raw_s.empty:
                return pd.Series(dtype=float)
            N = len(raw_s)
            rk = raw_s.rank(method="average", ascending=True)
            return ((rk - 0.5) / N) * 100.0

        for h in horizons:
            if has_enough_data:
                s_c = predict_model_raw(cross_section_df, h)
                raw_cur[h] = s_c
                pct_cur[h] = compute_percentiles(s_c)
                tier_cur[h] = assign_tiers(pct_cur[h], tier_cutoffs[h])
            else:
                raw_cur[h] = pd.Series(dtype=float)
                pct_cur[h] = pd.Series(dtype=float)
                tier_cur[h] = pd.Series(dtype=object)
                
            if len(prev_cross_df) >= 8:
                s_p = predict_model_raw(prev_cross_df, h)
                pct_prev[h] = compute_percentiles(s_p)
            else:
                pct_prev[h] = pd.Series(dtype=float)

        # Conditional volatility note: median vol_63d of ranked stocks exceeds 66.7th percentile
        med_vol_ann = float(cross_section_df["vol_63d"].median()) * 100.0 if not cross_section_df.empty else 0.0
        vol_cutoff = float(thresholds["risk_category_high_cutoff"]["value"])
        market_vol_elevated = bool(med_vol_ann > vol_cutoff)
        market_volatility_note = thresholds.get("market_volatility_note_rule", {}).get("user_note", "Market-wide volatility is elevated.") if market_vol_elevated else None

        # Build stock rows for this model
        for sid in universe_series:
            status_info = stock_status_map[sid]
            st = status_info["status"]
            last_traded = status_info["last_traded_date"]
            
            row_dict = {
                "as_of": target_as_of,
                "data_version": data_version,
                "data_hash": data_hash,
                "series_id": sid,
                "model_id": m_id,
                "status": st,
                "last_traded_date": last_traded
            }
            
            if not status_info["rankable"] or not has_enough_data:
                row_dict["rankable"] = False
                row_dict["overall_upside_score"] = np.nan
                row_dict["consistent_leader"] = False
                for h in horizons:
                    row_dict[f"score_{h}"] = np.nan
                    row_dict[f"tier_{h}"] = "N/A"
                    row_dict[f"confidence_{h}"] = "Not evaluable"
                    row_dict[f"rank_change_21d_{h}"] = np.nan
                    row_dict[f"reasons_{h}"] = {"top_positive": [], "top_negative": []}
                row_dict["risk_category"] = "N/A"
                row_dict["liquidity_warning"] = False
                row_dict["thin_trading_flag"] = (sid in stretched_series)
                row_dict["reasons"] = {"top_positive": [], "top_negative": []}
                row_dict["market_volatility_note"] = market_volatility_note
                score_rows.append(row_dict)
                continue
                
            stock_data = status_info["row"]
            row_dict["rankable"] = True
            
            for h in horizons:
                s_val = float(pct_cur[h].loc[sid])
                t_val = str(tier_cur[h].loc[sid])
                row_dict[f"score_{h}"] = round(s_val, 2)
                row_dict[f"tier_{h}"] = t_val
                
                # Confidence label strictly from data/app/models/{m_id}/evidence.json
                conf = model_evidence.get(m_id, {}).get(h, {}).get("confidence_label", "Moderate confidence")
                row_dict[f"confidence_{h}"] = conf
                
                if sid in pct_prev[h].index:
                    prev_s_val = float(pct_prev[h].loc[sid])
                    row_dict[f"rank_change_21d_{h}"] = round(s_val - prev_s_val, 2)
                else:
                    row_dict[f"rank_change_21d_{h}"] = np.nan
                    
            # Overall score: weighted composite
            ov_score = (
                row_dict["score_short"] * active_weights["short"] +
                row_dict["score_mid"] * active_weights["mid"] +
                row_dict["score_long"] * active_weights["long"]
            )
            row_dict["overall_upside_score"] = round(ov_score, 2)
            row_dict["rank_change_explanation"] = "indicative: computed with the current model, not an out-of-sample result"
            
            # Risk category from annualized vol_63d (vol_63d is already annualized, multiply by 100)
            v_ann = float(stock_data["vol_63d"]) * 100.0 if pd.notnull(stock_data.get("vol_63d")) else np.nan
            if pd.isna(v_ann):
                rc = "Medium"
            elif v_ann <= vol_low:
                rc = "Low"
            elif v_ann > vol_high:
                rc = "High"
            else:
                rc = "Medium"
            row_dict["risk_category"] = rc
            
            # Liquidity warning: trailing 63 market calendar days traded share
            sub_p_stock = df_p_raw[df_p_raw["series_id"] == sid]
            s_traded_dates = set(sub_p_stock["Date_str"])
            tds_count = len(s_traded_dates.intersection(last_63_mkt_dates))
            tds = tds_count / float(len(last_63_mkt_dates)) if last_63_mkt_dates else 1.0
            row_dict["trade_day_share_63m"] = round(tds, 4)
            liq_warn = bool(tds <= liq_cutoff)
            row_dict["liquidity_warning"] = liq_warn
            row_dict["thin_trading_flag"] = bool(sid in stretched_series)
            
            # Consistent Leader: High in >= 2 horizons, Low in none, no liquidity warning
            cl = is_consistent_leader(row_dict["tier_short"], row_dict["tier_mid"], row_dict["tier_long"], liq_warn)
            row_dict["consistent_leader"] = cl
            row_dict["market_volatility_note"] = market_volatility_note
            
            # Stock-specific reasons per horizon
            r_s = reasons(m_id, "short", stock_data, cross_section_df=cross_section_df, tier=row_dict["tier_short"])
            r_m = reasons(m_id, "mid", stock_data, cross_section_df=cross_section_df, tier=row_dict["tier_mid"])
            r_l = reasons(m_id, "long", stock_data, cross_section_df=cross_section_df, tier=row_dict["tier_long"])
            row_dict["reasons_short"] = r_s
            row_dict["reasons_mid"] = r_m
            row_dict["reasons_long"] = r_l
            row_dict["reasons"] = {
                "short": r_s,
                "mid": r_m,
                "long": r_l,
                "top_positive": r_s["top_positive"],
                "top_negative": r_s["top_negative"]
            }
            if any("thin trading (higher price impact)" in r_d.get("top_positive", []) for r_d in [r_s, r_m, r_l]):
                row_dict["thin_trading_note"] = "This stock trades less actively than most, so the signal may be harder to act on."
            
            score_rows.append(row_dict)
            
        df_model_scores = pd.DataFrame(score_rows)
        out_pq = out_dir / f"scores_{m_id}.parquet"
        df_model_scores.to_parquet(out_pq, index=False)
        print(f"Saved {out_pq} ({len(df_model_scores)} rows)")

    # 1b. Build and save scores_served.parquet resolving active model per horizon
    active_def_path = SERVING_DIR / "active_model_defaults.json"
    if active_def_path.exists():
        with open(active_def_path, "r", encoding="utf-8") as f:
            active_defaults = json.load(f)
        active_models = active_defaults.get("active_models", {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"})
    else:
        active_models = {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"}

    df_short_m = pd.read_parquet(out_dir / f"scores_{active_models['short']}.parquet").set_index("series_id")
    df_mid_m = pd.read_parquet(out_dir / f"scores_{active_models['mid']}.parquet").set_index("series_id")
    df_long_m = pd.read_parquet(out_dir / f"scores_{active_models['long']}.parquet").set_index("series_id")

    served_rows = []
    for sid in universe_series:
        s_r = df_short_m.loc[sid]
        m_r = df_mid_m.loc[sid]
        l_r = df_long_m.loc[sid]
        
        row_s = {
            "as_of": target_as_of,
            "data_version": data_version,
            "data_hash": data_hash,
            "series_id": sid,
            "status": s_r["status"],
            "last_traded_date": s_r["last_traded_date"],
            "rankable": bool(s_r["rankable"]),
            "score_short": s_r["score_short"],
            "tier_short": s_r["tier_short"],
            "confidence_short": model_evidence.get(active_models["short"], {}).get("short", {}).get("confidence_label", "Very low confidence"),
            "rank_change_21d_short": s_r["rank_change_21d_short"],
            "score_mid": m_r["score_mid"],
            "tier_mid": m_r["tier_mid"],
            "confidence_mid": model_evidence.get(active_models["mid"], {}).get("mid", {}).get("confidence_label", "Low confidence"),
            "rank_change_21d_mid": m_r["rank_change_21d_mid"],
            "score_long": l_r["score_long"],
            "tier_long": l_r["tier_long"],
            "confidence_long": model_evidence.get(active_models["long"], {}).get("long", {}).get("confidence_label", "Very low confidence"),
            "rank_change_21d_long": l_r["rank_change_21d_long"],
            "risk_category": s_r["risk_category"],
            "trade_day_share_63m": s_r["trade_day_share_63m"],
            "liquidity_warning": bool(s_r["liquidity_warning"]),
            "thin_trading_flag": bool(s_r["thin_trading_flag"]),
            "rank_change_explanation": "indicative: computed with the current model, not an out-of-sample result",
            "market_volatility_note": market_volatility_note,
            "reasons_short": s_r["reasons_short"],
            "reasons_mid": m_r["reasons_mid"],
            "reasons_long": l_r["reasons_long"],
            "reasons": {
                "short": s_r["reasons_short"],
                "mid": m_r["reasons_mid"],
                "long": l_r["reasons_long"],
                "top_positive": s_r["reasons_short"]["top_positive"],
                "top_negative": s_r["reasons_short"]["top_negative"]
            }
        }
        
        # Overall score & Consistent Leader for served configuration
        if row_s["rankable"]:
            ov_score = (
                row_s["score_short"] * active_weights["short"] +
                row_s["score_mid"] * active_weights["mid"] +
                row_s["score_long"] * active_weights["long"]
            )
            row_s["overall_upside_score"] = round(ov_score, 2)
            row_s["consistent_leader"] = is_consistent_leader(
                row_s["tier_short"], row_s["tier_mid"], row_s["tier_long"], row_s["liquidity_warning"]
            )
        else:
            row_s["overall_upside_score"] = np.nan
            row_s["consistent_leader"] = False
            
        if any("thin trading (higher price impact)" in r_d.get("top_positive", []) for r_d in [s_r["reasons_short"], m_r["reasons_mid"], l_r["reasons_long"]]):
            row_s["thin_trading_note"] = "This stock trades less actively than most, so the signal may be harder to act on."
            
        served_rows.append(row_s)
        
    df_served = pd.DataFrame(served_rows)
    out_served_pq = out_dir / "scores_served.parquet"
    df_served.to_parquet(out_served_pq, index=False)
    print(f"Saved {out_served_pq} ({len(df_served)} rows)")
        
    # 2. Output stocks.json
    stocks_meta = []
    prices_raw = pd.read_parquet(prices_path)
    for sid in universe_series:
        sub_p = prices_raw[prices_raw["series_id"] == sid].sort_values("Date")
        sub_f = df_u[(df_u["series_id"] == sid) & (df_u["Date_str"] <= target_as_of)]
        
        last_price = float(sub_p["Close"].iloc[-1]) if not sub_p.empty else np.nan
        prev_price = float(sub_p["Close"].iloc[-2]) if len(sub_p) > 1 else last_price
        d_chg = round((last_price - prev_price) / prev_price * 100.0, 2) if (prev_price and not np.isnan(prev_price)) else 0.0
        
        # 52-week range (252 trading days)
        p_52w = sub_p["Close"].iloc[-252:] if len(sub_p) >= 252 else sub_p["Close"]
        w52_low = float(p_52w.min()) if not p_52w.empty else last_price
        w52_high = float(p_52w.max()) if not p_52w.empty else last_price
        
        comp_name = sub_p["ShortName"].iloc[-1] if ("ShortName" in sub_p.columns and not sub_p.empty) else sid.split(".")[0]
        sub_type = sub_p["SubType"].iloc[-1] if ("SubType" in sub_p.columns and not sub_p.empty) else "Ordinary"
        
        stocks_meta.append({
            "code": sid,
            "company_name": comp_name,
            "last_price": last_price,
            "daily_change_pct": d_chg,
            "range_52w": [round(w52_low, 2), round(w52_high, 2)],
            "sub_type": sub_type,
            "as_of": target_as_of,
            "data_version": data_version,
            "data_hash": data_hash
        })
        
        # Output prices/{code}.parquet with technical chart indicators
        if not sub_p.empty:
            p_df = sub_p[["Date", "Open", "High", "Low", "Close", "TradeVolume"]].copy().sort_values("Date").reset_index(drop=True)
            p_df["sma_20"] = p_df["Close"].rolling(20, min_periods=20).mean()
            p_df["sma_50"] = p_df["Close"].rolling(50, min_periods=50).mean()
            p_df["sma_200"] = p_df["Close"].rolling(200, min_periods=200).mean()
            p_df["rsi_14"] = compute_rsi(p_df["Close"], window=14)
            m_line, s_line, h_line = compute_macd(p_df["Close"])
            p_df["macd_line"] = m_line
            p_df["macd_signal"] = s_line
            p_df["macd_hist"] = h_line
            bb_u, bb_m, bb_l = compute_bollinger_bands(p_df["Close"])
            p_df["bollinger_upper"] = bb_u
            p_df["bollinger_middle"] = bb_m
            p_df["bollinger_lower"] = bb_l
            p_df["as_of"] = target_as_of
            p_df["data_version"] = data_version
            p_df["data_hash"] = data_hash
            p_df.to_parquet(prices_out_dir / f"{sid}.parquet", index=False)
            
    with open(out_dir / "stocks.json", "w", encoding="utf-8") as f:
        json.dump(stocks_meta, f, indent=2)
    print(f"Saved {out_dir / 'stocks.json'}")
    
    # 3. Export active_model_defaults.json (from served_rule.json)
    with open(base_dir / "data/app/served_rule.json", "r", encoding="utf-8") as f:
        served_rules = json.load(f)
    active_defaults = {
        "as_of": target_as_of,
        "data_version": data_version,
        "data_hash": data_hash,
        "active_models": {h: served_rules[h]["chosen_model"] for h in horizons},
        "horizon_weights": active_weights,
        "rules": served_rules
    }
    with open(SERVING_DIR / "active_model_defaults.json", "w", encoding="utf-8") as f:
        json.dump(active_defaults, f, indent=2)
    print(f"Saved {SERVING_DIR / 'active_model_defaults.json'}")
    
    # 4. Update data/serving/CURRENT pointer
    current_pointer = SERVING_DIR / "CURRENT"
    current_pointer.write_text(target_as_of, encoding="utf-8")
    print(f"Updated {current_pointer} -> {target_as_of}")
    
    return {
        "as_of": target_as_of,
        "data_version": data_version,
        "data_hash": data_hash,
        "serving_dir": str(out_dir),
        "rankable_stocks": n_rankable
    }


if __name__ == "__main__":
    score_as_of()

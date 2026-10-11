"""Explainability module for Stock Pulse models providing TreeSHAP interpretations."""
from typing import Any, Union
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS: dict[str, str] = {
    "log_ret_5": "5-Day Return",
    "log_ret_10": "10-Day Return",
    "log_ret_21": "21-Day Return (1-Month Momentum)",
    "log_ret_63": "63-Day Return (3-Month Momentum)",
    "log_ret_126": "126-Day Return (6-Month Momentum)",
    "log_ret_252": "252-Day Return (12-Month Momentum)",
    "vol_21d": "21-Day Realized Volatility",
    "vol_63d": "63-Day Realized Volatility",
    "downside_vol_63d": "63-Day Downside Volatility",
    "parkinson_vol_21d": "21-Day Parkinson High-Low Volatility",
    "atr14_norm": "14-Day Average True Range (Normalized)",
    "max_drawdown_63d": "63-Day Max Drawdown",
    "dist_high_126d": "Distance to 126-Day High",
    "dist_low_126d": "Distance to 126-Day Low",
    "close_to_sma20": "Distance to 20-Day Moving Average",
    "close_to_sma50": "Distance to 50-Day Moving Average",
    "close_to_sma200": "Distance to 200-Day Moving Average",
    "sma50_to_sma200": "50-to-200 Day Moving Average Ratio",
    "rsi_14": "14-Day Relative Strength Index",
    "macd_hist_norm": "MACD Histogram (Normalized)",
    "bollinger_pct_b_20": "Bollinger %B (20-Day)",
    "log_trade_vol_21d": "21-Day Log Trading Volume",
    "vol_ratio_21_126": "Volume Ratio (21d / 126d)",
    "log_turnover_21d": "21-Day Log Turnover Ratio",
    "amihud_illiq_63d": "63-Day Amihud Illiquidity Ratio",
    "zero_trade_share_63d": "63-Day Zero-Trade Share",
    "rel_ret_21d": "21-Day Relative Return vs S&P SL20",
    "rel_ret_63d": "63-Day Relative Return vs S&P SL20",
    "index_ret_63d": "63-Day S&P SL20 Index Return",
    "index_vol_63d": "63-Day S&P SL20 Index Volatility",
    "cs_rank_mom_63d": "Cross-Sectional Momentum Rank",
    "cs_rank_vol_63d": "Cross-Sectional 63d Volatility Rank",
    "cs_rank_vol_21d": "Cross-Sectional 21d Volatility Rank",
    "mom_12_1": "12-1 Month Momentum",
    "max_drawdown_252d": "252-Day Max Drawdown",
    "dist_high_252d": "Distance to 252-Day High",
    "dist_low_252d": "Distance to 252-Day Low",
    "beta_252d": "252-Day Market Beta",
    "vol_21d_ownz": "21-Day Volatility (Own-History Z-Score)",
    "vol_63d_ownz": "63-Day Volatility (Own-History Z-Score)",
    "log_trade_vol_21d_ownz": "21-Day Volume (Own-History Z-Score)",
    "log_turnover_21d_ownz": "21-Day Turnover (Own-History Z-Score)",
    "amihud_illiq_63d_ownz": "63-Day Amihud Illiquidity (Own-History Z-Score)",
}

FEATURE_FAMILIES: dict[str, str] = {
    "log_ret_5": "Momentum & Reversal",
    "log_ret_10": "Momentum & Reversal",
    "log_ret_21": "Momentum & Reversal",
    "log_ret_63": "Momentum & Reversal",
    "log_ret_126": "Momentum & Reversal",
    "log_ret_252": "Momentum & Reversal",
    "mom_12_1": "Momentum & Reversal",
    "cs_rank_mom_63d": "Momentum & Reversal",
    "rel_ret_21d": "Momentum & Reversal",
    "rel_ret_63d": "Momentum & Reversal",
    "vol_21d": "Volatility & Risk",
    "vol_63d": "Volatility & Risk",
    "downside_vol_63d": "Volatility & Risk",
    "parkinson_vol_21d": "Volatility & Risk",
    "atr14_norm": "Volatility & Risk",
    "max_drawdown_63d": "Volatility & Risk",
    "max_drawdown_252d": "Volatility & Risk",
    "cs_rank_vol_21d": "Volatility & Risk",
    "cs_rank_vol_63d": "Volatility & Risk",
    "vol_ratio_21_126": "Volatility & Risk",
    "vol_21d_ownz": "Volatility & Risk",
    "vol_63d_ownz": "Volatility & Risk",
    "log_trade_vol_21d": "Liquidity & Turnover",
    "log_turnover_21d": "Liquidity & Turnover",
    "amihud_illiq_63d": "Liquidity & Turnover",
    "zero_trade_share_63d": "Liquidity & Turnover",
    "log_trade_vol_21d_ownz": "Liquidity & Turnover",
    "log_turnover_21d_ownz": "Liquidity & Turnover",
    "amihud_illiq_63d_ownz": "Liquidity & Turnover",
    "close_to_sma20": "Trend & Technical",
    "close_to_sma50": "Trend & Technical",
    "close_to_sma200": "Trend & Technical",
    "sma50_to_sma200": "Trend & Technical",
    "rsi_14": "Trend & Technical",
    "macd_hist_norm": "Trend & Technical",
    "bollinger_pct_b_20": "Trend & Technical",
    "dist_high_126d": "Trend & Technical",
    "dist_low_126d": "Trend & Technical",
    "dist_high_252d": "Trend & Technical",
    "dist_low_252d": "Trend & Technical",
    "index_ret_63d": "Market & Benchmark",
    "index_vol_63d": "Market & Benchmark",
    "beta_252d": "Market & Benchmark",
}

# Features identified with high stock identity (between-stock R2 > 0.40)
STOCK_IDENTITY_PROXIES: set[str] = {
    "zero_trade_share_63d",
    "cs_rank_vol_21d",
    "log_turnover_21d",
    "cs_rank_vol_63d"
}


def explain_prediction(
    model: Any,
    feature_row: Union[pd.Series, pd.DataFrame, dict, np.ndarray],
    top_n: int = 5,
    feature_names: list[str] | None = None
) -> dict[str, Any]:
    """Return top positive and negative contributing features for a single prediction row.
    
    Args:
        model: Trained model (LightGBM Booster, Estimator, or RandomForest)
        feature_row: Single row of features
        top_n: Number of top positive and negative features to extract
        feature_names: Feature names if feature_row is raw numpy array
        
    Returns:
        dict with keys:
            - prediction: Predicted score or probability
            - base_value: Baseline expected value / prior
            - top_positive: list of dicts for top positive contributors
            - top_negative: list of dicts for top negative contributors
            - all_contributions: dict mapping feature name to contribution
    """
    # Extract feature names and 2D array
    if isinstance(feature_row, pd.Series):
        f_names = list(feature_row.index)
        x_vals = feature_row.values.reshape(1, -1)
    elif isinstance(feature_row, pd.DataFrame):
        f_names = list(feature_row.columns)
        x_vals = feature_row.values[0:1]
    elif isinstance(feature_row, dict):
        f_names = list(feature_row.keys())
        x_vals = np.array([list(feature_row.values())], dtype=float)
    else:
        x_vals = np.asarray(feature_row)
        if x_vals.ndim == 1:
            x_vals = x_vals.reshape(1, -1)
        f_names = feature_names or [f"f_{i}" for i in range(x_vals.shape[1])]
        
    # Unwrap estimator if needed
    raw_model = model
    if hasattr(model, "model") and model.model is not None:
        raw_model = model.model
    if hasattr(model, "booster_") and model.booster_ is not None:
        raw_model = model.booster_
        
    p_contrib = None
    base_val = 0.5
    pred_val = 0.5
    
    # 1. Native LightGBM booster fast TreeSHAP
    if hasattr(raw_model, "predict") and hasattr(raw_model, "dump_model"):
        # raw LightGBM booster
        contribs = raw_model.predict(x_vals, pred_contrib=True)
        if contribs.ndim == 2:
            p_contrib = contribs[0, :-1]
            base_val = float(contribs[0, -1])
            pred_val = float(raw_model.predict(x_vals)[0])
    elif hasattr(raw_model, "predict_proba"):
        # RandomForest or classifier
        explainer = shap.TreeExplainer(raw_model)
        sv = explainer.shap_values(x_vals)
        if isinstance(sv, list) and len(sv) > 1:
            p_contrib = sv[1][0]
            base_val = float(explainer.expected_value[1]) if hasattr(explainer, "expected_value") else 0.5
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            p_contrib = sv[0, :, 1]
            base_val = float(explainer.expected_value[1]) if hasattr(explainer, "expected_value") else 0.5
        else:
            p_contrib = sv[0] if sv.ndim == 2 else sv
            base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.5
        probs = raw_model.predict_proba(x_vals)
        pred_val = float(probs[0, 1]) if probs.ndim == 2 and probs.shape[1] > 1 else float(probs[0])
    else:
        # Generic TreeExplainer
        explainer = shap.TreeExplainer(raw_model)
        sv = explainer.shap_values(x_vals)
        p_contrib = sv[0] if getattr(sv, "ndim", 1) == 2 else sv
        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.5
        pred_val = float(raw_model.predict(x_vals)[0])
        
    contrib_dict = {}
    items = []
    for i, name in enumerate(f_names):
        c_val = float(p_contrib[i]) if p_contrib is not None and i < len(p_contrib) else 0.0
        val = float(x_vals[0, i]) if not pd.isna(x_vals[0, i]) else np.nan
        contrib_dict[name] = c_val
        items.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": val,
            "contribution": c_val,
            "family": FEATURE_FAMILIES.get(name, "Other"),
            "is_identity_proxy": name in STOCK_IDENTITY_PROXIES
        })
        
    pos_items = [it for it in items if it["contribution"] > 0]
    neg_items = [it for it in items if it["contribution"] < 0]
    
    pos_items.sort(key=lambda x: x["contribution"], reverse=True)
    neg_items.sort(key=lambda x: x["contribution"])  # most negative first
    
    return {
        "prediction": pred_val,
        "base_value": base_val,
        "top_positive": pos_items[:top_n],
        "top_negative": neg_items[:top_n],
        "all_contributions": contrib_dict
    }

"""Preprocessing and feature transformation helpers fitted strictly on training folds."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

REDUNDANT_LINEAR_FEATURES = ["cs_rank_mom_63d", "cs_rank_vol_63d", "cs_rank_vol_21d"]

@dataclass
class PreprocessorState:
    """Stores training-fold learned parameters to guarantee zero test leakage."""
    medians: dict[str, float]
    target_p01: float | None
    target_p99: float | None


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None
) -> PreprocessorState:
    """Fit imputation medians and target winsorization percentiles on train fold only.
    No missing indicators are created.
    """
    medians = {}
    for col in feature_cols:
        s = train_df[col]
        med = float(s.median()) if s.notna().any() else 0.0
        medians[col] = med
            
    p01, p99 = None, None
    if target_col is not None and target_col in train_df.columns:
        valid_targets = train_df[target_col].dropna()
        if not valid_targets.empty:
            p01 = float(np.percentile(valid_targets, 1.0))
            p99 = float(np.percentile(valid_targets, 99.0))
            
    return PreprocessorState(
        medians=medians,
        target_p01=p01,
        target_p99=p99
    )


def transform_preprocessor(
    df: pd.DataFrame,
    state: PreprocessorState,
    feature_cols: list[str],
    target_col: str | None = None,
    impute_mode: str = "median"
) -> tuple[pd.DataFrame, list[str]]:
    """Transform dataset using fitted train medians. No missing indicators generated.
    Returns transformed DataFrame and feature list.
    """
    df_out = df.copy()
    all_feature_cols = list(feature_cols)
    
    # Impute missing feature values
    for col in feature_cols:
        if impute_mode == "median":
            fill_val = state.medians.get(col, 0.0)
            df_out[col] = df_out[col].fillna(fill_val)
        elif impute_mode == "neutral":
            df_out[col] = df_out[col].fillna(0.5)
        
    # Winsorize target if applicable
    if target_col is not None and target_col in df_out.columns:
        if state.target_p01 is not None and state.target_p99 is not None:
            df_out[target_col] = np.clip(df_out[target_col], state.target_p01, state.target_p99)
            
    return df_out, all_feature_cols


def apply_cross_sectional_rank(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Replace each continuous feature with its daily cross-sectional percentile rank [0, 1].
    Missing values (e.g. 2016 turnover) are imputed with neutral rank 0.5.
    """
    df_out = df.copy()
    for col in feature_cols:
        df_out[col] = df_out.groupby("Date")[col].rank(pct=True)
        # Neutral rank 0.5 for dates where feature is missing or zero variance
        df_out[col] = df_out[col].fillna(0.5)
    return df_out


def filter_linear_features(
    train_trans_df: pd.DataFrame,
    feature_cols: list[str],
    max_corr: float = 0.98
) -> tuple[list[str], list[dict[str, Any]]]:
    """Drop redundant cross-sectional rank inputs and check for collinearity > max_corr in training fold.
    
    1. Removes cs_rank_mom_63d, cs_rank_vol_63d, cs_rank_vol_21d upfront.
    2. Drops constant columns (std == 0) and any second column of a pair with |r| > max_corr.
    Returns cleaned feature list and drop log.
    """
    cols = [c for c in feature_cols if c not in REDUNDANT_LINEAR_FEATURES]
    drop_log = []
    
    # Check zero-variance / constant columns
    clean_cols = []
    for c in cols:
        s = train_trans_df[c]
        if s.std() < 1e-9 or s.isna().all():
            drop_log.append({
                "dropped_col": c,
                "reason": "zero_variance_or_constant",
                "paired_col": None,
                "corr": 0.0
            })
        else:
            clean_cols.append(c)
            
    # Check correlation matrix
    if len(clean_cols) > 1:
        corr_matrix = train_trans_df[clean_cols].corr().abs()
        to_drop = set()
        for i in range(len(clean_cols)):
            c1 = clean_cols[i]
            if c1 in to_drop:
                continue
            for j in range(i + 1, len(clean_cols)):
                c2 = clean_cols[j]
                if c2 in to_drop:
                    continue
                r_val = corr_matrix.loc[c1, c2]
                if pd.isna(r_val) or r_val > max_corr:
                    to_drop.add(c2)
                    drop_log.append({
                        "dropped_col": c2,
                        "reason": "collinear",
                        "paired_col": c1,
                        "corr": float(r_val) if pd.notnull(r_val) else 1.0
                    })
        final_cols = [c for c in clean_cols if c not in to_drop]
    else:
        final_cols = clean_cols
        
    return final_cols, drop_log

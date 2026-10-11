"""Pytest suite validating feature correctness, temporal causality, and leakage prevention."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config
from src.features.build_features import compute_technical_features

@pytest.fixture(scope="module")
def config():
    return load_config()

@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def features_df(config, base_dir):
    f_path = base_dir / config["paths"]["features_targets"]
    if not f_path.exists():
        pytest.fail(f"Features and targets file not found at {f_path}.")
    return pd.read_parquet(f_path)


def test_recomputed_features_match_manual(features_df):
    """Recompute 3 features at 5 sample rows by hand and verify exact numerical match."""
    # Find rows with valid 63d features
    valid_rows = features_df[
        features_df["log_ret_21"].notna() &
        features_df["close_to_sma20"].notna() &
        features_df["max_drawdown_63d"].notna()
    ]
    assert len(valid_rows) >= 5, "Not enough valid rows to test features."
    
    # Pick 5 deterministic spaced rows
    step = len(valid_rows) // 6
    sample_indices = [valid_rows.index[i * step] for i in range(1, 6)]
    
    for idx in sample_indices:
        row = features_df.loc[idx]
        sid = row["series_id"]
        seg = row["segment_id"]
        
        # Get historical slice within the same segment up to idx
        seg_rows = features_df[(features_df["series_id"] == sid) & (features_df["segment_id"] == seg)]
        loc_in_seg = seg_rows.index.get_loc(idx)
        
        # 1. Manual log_ret_21
        c_t = seg_rows["Close"].iloc[loc_in_seg]
        c_t_21 = seg_rows["Close"].iloc[loc_in_seg - 21]
        expected_log_ret_21 = np.log(c_t / c_t_21)
        assert row["log_ret_21"] == pytest.approx(expected_log_ret_21, rel=1e-6, abs=1e-6)
        
        # 2. Manual close_to_sma20
        c_20_slice = seg_rows["Close"].iloc[loc_in_seg - 19 : loc_in_seg + 1]
        expected_sma20 = c_20_slice.mean()
        expected_close_to_sma20 = c_t / expected_sma20 - 1.0
        assert row["close_to_sma20"] == pytest.approx(expected_close_to_sma20, rel=1e-6, abs=1e-6)
        
        # 3. Manual max_drawdown_63d
        c_63_slice = seg_rows["Close"].iloc[loc_in_seg - 62 : loc_in_seg + 1]
        peak_63 = c_63_slice.max()
        expected_mdd_63 = (c_t - peak_63) / peak_63
        assert row["max_drawdown_63d"] == pytest.approx(expected_mdd_63, rel=1e-6, abs=1e-6)


def test_temporal_causality_truncation(config, base_dir, features_df):
    """Prove that no feature changes when future data after date t is removed."""
    clean_path = base_dir / config["paths"]["clean_prices"]
    clean_df = pd.read_parquet(clean_path)
    clean_u = clean_df[clean_df["in_ranking_universe"]].copy()
    
    # Select 3 sample dates from different regimes
    test_dates = [pd.Timestamp("2015-06-15"), pd.Timestamp("2019-11-20"), pd.Timestamp("2024-09-10")]
    
    for t_date in test_dates:
        # Truncate dataset strictly at t_date
        truncated_df = clean_u[clean_u["Date"] <= t_date].copy()
        recomputed_tech = compute_technical_features(truncated_df, config)
        
        # Check matching series on t_date
        t_recomputed = recomputed_tech[recomputed_tech["Date"] == t_date]
        t_original = features_df[features_df["Date"] == t_date]
        
        for _, orig_row in t_original.iterrows():
            sid = orig_row["series_id"]
            rec_match = t_recomputed[t_recomputed["series_id"] == sid]
            if rec_match.empty:
                continue
            rec_row = rec_match.iloc[0]
            
            # Check momentum, risk, and ownz features match exactly
            test_cols = [
                "log_ret_21", "close_to_sma20", "atr14_norm",
                "log_turnover_21d_ownz", "log_trade_vol_21d_ownz",
                "amihud_illiq_63d_ownz", "vol_21d_ownz", "vol_63d_ownz"
            ]
            for col in test_cols:
                orig_val = orig_row[col]
                rec_val = rec_row[col]
                if pd.isna(orig_val):
                    assert pd.isna(rec_val), f"Expected NaN for {sid} {col} at {t_date}"
                else:
                    assert orig_val == pytest.approx(rec_val, rel=1e-6, abs=1e-6), (
                        f"Leakage detected: {col} differs after truncation on {t_date} for {sid}"
                    )


def test_no_feature_crosses_hard_break(features_df):
    """Assert that for any trading row position < W inside its segment, window W features are strictly NaN."""
    # Compute row index inside each segment
    features_df["row_in_seg"] = features_df.groupby(["series_id", "segment_id"]).cumcount()
    
    # For rows with row_in_seg < 21, log_ret_21 and vol_21d MUST be NaN
    early_21 = features_df[features_df["row_in_seg"] < 21]
    assert early_21["log_ret_21"].isna().all(), "log_ret_21 breached segment boundary!"
    assert early_21["vol_21d"].isna().all(), "vol_21d breached segment boundary!"
    
    # For rows with row_in_seg < 63, log_ret_63 and vol_63d MUST be NaN
    early_63 = features_df[features_df["row_in_seg"] < 63]
    assert early_63["log_ret_63"].isna().all(), "log_ret_63 breached segment boundary!"
    assert early_63["vol_63d"].isna().all(), "vol_63d breached segment boundary!"
    
    # For rows with row_in_seg < 63, all _ownz features MUST be NaN (min_periods=63)
    ownz_cols = ["log_turnover_21d_ownz", "log_trade_vol_21d_ownz", "amihud_illiq_63d_ownz", "vol_21d_ownz", "vol_63d_ownz"]
    for col in ownz_cols:
        assert early_63[col].isna().all(), f"{col} breached segment boundary before min_periods 63!"
    
    # For rows with row_in_seg < 252, log_ret_252 MUST be NaN
    early_252 = features_df[features_df["row_in_seg"] < 252]
    assert early_252["log_ret_252"].isna().all(), "log_ret_252 breached segment boundary!"


def test_targets_respect_segment_and_offsets(features_df):
    """Verify that forward targets look strictly within the same segment at exact row offsets."""
    valid_targets = features_df[
        features_df["fwd_ret_short"].notna() &
        features_df["fwd_ret_mid"].notna() &
        features_df["fwd_ret_long"].notna()
    ]
    assert len(valid_targets) >= 5, "Not enough valid targets to test."
    
    sample_indices = valid_targets.index[:5]
    
    for idx in sample_indices:
        row = features_df.loc[idx]
        sid = row["series_id"]
        seg = row["segment_id"]
        
        seg_rows = features_df[(features_df["series_id"] == sid) & (features_df["segment_id"] == seg)]
        loc = seg_rows.index.get_loc(idx)
        
        c_t = seg_rows["Close"].iloc[loc]
        
        # Short (21)
        c_t_21 = seg_rows["Close"].iloc[loc + 21]
        expected_short = (c_t_21 / c_t) - 1.0
        assert row["fwd_ret_short"] == pytest.approx(expected_short, rel=1e-6, abs=1e-6)
        
        # Mid (63)
        c_t_63 = seg_rows["Close"].iloc[loc + 63]
        expected_mid = (c_t_63 / c_t) - 1.0
        assert row["fwd_ret_mid"] == pytest.approx(expected_mid, rel=1e-6, abs=1e-6)
        
        # Long (252)
        c_t_252 = seg_rows["Close"].iloc[loc + 252]
        expected_long = (c_t_252 / c_t) - 1.0
        assert row["fwd_ret_long"] == pytest.approx(expected_long, rel=1e-6, abs=1e-6)

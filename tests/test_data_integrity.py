"""Pytest suite validating data integrity, split adjustments, segmentation, and return calculations."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config

@pytest.fixture(scope="module")
def config():
    return load_config()

@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def clean_df(config, base_dir):
    clean_path = base_dir / config["paths"]["clean_prices"]
    if not clean_path.exists():
        pytest.fail(f"Cleaned dataset not found at {clean_path}. Please run data pipeline first.")
    return pd.read_parquet(clean_path)

@pytest.fixture(scope="module")
def adjustments_df(config, base_dir):
    adj_path = base_dir / config["paths"]["adjustments_log"]
    if not adj_path.exists():
        pytest.fail(f"Adjustments log not found at {adj_path}. Please run data pipeline first.")
    return pd.read_csv(adj_path)


def test_no_duplicate_date_company(clean_df):
    """Test that there are zero duplicate (Date, CompanyCode) pairs."""
    duplicates = clean_df.duplicated(subset=["Date", "CompanyCode"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (Date, CompanyCode) records."


def test_segment_boundaries_respected(clean_df, config):
    """Test that hard break segment boundaries are strictly respected and returns do not bridge hard breaks."""
    hard_break_days = config["cleaning"].get("hard_break_gap_days", 60)
    
    # 1. Any consecutive date gap > hard_break_days (e.g. 278d, 642d) must trigger a new segment_id
    df_sorted = clean_df.sort_values(["series_id", "Date"]).copy()
    df_sorted["prev_series"] = df_sorted["series_id"].shift(1)
    df_sorted["prev_segment"] = df_sorted["segment_id"].shift(1)
    
    large_gaps = df_sorted[(df_sorted["series_id"] == df_sorted["prev_series"]) & (df_sorted["gap_days"] > hard_break_days)]
    assert len(large_gaps) > 0, "Expected to find large gaps (such as the 642-day absence)."
    
    for _, row in large_gaps.iterrows():
        assert row["segment_id"] != row["prev_segment"], (
            f"Segment ID did not increment across hard break of {row['gap_days']} days for {row['series_id']} on {row['Date']}."
        )

    # 2. At every segment boundary start, daily_return and log_return must be NaN
    segment_starts = df_sorted.groupby(["series_id", "segment_id"]).head(1)
    assert segment_starts["daily_return"].isna().all(), "Segment start row must have NaN daily_return."
    assert segment_starts["log_return"].isna().all(), "Segment start row must have NaN log_return."


def test_adjusted_split_rows_logged(adjustments_df):
    """Test that adjusted stock splits are logged with proper metadata."""
    assert not adjustments_df.empty, "Adjustments log must not be empty."
    required_cols = {"series_id", "CompanyCode", "SplitDate", "SplitType", "CanonicalRatio", "RowsAdjusted", "Applied"}
    assert required_cols.issubset(adjustments_df.columns), f"Missing columns in adjustments log: {required_cols - set(adjustments_df.columns)}"
    
    # Check that confirmed corporate split (e.g. JKH on 2024-11-06) was recorded
    jkh_split = adjustments_df[adjustments_df["CompanyCode"] == "JKH"]
    assert len(jkh_split) > 0, "Expected JKH corporate split to be logged."
    assert (jkh_split["RowsAdjusted"] > 0).all(), "Adjusted rows count must be > 0."


def test_recomputed_returns_match_manual_calculation(clean_df):
    """Test that recomputed simple returns and log returns match exact manual calculation on 3 sample rows."""
    df_sorted = clean_df.sort_values(["series_id", "Date"]).reset_index(drop=True)
    
    # Select 3 non-first rows within continuous segments from 3 different voting series
    sample_series = ["COMB.N", "JKH.N", "SAMP.N"]
    sample_indices = []
    
    for sid in sample_series:
        comp_df = df_sorted[(df_sorted["series_id"] == sid) & df_sorted["daily_return"].notna()]
        # Pick the 100th valid return row of each series
        idx = comp_df.index[100]
        sample_indices.append(idx)
        
    for idx in sample_indices:
        cur_row = df_sorted.loc[idx]
        prev_row = df_sorted.loc[idx - 1]
        
        # Verify same series and same segment
        assert cur_row["series_id"] == prev_row["series_id"]
        assert cur_row["segment_id"] == prev_row["segment_id"]
        
        close_t = cur_row["Close"]
        close_prev = prev_row["Close"]
        
        expected_simple_return = (close_t - close_prev) / close_prev
        expected_log_return = np.log(close_t / close_prev)
        
        assert cur_row["daily_return"] == pytest.approx(expected_simple_return, rel=1e-7, abs=1e-7), (
            f"Simple return mismatch at index {idx}: calculated {cur_row['daily_return']}, expected {expected_simple_return}"
        )
        assert cur_row["log_return"] == pytest.approx(expected_log_return, rel=1e-7, abs=1e-7), (
            f"Log return mismatch at index {idx}: calculated {cur_row['log_return']}, expected {expected_log_return}"
        )

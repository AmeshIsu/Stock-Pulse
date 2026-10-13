"""Stock Pulse Data Refresh Pipeline.

CLI: python -m src.pipeline.refresh --input FILE --mode append|replace [--retrain] [--dry-run] [--force]
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

from src.data.config_loader import load_config
from src.data.clean import clean_data
from src.features.build_features import build_features_and_targets
from src.data.create_rankable_mask import build_rankable_mask
from src.production.train import train_production_models
from src.scoring.score_latest import score_as_of

BASE_DIR = Path(__file__).resolve().parents[2]
VERSIONS_DIR = BASE_DIR / "data/versions"
SERVING_DIR = BASE_DIR / "data/serving"
PROCESSED_DIR = BASE_DIR / "data/processed"


def atomic_write_text(file_path: Path, content: str):
    """Atomically write text content to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, file_path)


def validate_input(input_path: Path) -> tuple[pd.DataFrame, dict]:
    """Validate input data file according to Section 3.1."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    file_size = input_path.stat().st_size
    MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
    if file_size > MAX_SIZE_BYTES:
        raise ValueError(f"Input file exceeds maximum allowed size (500MB): {file_size / (1024*1024):.1f} MB")
        
    # Read file
    suffix = input_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(input_path, sheet_name="Full_Dataset_2001_2025")
        except Exception:
            df = pd.read_excel(input_path, sheet_name=0)
    elif suffix in [".csv", ".txt"]:
        df = pd.read_csv(input_path)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Expected Excel, CSV, or Parquet.")

    total_rows = len(df)
    report = {
        "file_path": str(input_path),
        "file_size_bytes": file_size,
        "total_rows": total_rows,
        "valid": True,
        "warnings": [],
        "errors": []
    }

    # 1. Required columns
    required_cols = [
        "Date", "CompanyCode", "MainType",
        "Open", "High", "Low", "Close",
        "ShareVolume", "TradeVolume", "Turnover"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        report["valid"] = False
        report["errors"].append(f"Missing required columns: {missing_cols}")
        return df, report

    # 2. Date parsing and types
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to parse Date column: {str(e)}")
        return df, report

    nat_dates = int(df["Date"].isna().sum())
    report["nat_dates_count"] = nat_dates
    if nat_dates > 0:
        report["valid"] = False
        report["errors"].append(f"Found {nat_dates} NaT or unparseable dates.")

    # Convert numeric fields
    numeric_cols = ["Open", "High", "Low", "Close", "ShareVolume", "TradeVolume", "Turnover"]
    for nc in numeric_cols:
        df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # 3. Duplicate check on (Date, CompanyCode, MainType)
    dups = df.duplicated(subset=["Date", "CompanyCode", "MainType"], keep=False)
    dup_count = int(dups.sum())
    report["duplicate_rows_count"] = dup_count
    if dup_count > 0:
        report["warnings"].append(f"Found {dup_count} duplicate rows for (Date, CompanyCode, MainType). Later rows will overwrite.")

    # 4. Date ordering check per series
    df["MainType"] = df["MainType"].fillna("UNK").astype(str)
    date_order_issues = 0
    for _, grp in df.groupby(["CompanyCode", "MainType"]):
        if not grp["Date"].is_monotonic_increasing:
            date_order_issues += 1
    report["date_order_issues_series"] = date_order_issues
    if date_order_issues > 0:
        report["warnings"].append(f"{date_order_issues} series have non-chronologically ordered dates. Data will be sorted automatically.")

    # 5. Non-positive prices check
    non_pos = (df["Close"] <= 0) | (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0)
    non_pos_count = int(non_pos.sum())
    report["non_positive_price_rows"] = non_pos_count
    if non_pos_count > 0:
        report["valid"] = False
        report["errors"].append(f"Found {non_pos_count} rows with non-positive prices (<= 0).")

    # 6. High below Low check
    high_below_low = df["High"] < df["Low"]
    hbl_count = int(high_below_low.sum())
    report["high_below_low_rows"] = hbl_count
    if hbl_count > 0:
        report["valid"] = False
        report["errors"].append(f"Found {hbl_count} rows where High is strictly below Low.")

    # 7. Row limit check
    if total_rows > 5_000_000:
        report["valid"] = False
        report["errors"].append(f"Dataset exceeds 5,000,000 rows sanity limit: {total_rows}")

    return df, report


def load_current_raw(config: dict) -> pd.DataFrame:
    """Load existing base raw data."""
    raw_rel = config["paths"]["raw_data"]
    raw_path = BASE_DIR / raw_rel
    raw_sheet = config["paths"].get("raw_sheet", "Full_Dataset_2001_2025")
    
    if str(raw_path).endswith((".parquet", ".pq")):
        df = pd.read_parquet(raw_path)
    elif str(raw_path).endswith(".csv"):
        df = pd.read_csv(raw_path)
    else:
        try:
            df = pd.read_excel(raw_path, sheet_name=raw_sheet)
        except Exception:
            df = pd.read_excel(raw_path, sheet_name=0)
            
    df["Date"] = pd.to_datetime(df["Date"])
    df["MainType"] = df["MainType"].fillna("UNK").astype(str)
    return df


def run_refresh(
    input_file: str | Path,
    mode: str = "append",
    retrain: bool = False,
    dry_run: bool = False,
    force: bool = False
) -> dict:
    """Execute complete data refresh workflow."""
    start_time = time.time()
    input_path = Path(input_file).resolve()
    print(f"\n=======================================================")
    print(f"=== STOCK PULSE DATA REFRESH PIPELINE ===")
    print(f"Input:    {input_path}")
    print(f"Mode:     {mode}")
    print(f"Retrain:  {retrain}")
    print(f"Dry Run:  {dry_run}")
    print(f"Force:    {force}")
    print(f"=======================================================\n")

    # 1. Validate input data
    print("Step 1: Validating input dataset...")
    df_new, val_report = validate_input(input_path)
    if not val_report["valid"]:
        raise ValueError(f"Input data failed validation:\n" + "\n".join(val_report["errors"]))
    print(f"Input validated successfully: {val_report['total_rows']:,} rows.")

    config = load_config()

    # 2. Process according to mode
    if mode == "replace":
        print("Mode: REPLACE (Full historical replacement)")
        df_merged = df_new.copy()
        rows_added = len(df_merged)
    elif mode == "append":
        print("Mode: APPEND (Merging with existing dataset)")
        df_curr = load_current_raw(config)
        curr_last_date = df_curr["Date"].max()
        new_first_date = df_new["Date"].min()
        gap_days = (new_first_date - curr_last_date).days
        
        print(f"Current stored dataset last date: {curr_last_date.strftime('%Y-%m-%d')}")
        print(f"Appended dataset first date:       {new_first_date.strftime('%Y-%m-%d')}")
        print(f"Calendar gap between datasets:    {gap_days} days")
        
        # Gap policy: > 60 calendar days creates a hard break
        if gap_days > 60:
            gap_warn = (
                f"WARNING: Appended data starts on {new_first_date.strftime('%Y-%m-%d')}, which is "
                f"{gap_days} calendar days after stored last date {curr_last_date.strftime('%Y-%m-%d')} (> 60 days). "
                f"This creates a hard break and stocks become unrankable for roughly 126 trading days afterwards."
            )
            val_report["warnings"].append(gap_warn)
            print(f"\n!!! {gap_warn} !!!\n")
            if not force:
                raise ValueError(f"{gap_warn}\nOperation aborted. Specify --force to bypass.")
                
        # Merge: later rows win on duplicate (Date, CompanyCode, MainType)
        combined = pd.concat([df_curr, df_new], ignore_index=True)
        df_merged = combined.drop_duplicates(subset=["Date", "CompanyCode", "MainType"], keep="last")
        rows_added = len(df_merged) - len(df_curr)
        print(f"Merged successfully. Rows added: {rows_added:,} (Total rows: {len(df_merged):,})")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'append' or 'replace'.")

    # Sort deterministically
    df_merged = df_merged.sort_values(["CompanyCode", "MainType", "Date"]).reset_index(drop=True)
    as_of_date = df_merged["Date"].max().strftime("%Y-%m-%d")
    print(f"Target as-of date after refresh: {as_of_date}")

    # If dry run, print summary and terminate early
    if dry_run:
        duration = round(time.time() - start_time, 2)
        print("\n--- DRY RUN COMPLETED (No disk state modified) ---")
        print(f"Validation Valid:     {val_report['valid']}")
        print(f"Warnings Logged:      {len(val_report['warnings'])}")
        print(f"Projected As-Of Date: {as_of_date}")
        print(f"Projected Rows Added: {rows_added:,}")
        print(f"Duration:             {duration}s")
        return {
            "dry_run": True,
            "valid": True,
            "as_of_date": as_of_date,
            "rows_added": rows_added,
            "duration": duration,
            "validation_report": val_report
        }

    # 3. Create versioned output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = VERSIONS_DIR / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)
    version_serving_dir = version_dir / "serving"
    version_serving_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data snapshot in version directory
    version_raw_path = version_dir / "raw_data.parquet"
    df_merged.to_parquet(version_raw_path, index=False)
    print(f"Saved version raw dataset to: {version_raw_path}")

    # Set up config with version-specific paths
    v_config = config.copy()
    v_paths = config["paths"].copy()
    v_paths["raw_data"] = str(version_raw_path.relative_to(BASE_DIR))
    v_paths["clean_prices"] = str((version_dir / "prices_clean.parquet").relative_to(BASE_DIR))
    v_paths["features_targets"] = str((version_dir / "features_targets.parquet").relative_to(BASE_DIR))
    v_paths["latest_features"] = str((version_dir / "latest_features.parquet").relative_to(BASE_DIR))
    v_paths["rankable_mask"] = str((version_dir / "rankable_mask.parquet").relative_to(BASE_DIR))
    v_config["paths"] = v_paths

    # 4. Run data cleaning
    print("\nStep 2: Running cleaning and segment detection...")
    df_clean, df_adj, df_flags, _ = clean_data(v_config)

    # 5. Run feature engineering and targets
    print("\nStep 3: Building technical features and targets...")
    df_feats, _, _ = build_features_and_targets(v_config)

    # 6. Build rankable mask
    print("\nStep 4: Computing rankable mask...")
    build_rankable_mask(v_config)

    # 7. Retrain production models if requested
    if retrain:
        print("\nStep 5: Retraining production models on updated data...")
        train_production_models(v_config)
    else:
        print("\nStep 5: Using existing production models (no retraining requested).")

    # 8. Score as-of date into version serving directory
    print(f"\nStep 6: Scoring all servable models as of {as_of_date}...")
    score_res = score_as_of(
        as_of_date=as_of_date,
        config=v_config,
        out_serving_dir=version_serving_dir
    )

    # 9. Sync active processed files and serving directory
    print("\nStep 7: Synchronizing active processed and serving artifacts...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(version_dir / "prices_clean.parquet", PROCESSED_DIR / "prices_clean.parquet")
    shutil.copy2(version_dir / "features_targets.parquet", PROCESSED_DIR / "features_targets.parquet")
    shutil.copy2(version_dir / "latest_features.parquet", PROCESSED_DIR / "latest_features.parquet")
    shutil.copy2(version_dir / "rankable_mask.parquet", PROCESSED_DIR / "rankable_mask.parquet")

    # Mirror version serving dir into data/serving/{as_of}/
    active_serving_target = SERVING_DIR / as_of_date
    if active_serving_target.exists():
        shutil.rmtree(active_serving_target)
    shutil.copytree(version_serving_dir, active_serving_target)

    # 10. Atomic pointer swap
    print("Step 8: Atomically updating CURRENT pointers...")
    atomic_write_text(SERVING_DIR / "CURRENT", as_of_date)
    atomic_write_text(VERSIONS_DIR / "CURRENT", timestamp)

    duration = round(time.time() - start_time, 2)
    refresh_log = {
        "version_id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "as_of_date": as_of_date,
        "mode": mode,
        "input_file": str(input_path),
        "rows_added": rows_added,
        "total_rows": len(df_merged),
        "duration_seconds": duration,
        "retrained": retrain,
        "validation_warnings": val_report["warnings"]
    }

    # Save reports in version folder
    with open(version_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    with open(version_dir / "refresh_log.json", "w", encoding="utf-8") as f:
        json.dump(refresh_log, f, indent=2)

    print(f"\n=======================================================")
    print(f"=== DATA REFRESH COMPLETE ===")
    print(f"Version ID:        {timestamp}")
    print(f"As-Of Date:        {as_of_date}")
    print(f"Rows Added:        {rows_added:,}")
    print(f"Total Rows:        {len(df_merged):,}")
    print(f"Duration:          {duration}s")
    print(f"Serving Pointer:   data/serving/CURRENT -> {as_of_date}")
    print(f"Version Pointer:   data/versions/CURRENT -> {timestamp}")
    print(f"=======================================================\n")
    return refresh_log


def main():
    parser = argparse.ArgumentParser(description="Stock Pulse Data Refresh Pipeline")
    parser.add_argument("--input", required=True, help="Path to input Excel, CSV, or Parquet file")
    parser.add_argument("--mode", choices=["append", "replace"], default="append", help="Refresh mode: append or replace")
    parser.add_argument("--retrain", action="store_true", help="Retrain production models on refreshed data")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without modifying disk state")
    parser.add_argument("--force", action="store_true", help="Force refresh even if data gap exceeds 60 days")

    args = parser.parse_args()
    run_refresh(
        input_file=args.input,
        mode=args.mode,
        retrain=args.retrain,
        dry_run=args.dry_run,
        force=args.force
    )


if __name__ == "__main__":
    main()

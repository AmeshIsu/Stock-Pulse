"""Execution script to run the full audit, cleaning, and feature engineering pipeline for Stock Pulse."""
from src.data.config_loader import load_config
from src.data.audit import run_audit
from src.data.clean import clean_data
from src.features.build_features import build_features_and_targets

def main():
    config = load_config()
    print("=== STEP 1: RUNNING DATA AUDIT ===")
    audit_summary = run_audit(config)
    
    print("\n=== STEP 2: RUNNING DATA CLEANING (PER SERIES_ID) ===")
    df_clean, df_adj, df_flags, df_usability = clean_data(config)
    
    print("\n=== STEP 3: RUNNING FEATURE & TARGET ENGINEERING ===")
    df_features, df_avail, df_latest = build_features_and_targets(config)
    
    print("\n=== PIPELINE EXECUTION SUMMARY ===")
    print(f"Total Rows In Raw Dataset:         {audit_summary['total_rows']:,}")
    print(f"Total Rows In Clean Parquet:       {len(df_clean):,}")
    print(f"Total In-Universe Voting Rows:     {len(df_features):,}")
    print(f"Split Adjustments Logged:          {len(df_adj):,}")
    print(f"Suspect Moves Flagged:             {len(df_flags):,}")
    print(f"Clean Price Segments Created:      {df_clean['segment_id'].nunique():,}")
    print(f"Feature & Target Columns Produced: {len(df_features.columns):,}")
    print(f"Latest Inference Snapshot Series:  {len(df_latest):,}")

if __name__ == "__main__":
    main()

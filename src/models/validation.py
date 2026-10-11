"""Walk-forward annual fold builder with strict purging and embargo for Stock Pulse."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.config_loader import load_config

@dataclass
class Fold:
    """Dataclass holding training and testing slice metadata and indices."""
    horizon: str
    fold_name: str
    split_type: str  # 'dev', 'stress', 'holdout'
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    num_train_purged: int
    num_test: int
    avg_rankable_stocks: float
    num_test_dates: int


def build_folds(df: pd.DataFrame, config: dict | None = None) -> list[Fold]:
    """Construct walk-forward annual expanding-window folds with purge and embargo."""
    if config is None:
        config = load_config()
        
    val_cfg = config["validation"]
    horizons = config["horizons"]
    first_years = val_cfg["first_test_year"]
    last_dev_year = val_cfg["last_dev_year"]
    stress_year = val_cfg["stress_year"]
    holdout_years = val_cfg["holdout_years"]
    embargo_days = val_cfg.get("embargo_trading_days", 5)
    min_train_rows = val_cfg.get("min_train_rows", 200)
    min_cross_section = val_cfg.get("min_cross_section", 8)
    
    # Calculate target end dates per horizon within each (series_id, segment_id)
    df = df.copy()
    for h_name, h_val in horizons.items():
        df[f"target_end_date_{h_name}"] = df.groupby(["series_id", "segment_id"])["Date"].shift(-h_val)
        
    all_trading_dates = sorted(df["Date"].unique())
    folds = []
    
    for h_name, h_val in horizons.items():
        target_col = f"fwd_rankpct_{h_name}"
        target_end_col = f"target_end_date_{h_name}"
        start_year = first_years[h_name]
        
        # 1. Dev Folds (Annual walk-forward blocks from start_year to last_dev_year)
        for yr in range(start_year, last_dev_year + 1):
            test_start = pd.Timestamp(f"{yr}-01-01")
            test_end = pd.Timestamp(f"{yr}-12-31")
            
            # Find trading dates strictly before test_start for embargo boundary
            prior_dates = [d for d in all_trading_dates if d < test_start]
            if not prior_dates:
                continue
            embargo_cutoff = prior_dates[-embargo_days] if len(prior_dates) >= embargo_days else prior_dates[0]
            
            # Training rows before test start with non-null target
            train_mask = (df["Date"] < test_start) & df[target_col].notna()
            # PURGE & EMBARGO: target_end_date must be strictly before embargo_cutoff
            # and trading date must be strictly before embargo_cutoff
            purged_train_mask = train_mask & (df[target_end_col] < embargo_cutoff) & (df["Date"] < embargo_cutoff)
            
            # Testing rows in year yr with non-null target
            test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end) & df[target_col].notna()
            
            train_idx = df.index[purged_train_mask].to_numpy()
            test_idx = df.index[test_mask].to_numpy()
            
            if len(train_idx) < min_train_rows or len(test_idx) == 0:
                continue
                
            test_dates = df.loc[test_idx, "Date"].nunique()
            avg_stocks = len(test_idx) / test_dates if test_dates > 0 else 0.0
            
            if avg_stocks < min_cross_section:
                continue
                
            folds.append(Fold(
                horizon=h_name,
                fold_name=f"dev_{yr}",
                split_type="dev",
                train_indices=train_idx,
                test_indices=test_idx,
                train_start=df.loc[train_idx, "Date"].min(),
                train_end=df.loc[train_idx, "Date"].max(),
                test_start=df.loc[test_idx, "Date"].min(),
                test_end=df.loc[test_idx, "Date"].max(),
                num_train_purged=len(train_idx),
                num_test=len(test_idx),
                avg_rankable_stocks=round(avg_stocks, 2),
                num_test_dates=test_dates
            ))
            
        # 2. Stress Fold (2020: Short and Mid horizons only)
        if h_name in ["short", "mid"]:
            test_start = pd.Timestamp(f"{stress_year}-01-01")
            test_end = pd.Timestamp(f"{stress_year}-12-31")
            
            prior_dates = [d for d in all_trading_dates if d < test_start]
            embargo_cutoff = prior_dates[-embargo_days] if len(prior_dates) >= embargo_days else prior_dates[0]
            
            train_mask = (df["Date"] < test_start) & df[target_col].notna()
            purged_train_mask = train_mask & (df[target_end_col] < embargo_cutoff) & (df["Date"] < embargo_cutoff)
            test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end) & df[target_col].notna()
            
            train_idx = df.index[purged_train_mask].to_numpy()
            test_idx = df.index[test_mask].to_numpy()
            
            if len(train_idx) >= min_train_rows and len(test_idx) > 0:
                test_dates = df.loc[test_idx, "Date"].nunique()
                avg_stocks = len(test_idx) / test_dates if test_dates > 0 else 0.0
                if avg_stocks >= min_cross_section:
                    folds.append(Fold(
                        horizon=h_name,
                        fold_name=f"stress_{stress_year}",
                        split_type="stress",
                        train_indices=train_idx,
                        test_indices=test_idx,
                        train_start=df.loc[train_idx, "Date"].min(),
                        train_end=df.loc[train_idx, "Date"].max(),
                        test_start=df.loc[test_idx, "Date"].min(),
                        test_end=df.loc[test_idx, "Date"].max(),
                        num_train_purged=len(train_idx),
                        num_test=len(test_idx),
                        avg_rankable_stocks=round(avg_stocks, 2),
                        num_test_dates=test_dates
                    ))
                    
        # 3. Final Holdout Fold (2024–2025: All horizons)
        holdout_start = pd.Timestamp(f"{holdout_years[0]}-01-01")
        holdout_end = pd.Timestamp(f"{holdout_years[-1]}-12-31")
        
        prior_dates = [d for d in all_trading_dates if d < holdout_start]
        embargo_cutoff = prior_dates[-embargo_days] if len(prior_dates) >= embargo_days else prior_dates[0]
        
        train_mask = (df["Date"] < holdout_start) & df[target_col].notna()
        purged_train_mask = train_mask & (df[target_end_col] < embargo_cutoff) & (df["Date"] < embargo_cutoff)
        test_mask = (df["Date"] >= holdout_start) & (df["Date"] <= holdout_end) & df[target_col].notna()
        
        train_idx = df.index[purged_train_mask].to_numpy()
        test_idx = df.index[test_mask].to_numpy()
        
        if len(train_idx) >= min_train_rows and len(test_idx) > 0:
            test_dates = df.loc[test_idx, "Date"].nunique()
            avg_stocks = len(test_idx) / test_dates if test_dates > 0 else 0.0
            if avg_stocks >= min_cross_section:
                folds.append(Fold(
                    horizon=h_name,
                    fold_name=f"holdout_{holdout_years[0]}_{holdout_years[-1]}",
                    split_type="holdout",
                    train_indices=train_idx,
                    test_indices=test_idx,
                    train_start=df.loc[train_idx, "Date"].min(),
                    train_end=df.loc[train_idx, "Date"].max(),
                    test_start=df.loc[test_idx, "Date"].min(),
                    test_end=df.loc[test_idx, "Date"].max(),
                    num_train_purged=len(train_idx),
                    num_test=len(test_idx),
                    avg_rankable_stocks=round(avg_stocks, 2),
                    num_test_dates=test_dates
                ))
                
    return folds


def save_fold_summary(folds: list[Fold], out_csv: Path | str) -> pd.DataFrame:
    """Save summary of all generated folds to reports/folds.csv."""
    rows = []
    for f in folds:
        rows.append({
            "horizon": f.horizon,
            "fold_name": f.fold_name,
            "split_type": f.split_type,
            "train_start": f.train_start.strftime("%Y-%m-%d"),
            "train_end": f.train_end.strftime("%Y-%m-%d"),
            "rows_after_purge": f.num_train_purged,
            "test_start": f.test_start.strftime("%Y-%m-%d"),
            "test_end": f.test_end.strftime("%Y-%m-%d"),
            "test_rows": f.num_test,
            "avg_rankable_stocks": f.avg_rankable_stocks,
            "num_test_dates": f.num_test_dates
        })
    df_folds = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(out_path, index=False)
    print(f"Saved fold summary table to: {out_path} ({len(df_folds)} folds).")
    return df_folds


def get_series_coverage(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compute per-series coverage of valid core features and valid targets per horizon."""
    if config is None:
        config = load_config()
    core_features = config["feature_sets"]["core"]
    
    rows = []
    for sid, grp in df.groupby("series_id"):
        # Complete core features count
        has_core = grp[core_features].notna().all(axis=1)
        core_cnt = has_core.sum()
        
        row_dict = {
            "series_id": sid,
            "CompanyCode": grp["CompanyCode"].iloc[0],
            "total_rows": len(grp),
            "valid_core_features": core_cnt,
            "valid_target_short": grp["fwd_rankpct_short"].notna().sum(),
            "valid_target_mid": grp["fwd_rankpct_mid"].notna().sum(),
            "valid_target_long": grp["fwd_rankpct_long"].notna().sum(),
            "flagged": "FLAG: Limited trading rows (interleaved share class history)" if sid in ["SEYB.N", "HNB.N"] else "Normal"
        }
        rows.append(row_dict)
        
    return pd.DataFrame(rows)

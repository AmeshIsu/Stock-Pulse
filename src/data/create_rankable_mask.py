"""Generate and save the common rankable mask for all models and baselines."""
from pathlib import Path
import pandas as pd
from src.data.config_loader import load_config

def build_rankable_mask(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()
    
    base_dir = Path(__file__).resolve().parents[2]
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(features_path)
    core_feats = config["feature_sets"]["core"]
    turnover_feats = ["log_turnover_21d", "amihud_illiq_63d"]
    core_non_turnover = [c for c in core_feats if c not in turnover_feats]
    
    # A rankable row: all core features EXCEPT turnover ones are non-null
    rankable_series = df[core_non_turnover].notna().all(axis=1)
    
    mask_df = pd.DataFrame({
        "Date": df["Date"],
        "series_id": df["series_id"],
        "rankable": rankable_series
    }, index=df.index)
    
    mask_df.to_parquet(mask_path, engine="pyarrow", index=True)
    print(f"Saved rankable mask to: {mask_path}")
    
    # Print share of rows kept per year in ranking universe
    df_u = df[df["in_ranking_universe"]].copy()
    df_u["rankable"] = rankable_series.loc[df_u.index]
    df_u["year"] = pd.to_datetime(df_u["Date"]).dt.year
    
    by_year = df_u.groupby("year")["rankable"].agg(["count", "sum", "mean"])
    print("\n--- Share of Rows Kept per Year (Ranking Universe) ---")
    for yr, row in by_year.iterrows():
        print(f"Year {yr}: {int(row['sum'])} / {int(row['count'])} ({row['mean']:.2%})")
        
    overall_share = df_u["rankable"].mean()
    print(f"\nOverall Ranking Universe Rankable Share: {df_u['rankable'].sum()} / {len(df_u)} ({overall_share:.2%})")
    return mask_df

if __name__ == "__main__":
    build_rankable_mask()

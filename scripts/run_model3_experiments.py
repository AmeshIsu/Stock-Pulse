"""Run Model 3 (LightGBM) experiments across walk-forward folds."""
import argparse
import copy
from pathlib import Path
import sys
import time
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.contract import run_experiment
from src.models.boosting_models import (
    get_lgbm_specs,
    _LGBM_HYPERPARAMS_LOG,
    reset_lgbm_hyperparams_log
)

def run_lgbm_horizon(horizon: str, force: bool = False):
    config = load_config()
    specs = get_lgbm_specs(config)
    
    base_dir = Path(__file__).resolve().parents[1]
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    oos_dir.mkdir(parents=True, exist_ok=True)
    
    # Isolate results path per horizon to allow safe parallel execution
    config_h = copy.deepcopy(config)
    config_h["paths"]["results_long_csv"] = f"reports/results_long_{horizon}.csv"
    
    reset_lgbm_hyperparams_log()
    print(f"=== RUNNING LIGHTGBM EXPERIMENTS: Horizon={horizon} ===")
    t_start = time.time()
    
    for spec, f_set in specs:
        oos_file = oos_dir / f"{spec.name}_{horizon}_{f_set}.parquet"
        if oos_file.exists() and not force:
            print(f"[SKIP] OOS predictions already exist for {spec.name}_{horizon}_{f_set}: {oos_file}")
            continue
            
        t0 = time.time()
        print(f"\n--> Running {spec.name.upper()} | Horizon: {horizon} | Feature Set: {f_set}...")
        run_experiment(spec, horizon=horizon, feature_set=f_set, config=config_h)
        print(f"Completed {spec.name}_{horizon}_{f_set} in {time.time() - t0:.1f}s.")
        
    hp_path = base_dir / f"reports/06_lgbm_hyperparams_{horizon}.csv"
    if _LGBM_HYPERPARAMS_LOG:
        hp_df = pd.DataFrame(_LGBM_HYPERPARAMS_LOG)
        hp_df.to_csv(hp_path, index=False)
        print(f"Saved {horizon} hyperparameters to: {hp_path} ({len(hp_df)} rows)")
        
    print(f"\nFinished horizon {horizon} in {time.time() - t_start:.1f}s.")


def merge_results():
    base_dir = Path(__file__).resolve().parents[1]
    main_results_path = base_dir / "reports/results_long.csv"
    df_main = pd.read_csv(main_results_path) if main_results_path.exists() else pd.DataFrame()
    
    # Merge horizon results
    for h in ["short", "mid", "long"]:
        h_res_path = base_dir / f"reports/results_long_{h}.csv"
        if h_res_path.exists():
            df_h = pd.read_csv(h_res_path)
            # Remove any matching lgbm entries from main
            lgbm_models = df_h["model"].unique()
            df_main = df_main[~(
                (df_main["horizon"] == h) &
                (df_main["model"].isin(lgbm_models))
            )]
            df_main = pd.concat([df_main, df_h], ignore_index=True)
            print(f"Merged {len(df_h)} rows from {h_res_path}")
            
    df_main.to_csv(main_results_path, index=False)
    print(f"Updated main results_long.csv: total {len(df_main)} rows")
    
    # Merge hyperparams
    hp_main_path = base_dir / "reports/06_lgbm_hyperparams.csv"
    hp_dfs = []
    for h in ["short", "mid", "long"]:
        hp_h_path = base_dir / f"reports/06_lgbm_hyperparams_{h}.csv"
        if hp_h_path.exists():
            hp_dfs.append(pd.read_csv(hp_h_path))
    if hp_dfs:
        hp_combined = pd.concat(hp_dfs, ignore_index=True)
        hp_combined.to_csv(hp_main_path, index=False)
        print(f"Updated reports/06_lgbm_hyperparams.csv: total {len(hp_combined)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="all", choices=["short", "mid", "long", "all", "merge"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    if args.horizon == "merge":
        merge_results()
    elif args.horizon in ["short", "mid", "long"]:
        run_lgbm_horizon(args.horizon, force=args.force)
    else:
        for h in ["short", "mid", "long"]:
            run_lgbm_horizon(h, force=args.force)
        merge_results()

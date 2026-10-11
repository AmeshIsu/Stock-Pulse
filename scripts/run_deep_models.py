"""Runner script for Deep Learning models: LSTM and TabPFN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import pandas as pd
from src.data.config_loader import load_config
from src.models.contract import ModelSpec, run_experiment
from src.models.deep_models import LSTMEstimator
from src.models.tabpfn_models import TabPFNRegressorEstimator, TabPFNClassifierEstimator

def main():
    config = load_config()
    horizons = ["short", "mid", "long"]
    total_start = time.time()
    
    print("=" * 60)
    print("STARTING DEEP LEARNING EXPERIMENTS: LSTM & TABPFN")
    print("=" * 60)
    
    # 1. Run LSTM models across all 3 horizons
    for h in horizons:
        print(f"\n>>> Running LSTM Regression for Horizon: {h.upper()} <<<")
        t0 = time.time()
        lstm_spec = ModelSpec(
            name="lstm_reg",
            kind="regression",
            build_fn=lambda h=h: LSTMEstimator(horizon=h),
            preprocessing="raw",
            target_column=f"fwd_rankpct_{h}"
        )
        run_experiment(lstm_spec, horizon=h, feature_set="core", config=config)
        print(f"Finished LSTM for {h.upper()} in {time.time()-t0:.1f}s")
        
    # 2. Run TabPFN Regressor across all 3 horizons
    for h in horizons:
        print(f"\n>>> Running TabPFN Regressor for Horizon: {h.upper()} <<<")
        t0 = time.time()
        tabpfn_reg_spec = ModelSpec(
            name="tabpfn_reg",
            kind="regression",
            build_fn=lambda h=h: TabPFNRegressorEstimator(horizon=h),
            preprocessing="raw",
            target_column=f"fwd_rankpct_{h}"
        )
        run_experiment(tabpfn_reg_spec, horizon=h, feature_set="core", config=config)
        print(f"Finished TabPFN Regressor for {h.upper()} in {time.time()-t0:.1f}s")

    # 3. Run TabPFN Classifier across all 3 horizons
    for h in horizons:
        print(f"\n>>> Running TabPFN Classifier for Horizon: {h.upper()} <<<")
        t0 = time.time()
        tabpfn_clf_spec = ModelSpec(
            name="tabpfn_clf",
            kind="classification",
            build_fn=lambda h=h: TabPFNClassifierEstimator(horizon=h),
            preprocessing="raw",
            target_column=f"outperform_{h}"
        )
        run_experiment(tabpfn_clf_spec, horizon=h, feature_set="core", config=config)
        print(f"Finished TabPFN Classifier for {h.upper()} in {time.time()-t0:.1f}s")
        
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"ALL DEEP LEARNING EXPERIMENTS COMPLETED IN {total_elapsed/60:.2f} MINUTES ({total_elapsed:.1f}s)")
    print("=" * 60)

if __name__ == "__main__":
    main()

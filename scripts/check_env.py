import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import tabpfn
from tabpfn import TabPFNRegressor, TabPFNClassifier
import pandas as pd
from src.data.config_loader import load_config
from src.models.validation import build_folds

print("=== 0. PRE-CHECKS: ENVIRONMENT & PACKAGES ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"TabPFN version: {tabpfn.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute device: {device}")
if device == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

# Check TabPFN limits and weights
reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", device=device, ignore_pretraining_limits=True)
clf = TabPFNClassifier(model_path="tabpfn-v2-classifier.ckpt", device=device, ignore_pretraining_limits=True)

# Max features and rows from inference config
inf_cfg = getattr(reg, "inference_config_", {})
max_features = inf_cfg.get("MAX_NUMBER_OF_FEATURES", 500)
max_rows = inf_cfg.get("MAX_NUMBER_OF_SAMPLES", 10000)
max_cpu_samples = inf_cfg.get("MAX_CPU_SAMPLES", 1000)

print(f"TabPFN MAX_NUMBER_OF_FEATURES: {max_features}")
print(f"TabPFN MAX_NUMBER_OF_SAMPLES: {max_rows}")
print(f"TabPFN MAX_CPU_SAMPLES: {max_cpu_samples}")
print(f"TabPFN Weights License: Prior Labs License (priorlabs-1-1)")
print(f"TabPFN License URL: https://github.com/PriorLabs/TabPFN/blob/49394b053a6759cfe68e90c21a2d51c31b396768/LICENSE")

print("\n=== 0. PRE-CHECKS: FOLD TABLE PER HORIZON ===")
config = load_config()
base_dir = Path(__file__).resolve().parents[1]
features_path = base_dir / config["paths"]["features_targets"]
mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")

df = pd.read_parquet(features_path)
mask_df = pd.read_parquet(mask_path)
rankable_mask = mask_df["rankable"].values
df_u = df[df["in_ranking_universe"] & rankable_mask].copy()

folds = build_folds(df_u, config)

fold_summary = []
for f in folds:
    fold_summary.append({
        "horizon": f.horizon,
        "fold_name": f.fold_name,
        "split_type": f.split_type,
        "train_len": len(f.train_indices),
        "test_len": len(f.test_indices),
        "train_start": f.train_start,
        "train_end": f.train_end,
        "test_start": f.test_start,
        "test_end": f.test_end
    })

fold_df = pd.DataFrame(fold_summary)
for h in ["short", "mid", "long"]:
    h_sub = fold_df[fold_df["horizon"] == h]
    dev_count = len(h_sub[h_sub["split_type"] == "dev"])
    stress_count = len(h_sub[h_sub["split_type"] == "stress"])
    holdout_count = len(h_sub[h_sub["split_type"] == "holdout"])
    total_count = len(h_sub)
    print(f"\nHorizon: {h.upper()}")
    print(f"  Dev folds: {dev_count}")
    print(f"  Stress folds: {stress_count}")
    print(f"  Holdout folds: {holdout_count}")
    print(f"  Total folds: {total_count}")
    print(h_sub[["fold_name", "split_type", "train_start", "train_end", "test_start", "test_end", "train_len", "test_len"]].to_string(index=False))
    
    if h == "short":
        assert dev_count == 12, f"Expected 12 dev folds for short, got {dev_count}"
        assert stress_count == 1, f"Expected 1 stress fold for short, got {stress_count}"
        assert holdout_count == 1, f"Expected 1 holdout fold for short, got {holdout_count}"
    elif h == "mid":
        assert dev_count == 12, f"Expected 12 dev folds for mid, got {dev_count}"
        assert stress_count == 1, f"Expected 1 stress fold for mid, got {stress_count}"
        assert holdout_count == 1, f"Expected 1 holdout fold for mid, got {holdout_count}"
    elif h == "long":
        assert dev_count == 10, f"Expected 10 dev folds for long, got {dev_count}"
        assert stress_count == 0, f"Expected 0 stress fold for long, got {stress_count}"
        assert holdout_count == 1, f"Expected 1 holdout fold for long, got {holdout_count}"

print("\nALL FOLD ASSERTIONS PASSED!")

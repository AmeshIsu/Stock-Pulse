import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import torch
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor, TabPFNClassifier
from src.data.config_loader import load_config
from src.models.validation import build_folds

print("Testing 1 fold TabPFN on CUDA...")
config = load_config()
base_dir = Path(__file__).resolve().parents[1]
features_path = base_dir / config["paths"]["features_targets"]
mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")

df = pd.read_parquet(features_path)
mask_df = pd.read_parquet(mask_path)
df_u = df[df["in_ranking_universe"] & mask_df["rankable"].values].copy()

folds = build_folds(df_u, config)
short_dev_0 = [f for f in folds if f.horizon == "short" and f.split_type == "dev"][0]

feature_cols = list(config["feature_sets"]["core"])
train_df = df_u.loc[short_dev_0.train_indices].copy()
test_df = df_u.loc[short_dev_0.test_indices].copy()

# Median fill
train_medians = train_df[feature_cols].median()
train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)
test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)

# Build context: evenly spaced trading dates across whole training history
dates = sorted(train_df["Date"].unique())
n_ctx_target = min(10000, len(train_df))

# Stride selection
stride = 1
while True:
    selected_dates = dates[::stride]
    sel_rows = train_df[train_df["Date"].isin(selected_dates)]
    if len(sel_rows) <= n_ctx_target:
        break
    stride += 1

context_df = train_df[train_df["Date"].isin(selected_dates)].copy()
print(f"Train dates: {len(dates)}, Stride: {stride}, Context rows: {len(context_df)}, Target: {n_ctx_target}")

X_tr = context_df[feature_cols].values.astype(np.float32)
y_tr = context_df["fwd_rankpct_short"].values.astype(np.float32)
X_te = test_df[feature_cols].values.astype(np.float32)

print(f"Test rows: {len(X_te)}")
t0 = time.time()
reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", device="cuda", ignore_pretraining_limits=True, random_state=42)
reg.fit(X_tr, y_tr)
t_fit = time.time() - t0
print(f"Fit took {t_fit:.2f}s")

t1 = time.time()
preds = reg.predict(X_te)
t_pred = time.time() - t1
print(f"Predict {len(X_te)} rows took {t_pred:.2f}s! Total fold time: {t_fit+t_pred:.2f}s")

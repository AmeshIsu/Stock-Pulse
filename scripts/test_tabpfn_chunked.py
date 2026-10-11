import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] = "250"
import time
import torch
import numpy as np
from tabpfn import TabPFNRegressor

X_tr = np.random.randn(4000, 30).astype(np.float32)
y_tr = np.random.uniform(0, 1, size=4000).astype(np.float32)
X_te = np.random.randn(1000, 30).astype(np.float32)

t0 = time.time()
reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", device="cuda", ignore_pretraining_limits=True, n_estimators=4, fit_mode="fit_with_cache", random_state=42)
reg.fit(X_tr, y_tr)
t1 = time.time()

# Chunked predict
chunk_size = 250
preds = []
for i in range(0, len(X_te), chunk_size):
    p = reg.predict(X_te[i:i+chunk_size])
    preds.append(p)
preds = np.concatenate(preds)
t2 = time.time()

print(f"Fit (cache): {t1-t0:.2f}s, Chunked Pred (1000 rows): {t2-t1:.2f}s, Total: {t2-t0:.2f}s", flush=True)

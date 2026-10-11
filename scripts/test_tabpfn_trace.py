import sys
import time
print("Importing torch...", flush=True)
import torch
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print("Importing tabpfn...", flush=True)
from tabpfn import TabPFNRegressor
print("Initializing TabPFNRegressor...", flush=True)
reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", device="cuda" if torch.cuda.is_available() else "cpu", ignore_pretraining_limits=True)
print("Initialized successfully!", flush=True)

import numpy as np
X_tr = np.random.randn(1000, 30).astype(np.float32)
y_tr = np.random.uniform(0, 1, size=1000).astype(np.float32)
X_te = np.random.randn(250, 30).astype(np.float32)

print("Fitting 1000 rows...", flush=True)
t0 = time.time()
reg.fit(X_tr, y_tr)
print(f"Fitted in {time.time()-t0:.2f}s", flush=True)

print("Predicting 250 rows...", flush=True)
t1 = time.time()
preds = reg.predict(X_te)
print(f"Predicted in {time.time()-t1:.2f}s", flush=True)

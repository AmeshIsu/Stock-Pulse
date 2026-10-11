import time
import torch
import numpy as np
from tabpfn import TabPFNRegressor

print(f"CUDA Available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

for n_ctx in [4000, 10000]:
    X_tr = np.random.randn(n_ctx, 30).astype(np.float32)
    y_tr = np.random.uniform(0, 1, size=n_ctx).astype(np.float32)
    X_te = np.random.randn(250, 30).astype(np.float32)

    t0 = time.time()
    reg = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt", device=device, ignore_pretraining_limits=True)
    reg.fit(X_tr, y_tr)
    t_fit = time.time() - t0

    t1 = time.time()
    preds = reg.predict(X_te)
    t_pred = time.time() - t1
    print(f"CUDA N_ctx={n_ctx}: Fit={t_fit:.2f}s, Pred (250 rows)={t_pred:.2f}s, Total={t_fit+t_pred:.2f}s")

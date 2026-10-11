import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from src.data.config_loader import load_config
from src.models.validation import build_folds

print("Testing 1 fold LSTM training on CUDA...")
config = load_config()
base_dir = Path(__file__).resolve().parents[1]
features_path = base_dir / config["paths"]["features_targets"]
mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")

df = pd.read_parquet(features_path)
mask_df = pd.read_parquet(mask_path)
df_u = df[df["in_ranking_universe"] & mask_df["rankable"].values].copy()

folds = build_folds(df_u, config)
short_dev_0 = [f for f in folds if f.horizon == "short" and f.split_type == "dev"][0]

print(f"Fold: {short_dev_0.fold_name}, Train len: {len(short_dev_0.train_indices)}, Test len: {len(short_dev_0.test_indices)}")

# Create synthetic sequence tensors of exact dimensions
N_tr = len(short_dev_0.train_indices) // 3
N_val = len(short_dev_0.train_indices) // 10
N_te = len(short_dev_0.test_indices)

X_tr = torch.randn(N_tr, 63, 6, device="cuda")
y_tr = torch.rand(N_tr, device="cuda")
X_val = torch.randn(N_val, 63, 6, device="cuda")
y_val = torch.rand(N_val, device="cuda")
X_te = torch.randn(N_te, 63, 6, device="cuda")

class StockLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.dropout(out[:, -1, :])
        return self.head(feat).squeeze(-1)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)

t0 = time.time()
for seed in [42, 43, 44]:
    torch.manual_seed(seed)
    model = StockLSTM().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(15):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()

elapsed = time.time() - t0
print(f"3 seeds x 15 epochs trained and evaluated in {elapsed:.2f}s!")

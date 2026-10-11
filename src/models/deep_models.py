"""Deep learning models for Stock Pulse: 1-layer LSTM cross-sectional ranker."""
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data.config_loader import load_config
from src.models.metrics import compute_rank_ic


class StockLSTM(nn.Module):
    """1-layer LSTM with dropout and linear head."""
    def __init__(self, input_size: int = 6, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len=63, input_size=6)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        feat = self.dropout(last_step)
        pred = self.head(feat)
        return pred.squeeze(-1)


def compute_lstm_channels(df_full: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 causal channels for LSTM input on the full dataset.
    
    Channels:
    (1) daily log return from adjusted Close;
    (2) daily log return minus the equal-weight universe index return that day (0 if undefined);
    (3) log(High/Low);
    (4) own-history z-score of log trade volume (trailing 126 rows within segment, min 63);
    (5) own-history z-score of log turnover, with missing values set to 0 (missing 2016 filled with 0);
    (6) a zero-volume-day indicator.
    """
    df = df_full.sort_values(["series_id", "Date"]).copy()
    
    # Channel 1: daily log return
    if "log_return" in df.columns:
        log_ret = df["log_return"].values
    else:
        log_ret = df.groupby(["series_id", "segment_id"])["Close"].apply(lambda s: np.log(s / s.shift(1))).values
    df["ch1_log_ret"] = np.nan_to_num(log_ret, nan=0.0)

    # Channel 2: daily log return minus equal-weight universe index return
    in_u = df["in_ranking_universe"].fillna(False).values
    mkt_ret_map = df[in_u].groupby("Date")["ch1_log_ret"].mean()
    mkt_ret = df["Date"].map(mkt_ret_map).fillna(0.0).values
    df["ch2_excess_ret"] = df["ch1_log_ret"] - mkt_ret

    # Channel 3: log(High/Low)
    high = np.maximum(df["High"].values, 1e-6)
    low = np.maximum(df["Low"].values, 1e-6)
    ratio = np.maximum(high / low, 1.0)
    df["ch3_log_hl"] = np.log(ratio)

    # Channel 4: own-history z-score of log trade volume (trailing 126 rows, min 63)
    log_vol = np.log1p(np.maximum(df["TradeVolume"].fillna(0.0).values, 0.0))
    df["_log_vol"] = log_vol
    m126_v = df.groupby(["series_id", "segment_id"])["_log_vol"].rolling(126, min_periods=63).mean().values
    s126_v = df.groupby(["series_id", "segment_id"])["_log_vol"].rolling(126, min_periods=63).std(ddof=1).values
    vol_ownz = np.where(s126_v > 1e-9, (log_vol - m126_v) / s126_v, 0.0)
    df["ch4_vol_ownz"] = np.nan_to_num(vol_ownz, nan=0.0)

    # Channel 5: own-history z-score of log turnover (trailing 126 rows, min 63, missing=0)
    to_raw = df["Turnover"].values
    to_valid = np.where(pd.notnull(to_raw) & (to_raw >= 0), np.log1p(to_raw), np.nan)
    df["_log_to"] = to_valid
    m126_to = df.groupby(["series_id", "segment_id"])["_log_to"].rolling(126, min_periods=63).mean().values
    s126_to = df.groupby(["series_id", "segment_id"])["_log_to"].rolling(126, min_periods=63).std(ddof=1).values
    to_ownz = np.where(s126_to > 1e-9, (to_valid - m126_to) / s126_to, 0.0)
    df["ch5_to_ownz"] = np.nan_to_num(to_ownz, nan=0.0)

    # Channel 6: zero volume indicator
    trade_vol = df["TradeVolume"].fillna(0.0).values
    df["ch6_zero_vol"] = (trade_vol == 0).astype(float)

    return df


CHANNEL_COLS = [
    "ch1_log_ret",
    "ch2_excess_ret",
    "ch3_log_hl",
    "ch4_vol_ownz",
    "ch5_to_ownz",
    "ch6_zero_vol"
]


class LSTMEstimator:
    """LSTM Estimator adhering to Stock Pulse ModelSpec contract.
    
    Fixed hyper-parameters:
    - 1-layer LSTM, hidden_size 32, dropout 0.2, linear head
    - AdamW lr=1e-3, weight_decay=1e-3, batch_size 512, MSE loss
    - Inner validation on last 20% dates (purged by h days), max 30 epochs, patience 5
    - Retrain on full train fold for best epoch across 3 seeds [42, 43, 44]
    - Average per-date cross-sectional ranks of predictions
    """
    def __init__(
        self,
        horizon: str = "short",
        hidden_size: int = 32,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 30,
        patience: int = 5,
        train_stride: int = 3,
        seeds: list[int] | None = None,
        device: str | None = None
    ):
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_stride = train_stride
        self.seeds = seeds or [42, 43, 44]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Context set by contract
        self.train_df = None
        self.test_df = None
        self.fold = None
        self.target_col = None
        self.feature_names = None
        
        # Internal state
        self.best_epoch_ = 1
        self.best_val_ic_ = -999.0
        self.total_runtime_ = 0.0
        self.models_ = []
        self.scaler_stats_ = {}
        
    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold
        
    def set_test_context(self, test_df: pd.DataFrame):
        self.test_df = test_df

    def _fit_scalers(self, train_channel_df: pd.DataFrame):
        """Fit winsorization thresholds and z-score stats on train rows."""
        stats = {}
        # Winsorize channels 1, 2, 3 at 0.5% and 99.5%
        for ch in ["ch1_log_ret", "ch2_excess_ret", "ch3_log_hl"]:
            vals = train_channel_df[ch].values
            p05 = float(np.percentile(vals, 0.5))
            p995 = float(np.percentile(vals, 99.5))
            stats[ch] = {"p05": p05, "p995": p995}
            
        # Standardize all 6 channels on train
        for i, col in enumerate(CHANNEL_COLS, 1):
            vals = train_channel_df[col].values
            if col in stats:
                vals = np.clip(vals, stats[col]["p05"], stats[col]["p995"])
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            if std < 1e-6:
                std = 1.0
            stats[f"norm_{i}"] = {"col": col, "mean": mean, "std": std}
            
        self.scaler_stats_ = stats

    def _transform_channels(self, channel_df: pd.DataFrame) -> np.ndarray:
        """Apply train scalers, clip to [-5, 5], set NaN to 0, return (N, 6) float32 array."""
        scaled_cols = []
        for i, col in enumerate(CHANNEL_COLS, 1):
            norm_info = self.scaler_stats_[f"norm_{i}"]
            vals = channel_df[col].values.copy()
            if col in self.scaler_stats_:
                vals = np.clip(vals, self.scaler_stats_[col]["p05"], self.scaler_stats_[col]["p995"])
            mean = norm_info["mean"]
            std = norm_info["std"]
            scaled = (vals - mean) / std
            scaled = np.clip(scaled, -5.0, 5.0)
            scaled = np.nan_to_num(scaled, nan=0.0)
            scaled_cols.append(scaled.astype(np.float32))
            
        return np.column_stack(scaled_cols)

    def _extract_sequences(self, df_sample: pd.DataFrame, full_channels: np.ndarray, full_df: pd.DataFrame) -> np.ndarray:
        """Extract (N, 63, 6) causal tensor for the rows in df_sample ending at date t."""
        # Map sample row index in full_df
        sample_indices = full_df.index.get_indexer(df_sample.index)
        # Using sliding window view over full_channels (shape: total_rows, 6)
        windows = np.lib.stride_tricks.sliding_window_view(full_channels, 63, axis=0) # shape (total_rows - 62, 6, 63)
        # window index is sample_indices - 62
        seqs = windows[sample_indices - 62].swapaxes(1, 2) # shape (len(df_sample), 63, 6)
        return seqs.copy()

    def fit(self, X: np.ndarray, y: np.ndarray):
        start_time = time.time()
        config = load_config()
        h_days = config["horizons"].get(self.horizon, 21)
        base_dir = Path(__file__).resolve().parents[2]
        features_path = base_dir / config["paths"]["features_targets"]
        
        # Load full features dataset to ensure full causal history for channels
        full_df = pd.read_parquet(features_path).sort_values(["series_id", "Date"]).reset_index(drop=True)
        # Compute 6 channels on full data
        channel_df = compute_lstm_channels(full_df)
        
        # Filter train rows in channel_df
        train_indices_in_full = full_df[full_df.index.isin(self.train_df.index)].index
        train_channel_df = channel_df.loc[train_indices_in_full]
        
        # 1. Fit train-fold scalers
        self._fit_scalers(train_channel_df)
        full_transformed = self._transform_channels(channel_df)
        
        # 2. Inner validation split (last 20% of training dates, h-day purge)
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        split_idx = int(n_dates * 0.8)
        val_start = dates[split_idx]
        
        df_copy = self.train_df.copy()
        df_copy["_t_end"] = df_copy.groupby(["series_id", "segment_id"])["Date"].shift(-h_days)
        inner_tr_mask = (df_copy["Date"] < val_start) & (df_copy["_t_end"] < val_start)
        inner_val_mask = (df_copy["Date"] >= val_start)
        
        inner_tr_df = df_copy[inner_tr_mask].copy()
        inner_val_df = df_copy[inner_val_mask].copy()
        
        # Subsample inner training dates by train_stride
        inner_tr_dates = sorted(inner_tr_df["Date"].unique())
        sub_tr_dates = inner_tr_dates[::self.train_stride]
        inner_tr_df_sub = inner_tr_df[inner_tr_df["Date"].isin(sub_tr_dates)].copy()
        
        # Build sequences for inner training & validation
        X_inner_tr = self._extract_sequences(inner_tr_df_sub, full_transformed, full_df)
        y_inner_tr = inner_tr_df_sub[self.target_col].values.astype(np.float32)
        
        X_inner_val = self._extract_sequences(inner_val_df, full_transformed, full_df)
        
        # Tensor data loaders
        train_tensor_ds = TensorDataset(torch.from_numpy(X_inner_tr), torch.from_numpy(y_inner_tr))
        train_loader = DataLoader(train_tensor_ds, batch_size=self.batch_size, shuffle=True)
        X_val_tensor = torch.from_numpy(X_inner_val).to(self.device)
        
        # Early stopping loop on seed 42
        torch.manual_seed(42)
        model_inner = StockLSTM(hidden_size=self.hidden_size, dropout=self.dropout).to(self.device)
        opt = torch.optim.AdamW(model_inner.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        best_epoch = 1
        best_val_ic = -999.0
        patience_counter = 0
        target_fwd = f"fwd_excess_{self.horizon}" if f"fwd_excess_{self.horizon}" in inner_val_df.columns else self.target_col
        
        for epoch in range(1, self.max_epochs + 1):
            model_inner.train()
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                opt.zero_grad()
                pred = model_inner(b_x)
                loss = criterion(pred, b_y)
                loss.backward()
                opt.step()
                
            # Score inner val
            model_inner.eval()
            with torch.no_grad():
                val_preds = model_inner(X_val_tensor).cpu().numpy()
            inner_val_df["pred_score"] = val_preds
            ics = compute_rank_ic(inner_val_df, target_col=target_fwd, min_stocks=8)
            val_ic = float(np.mean(ics)) if len(ics) > 0 else -999.0
            
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
                    
        self.best_epoch_ = best_epoch
        self.best_val_ic_ = best_val_ic
        
        # 3. Retrain on full train fold for best_epoch using 3 seeds
        all_tr_dates = sorted(self.train_df["Date"].unique())
        sub_full_tr_dates = all_tr_dates[::self.train_stride]
        full_tr_sub = self.train_df[self.train_df["Date"].isin(sub_full_tr_dates)].copy()
        
        X_full_tr = self._extract_sequences(full_tr_sub, full_transformed, full_df)
        y_full_tr = full_tr_sub[self.target_col].values.astype(np.float32)
        
        full_train_ds = TensorDataset(torch.from_numpy(X_full_tr), torch.from_numpy(y_full_tr))
        full_loader = DataLoader(full_train_ds, batch_size=self.batch_size, shuffle=True)
        
        self.models_ = []
        for s in self.seeds:
            torch.manual_seed(s)
            model_s = StockLSTM(hidden_size=self.hidden_size, dropout=self.dropout).to(self.device)
            opt_s = torch.optim.AdamW(model_s.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
            for ep in range(1, self.best_epoch_ + 1):
                model_s.train()
                for b_x, b_y in full_loader:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    opt_s.zero_grad()
                    pred = model_s(b_x)
                    loss = criterion(pred, b_y)
                    loss.backward()
                    opt_s.step()
            model_s.eval()
            self.models_.append(model_s)
            
        elapsed = time.time() - start_time
        self.total_runtime_ = elapsed
        
        # Log to reports/07_lstm_training.csv
        log_path = base_dir / "reports" / "07_lstm_training.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_row = {
            "horizon": self.horizon,
            "fold": self.fold.fold_name if self.fold else "unknown",
            "split_type": self.fold.split_type if self.fold else "unknown",
            "best_epoch": self.best_epoch_,
            "best_val_ic": round(self.best_val_ic_, 5),
            "total_epochs_trained": self.best_epoch_,
            "runtime_sec": round(self.total_runtime_, 2)
        }
        log_df = pd.DataFrame([log_row])
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
            
        print(f"[{self.horizon.upper()}|{self.fold.fold_name if self.fold else ''}] LSTM chosen epoch: {self.best_epoch_} (val IC: {self.best_val_ic_:+.4f}) in {elapsed:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on test samples, averaging per-date cross-sectional ranks across seeds."""
        base_dir = Path(__file__).resolve().parents[2]
        config = load_config()
        features_path = base_dir / config["paths"]["features_targets"]
        
        full_df = pd.read_parquet(features_path).sort_values(["series_id", "Date"]).reset_index(drop=True)
        channel_df = compute_lstm_channels(full_df)
        full_transformed = self._transform_channels(channel_df)
        
        X_test_seq = self._extract_sequences(self.test_df, full_transformed, full_df)
        X_te_tensor = torch.from_numpy(X_test_seq).to(self.device)
        
        seed_preds = []
        for model_s in self.models_:
            model_s.eval()
            with torch.no_grad():
                preds_s = model_s(X_te_tensor).cpu().numpy()
            # Rank predictions cross-sectionally per date
            tmp_df = self.test_df[["Date"]].copy()
            tmp_df["raw_score"] = preds_s
            tmp_df["cs_rank"] = tmp_df.groupby("Date")["raw_score"].rank(pct=True)
            seed_preds.append(tmp_df["cs_rank"].values)
            
        # Average per-date cross-sectional ranks across seeds
        avg_rank = np.mean(seed_preds, axis=0)
        return avg_rank

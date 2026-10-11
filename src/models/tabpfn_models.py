"""TabPFN models for Stock Pulse: Foundation tabular prior-data-fitted network."""
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor, TabPFNClassifier
import torch

from src.data.config_loader import load_config


class TabPFNRegressorEstimator:
    """TabPFN Regressor on continuous forward rank percentile."""
    def __init__(
        self,
        horizon: str = "short",
        max_ctx: int = 4000,
        n_estimators: int = 2,
        chunk_size: int = 250,
        random_state: int = 42,
        device: str | None = None
    ):
        self.horizon = horizon
        self.max_ctx = max_ctx
        self.n_estimators = n_estimators
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_df = None
        self.test_df = None
        self.fold = None
        self.target_col = None
        self.feature_names = None
        self.model = None
        self.fit_time_sec = 0.0

    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold

    def set_test_context(self, test_df: pd.DataFrame):
        self.test_df = test_df

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.time()
        # Evenly spaced trading dates across purged training history
        dates = sorted(self.train_df["Date"].unique())
        n_dates = len(dates)
        stride = 1
        
        while True:
            sel_dates = dates[::stride]
            sel_rows = self.train_df[self.train_df["Date"].isin(sel_dates)]
            if len(sel_rows) <= self.max_ctx or stride >= n_dates:
                break
            stride += 1
            
        context_df = self.train_df[self.train_df["Date"].isin(sel_dates)].copy()
        
        # Verify context rows strictly belong to training set (never test)
        if self.fold is not None:
            assert context_df["Date"].max() <= self.fold.train_end, "Leakage: Context row beyond train_end!"
            
        X_ctx = context_df[self.feature_names].values.astype(np.float32)
        y_ctx = context_df[self.target_col].values.astype(np.float32)
        
        # Initialize and fit TabPFN Regressor
        self.model = TabPFNRegressor(
            model_path="tabpfn-v2-regressor.ckpt",
            device=self.device,
            ignore_pretraining_limits=True,
            n_estimators=self.n_estimators,
            fit_mode="fit_with_cache",
            random_state=self.random_state
        )
        self.model.fit(X_ctx, y_ctx)
        self.fit_time_sec = time.time() - t0
        
        # Log context details to reports/07_tabpfn_context.csv
        base_dir = Path(__file__).resolve().parents[2]
        log_path = base_dir / "reports" / "07_tabpfn_context.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_row = {
            "model": "tabpfn_reg",
            "horizon": self.horizon,
            "fold": self.fold.fold_name if self.fold else "unknown",
            "split_type": self.fold.split_type if self.fold else "unknown",
            "date_stride": stride,
            "total_train_dates": n_dates,
            "selected_context_dates": len(sel_dates),
            "context_rows": len(context_df),
            "max_ctx_target": self.max_ctx,
            "fit_time_sec": round(self.fit_time_sec, 2)
        }
        log_df = pd.DataFrame([log_row])
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
            
        print(f"[{self.horizon.upper()}|{self.fold.fold_name if self.fold else ''}] tabpfn_reg fit: stride={stride}, context_rows={len(context_df)} in {self.fit_time_sec:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on test samples in chunks to manage GPU memory."""
        X_test = X.astype(np.float32)
        preds = []
        for i in range(0, len(X_test), self.chunk_size):
            p = self.model.predict(X_test[i:i + self.chunk_size])
            preds.append(p)
        return np.concatenate(preds)


class TabPFNClassifierEstimator:
    """TabPFN Classifier on outperform label (tie rows dropped)."""
    def __init__(
        self,
        horizon: str = "short",
        max_ctx: int = 4000,
        n_estimators: int = 2,
        chunk_size: int = 250,
        random_state: int = 42,
        device: str | None = None
    ):
        self.horizon = horizon
        self.max_ctx = max_ctx
        self.n_estimators = n_estimators
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_df = None
        self.test_df = None
        self.fold = None
        self.target_col = None
        self.feature_names = None
        self.model = None
        self.fit_time_sec = 0.0

    def set_context(self, feature_names: list[str], train_df: pd.DataFrame, target_col: str, fold: Any):
        self.feature_names = feature_names
        self.train_df = train_df
        self.target_col = target_col
        self.fold = fold

    def set_test_context(self, test_df: pd.DataFrame):
        self.test_df = test_df

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.time()
        # Drop tie rows where target is NaN
        valid_df = self.train_df.dropna(subset=[self.target_col]).copy()
        dates = sorted(valid_df["Date"].unique())
        n_dates = len(dates)
        stride = 1
        
        while True:
            sel_dates = dates[::stride]
            sel_rows = valid_df[valid_df["Date"].isin(sel_dates)]
            if len(sel_rows) <= self.max_ctx or stride >= n_dates:
                break
            stride += 1
            
        context_df = valid_df[valid_df["Date"].isin(sel_dates)].copy()
        
        if self.fold is not None:
            assert context_df["Date"].max() <= self.fold.train_end, "Leakage: Context row beyond train_end!"
            
        X_ctx = context_df[self.feature_names].values.astype(np.float32)
        y_ctx = context_df[self.target_col].astype(int).values
        
        self.model = TabPFNClassifier(
            model_path="tabpfn-v2-classifier.ckpt",
            device=self.device,
            ignore_pretraining_limits=True,
            n_estimators=self.n_estimators,
            fit_mode="fit_with_cache",
            random_state=self.random_state
        )
        self.model.fit(X_ctx, y_ctx)
        self.fit_time_sec = time.time() - t0
        
        # Log context details to reports/07_tabpfn_context.csv
        base_dir = Path(__file__).resolve().parents[2]
        log_path = base_dir / "reports" / "07_tabpfn_context.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_row = {
            "model": "tabpfn_clf",
            "horizon": self.horizon,
            "fold": self.fold.fold_name if self.fold else "unknown",
            "split_type": self.fold.split_type if self.fold else "unknown",
            "date_stride": stride,
            "total_train_dates": n_dates,
            "selected_context_dates": len(sel_dates),
            "context_rows": len(context_df),
            "max_ctx_target": self.max_ctx,
            "fit_time_sec": round(self.fit_time_sec, 2)
        }
        log_df = pd.DataFrame([log_row])
        if log_path.exists():
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
            
        print(f"[{self.horizon.upper()}|{self.fold.fold_name if self.fold else ''}] tabpfn_clf fit: stride={stride}, context_rows={len(context_df)} in {self.fit_time_sec:.2f}s")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities on test samples in chunks."""
        X_test = X.astype(np.float32)
        probs = []
        for i in range(0, len(X_test), self.chunk_size):
            p = self.model.predict_proba(X_test[i:i + self.chunk_size])
            probs.append(p)
        probs_all = np.vstack(probs)
        return probs_all

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

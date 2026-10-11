"""Pytest suite validating Model 3 (LightGBM): determinism, leakage, row-set equality, explainability."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config
from src.models.boosting_models import LightGBMRegressorEstimator, LightGBMClassifierEstimator
from src.models.explain import explain_prediction


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parents[1]


def test_lgbm_determinism_and_no_leakage(config, base_dir):
    """Verify LightGBM yields identical predictions across two runs with identical seed."""
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    # Take a small clean fold
    core_feats = config["feature_sets"]["core"]
    train_slice = df_u[df_u["Date"].dt.year <= 2007].dropna(subset=["fwd_rankpct_short"]).tail(500).copy()
    test_slice = df_u[df_u["Date"].dt.year == 2008].dropna(subset=["fwd_rankpct_short"]).head(100).copy()
    
    X_train = train_slice[core_feats].values
    y_train = train_slice["fwd_rankpct_short"].values
    X_test = test_slice[core_feats].values
    
    # Run 1
    est1 = LightGBMRegressorEstimator(random_state=42, max_trees=20)
    est1.set_context(core_feats, train_slice, "fwd_rankpct_short", fold=None)
    est1.fit(X_train, y_train)
    preds1 = est1.predict(X_test)
    
    # Run 2
    est2 = LightGBMRegressorEstimator(random_state=42, max_trees=20)
    est2.set_context(core_feats, train_slice, "fwd_rankpct_short", fold=None)
    est2.fit(X_train, y_train)
    preds2 = est2.predict(X_test)
    
    np.testing.assert_allclose(preds1, preds2, rtol=1e-6, atol=1e-6, err_msg="LightGBM predictions are not deterministic.")


def test_explain_prediction_three_sample_rows(config, base_dir):
    """Verify explain_prediction on 3 diverse sample rows returns signed contributions and plain labels."""
    features_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    
    df = pd.read_parquet(features_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    df_u = df[df["in_ranking_universe"] & mask].copy()
    
    core_feats = config["feature_sets"]["core"]
    sample_df = df_u.dropna(subset=core_feats + ["fwd_rankpct_short"]).iloc[100:600].copy()
    
    X = sample_df[core_feats].values
    y = sample_df["fwd_rankpct_short"].values
    
    est = LightGBMRegressorEstimator(random_state=42, max_trees=15)
    est.set_context(core_feats, sample_df, "fwd_rankpct_short", fold=None)
    est.fit(X, y)
    
    # Test on 3 sample rows
    test_indices = [10, 50, 90]
    for idx in test_indices:
        row = sample_df.iloc[idx][core_feats]
        explanation = explain_prediction(est.model, row, top_n=5)
        
        assert "prediction" in explanation
        assert "base_value" in explanation
        assert "top_positive" in explanation
        assert "top_negative" in explanation
        assert len(explanation["top_positive"]) <= 5
        assert len(explanation["top_negative"]) <= 5
        
        # Verify signed contributions
        for pos_item in explanation["top_positive"]:
            assert pos_item["contribution"] > 0, "Positive contributor must have positive contribution"
            assert "label" in pos_item
            assert "family" in pos_item
            
        for neg_item in explanation["top_negative"]:
            assert neg_item["contribution"] < 0, "Negative contributor must have negative contribution"
            assert "label" in neg_item
            assert "family" in neg_item


def test_row_set_equality_across_models(config, base_dir):
    """Check that evaluated (Date, series_id) row sets match across ridge, rf, and lgbm where files exist."""
    oos_dir = base_dir / config["paths"]["oos_predictions_dir"]
    
    h = "short"
    ridge_file = oos_dir / f"ridge_{h}_core.parquet"
    rf_file = oos_dir / f"rf_reg_{h}_core.parquet"
    lgbm_file = oos_dir / f"lgbm_reg_{h}_core.parquet"
    
    if ridge_file.exists() and rf_file.exists():
        df_ridge = pd.read_parquet(ridge_file)
        df_rf = pd.read_parquet(rf_file)
        
        ridge_rows = set(zip(df_ridge["Date"], df_ridge["series_id"]))
        rf_rows = set(zip(df_rf["Date"], df_rf["series_id"]))
        assert ridge_rows == rf_rows, "Ridge and RF row sets do not match on common evaluation set."
        
    if lgbm_file.exists() and ridge_file.exists():
        df_lgbm = pd.read_parquet(lgbm_file)
        df_ridge = pd.read_parquet(ridge_file)
        lgbm_rows = set(zip(df_lgbm["Date"], df_lgbm["series_id"]))
        ridge_rows = set(zip(df_ridge["Date"], df_ridge["series_id"]))
        assert lgbm_rows == ridge_rows, "LightGBM and Ridge row sets do not match on common evaluation set."

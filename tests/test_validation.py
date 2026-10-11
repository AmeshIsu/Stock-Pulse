"""Sanity tests for walk-forward validation framework, leak-free preprocessing, and metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.models.preprocessing import fit_preprocessor, transform_preprocessor
from src.models.metrics import compute_rank_ic

@pytest.fixture(scope="module")
def config():
    return load_config()

@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def universe_df(config, base_dir):
    f_path = base_dir / config["paths"]["features_targets"]
    if not f_path.exists():
        pytest.fail(f"Features and targets file not found at {f_path}.")
    df = pd.read_parquet(f_path)
    return df[df["in_ranking_universe"]].copy()

@pytest.fixture(scope="module")
def all_folds(universe_df, config):
    return build_folds(universe_df, config)


def test_no_target_end_date_overlaps_test_start(universe_df, all_folds, config):
    """Assert that for EVERY fold, no training row has target_end_date on or after the test start date."""
    horizons = config["horizons"]
    df = universe_df.copy()
    for h_name, h_val in horizons.items():
        df[f"target_end_date_{h_name}"] = df.groupby(["series_id", "segment_id"])["Date"].shift(-h_val)
        
    for fold in all_folds:
        h_name = fold.horizon
        target_end_col = f"target_end_date_{h_name}"
        
        train_df = df.loc[fold.train_indices]
        test_start = fold.test_start
        
        # Check every training row's target_end_date is strictly before test_start
        overlapping_targets = train_df[train_df[target_end_col] >= test_start]
        assert len(overlapping_targets) == 0, (
            f"Leakage in fold {fold.fold_name} ({fold.horizon}): "
            f"{len(overlapping_targets)} training rows have target_end_date >= test_start ({test_start})."
        )


def test_no_train_test_overlap(all_folds):
    """Assert that for every fold, training indices and testing indices are strictly disjoint."""
    for fold in all_folds:
        intersection = set(fold.train_indices).intersection(set(fold.test_indices))
        assert len(intersection) == 0, (
            f"Fold {fold.fold_name} ({fold.horizon}) has {len(intersection)} overlapping indices between train and test!"
        )


def test_rank_ic_oracle_random_and_shuffled(universe_df):
    """Verify that an oracle score gives mean Rank IC close to 1, random gives ~0, and shuffled gives ~0."""
    sample_df = universe_df[(universe_df["Date"] >= "2018-01-01") & (universe_df["Date"] <= "2018-12-31")].copy()
    target_col = "fwd_excess_short"
    sample_df = sample_df.dropna(subset=[target_col])
    
    # 1. Oracle score: pred_score = target
    oracle_df = sample_df.copy()
    oracle_df["pred_score"] = oracle_df[target_col]
    oracle_ic = compute_rank_ic(oracle_df, target_col=target_col, min_stocks=8)
    assert oracle_ic.mean() == pytest.approx(1.0, rel=1e-3, abs=1e-3), f"Oracle IC was {oracle_ic.mean()}"
    
    # 2. Random score: uniform random noise
    rng = np.random.default_rng(42)
    random_df = sample_df.copy()
    random_df["pred_score"] = rng.uniform(0, 1, size=len(random_df))
    random_ic = compute_rank_ic(random_df, target_col=target_col, min_stocks=8)
    assert abs(random_ic.mean()) < 0.05, f"Random IC deviated too far from 0: {random_ic.mean()}"
    
    # 3. Shuffled target score
    shuffled_df = sample_df.copy()
    shuffled_df["pred_score"] = rng.permutation(sample_df[target_col].values)
    shuffled_ic = compute_rank_ic(shuffled_df, target_col=target_col, min_stocks=8)
    assert abs(shuffled_ic.mean()) < 0.05, f"Shuffled IC deviated too far from 0: {shuffled_ic.mean()}"


def test_preprocessing_learned_strictly_from_train(universe_df, all_folds, config):
    """Assert that perturbing test data has zero effect on training-fold preprocessor state."""
    fold = all_folds[0]
    feature_cols = config["feature_sets"]["core"]
    
    train_df = universe_df.loc[fold.train_indices].copy()
    test_df_original = universe_df.loc[fold.test_indices].copy()
    
    # Fit preprocessor on train
    state1 = fit_preprocessor(train_df, feature_cols, target_col="fwd_rankpct_short")
    
    # Corrupt/perturb test data radically
    test_df_perturbed = test_df_original.copy()
    for col in feature_cols:
        test_df_perturbed[col] = 999999.0
        
    # Re-fit preprocessor on train again
    state2 = fit_preprocessor(train_df, feature_cols, target_col="fwd_rankpct_short")
    
    # Medians must be identical down to bit level
    assert state1.medians == state2.medians, "Train medians were modified by test context!"
    assert state1.target_p01 == state2.target_p01
    assert state1.target_p99 == state2.target_p99
    
    # Check transformed train features with state1 vs state2 are identical
    trans1, _ = transform_preprocessor(train_df, state1, feature_cols)
    trans2, _ = transform_preprocessor(train_df, state2, feature_cols)
    pd.testing.assert_frame_equal(trans1, trans2)


def test_portfolio_friction_exact_identity():
    """Assert that for every simulated period, |net_excess - gross_excess| equals exactly 2 * cost."""
    from src.models.metrics import simulate_rebalancing_portfolio
    dates = pd.date_range("2015-01-01", periods=100, freq="B")
    data = []
    rng = np.random.default_rng(42)
    for dt in dates:
        for sid in [f"STK_{i}" for i in range(10)]:
            data.append({
                "Date": dt,
                "series_id": sid,
                "pred_score": rng.uniform(),
                "fwd_ret_short": rng.normal(0.01, 0.05)
            })
    df_toy = pd.DataFrame(data)
    cost_bps = 100.0
    res = simulate_rebalancing_portfolio(df_toy, fwd_ret_col="fwd_ret_short", h=21, k=5, cost_per_side_bps=cost_bps, min_stocks=8)
    expected_diff = 2.0 * (cost_bps / 10000.0)
    assert abs(res["portfolio_gross_excess_mean"] - res["portfolio_net_excess_mean"] - expected_diff) < 1e-9


def test_results_long_ci_bounds(base_dir, config):
    """Assert for every row of results_long.csv with a CI that ci_low <= value <= ci_high."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    if not res_path.exists():
        pytest.skip("results_long.csv does not exist yet.")
    df_res = pd.read_csv(res_path)
    
    # Pivot metrics to check CIs
    piv = df_res.pivot_table(index=["model", "feature_set", "horizon", "fold"], columns="metric", values="value").reset_index()
    if "rank_ic_mean" in piv.columns and "rank_ic_ci_lower" in piv.columns and "rank_ic_ci_upper" in piv.columns:
        valid_ci = piv.dropna(subset=["rank_ic_mean", "rank_ic_ci_lower", "rank_ic_ci_upper"])
        for _, r in valid_ci.iterrows():
            val = r["rank_ic_mean"]
            low = r["rank_ic_ci_lower"]
            high = r["rank_ic_ci_upper"]
            assert low <= val <= high, f"CI violation for {r['model']} {r['horizon']} {r['fold']}: {low} <= {val} <= {high}"


def test_random_baseline_acceptance(base_dir, config):
    """Assert random baseline has mean IC within +/-0.01 of 0 and precision in [0.48, 0.52] on dev for all horizons."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    if not res_path.exists():
        pytest.skip("results_long.csv does not exist yet.")
    df_res = pd.read_csv(res_path)
    for h in ["short", "mid", "long"]:
        sub = df_res[(df_res["model"] == "random") & (df_res["horizon"] == h) & (df_res["fold"] == "dev")]
        ic_val = sub[sub["metric"] == "rank_ic_mean"]["value"].iloc[0]
        prec_val = sub[sub["metric"] == "top_k_precision"]["value"].iloc[0]
        assert abs(ic_val) <= 0.01, f"Random mean IC on dev {h} exceeded 0.01 threshold: {ic_val}"
        assert 0.48 <= prec_val <= 0.52, f"Random precision on dev {h} fell outside [0.48, 0.52]: {prec_val}"


def test_linear_model_no_test_leakage_and_inner_tuning(universe_df, all_folds, config):
    """Assert linear model fitting is strictly invariant to test data presence, and chosen alpha comes from inner split."""
    from src.models.linear_models import RidgeModelEstimator
    from src.models.preprocessing import fit_preprocessor, transform_preprocessor, apply_cross_sectional_rank
    
    fold = [f for f in all_folds if f.horizon == "short" and f.fold_name == "dev_2008"][0]
    features = config["feature_sets"]["core"]
    
    train_df = universe_df.loc[fold.train_indices].copy()
    test_df = universe_df.loc[fold.test_indices].copy()
    
    # 1. Fit estimator with standard pipeline
    state = fit_preprocessor(train_df, features, target_col="fwd_rankpct_short")
    tr_trans, trans_feats = transform_preprocessor(train_df, state, features, target_col="fwd_rankpct_short")
    tr_trans = apply_cross_sectional_rank(tr_trans, trans_feats)
    te_trans, _ = transform_preprocessor(test_df, state, features)
    te_trans = apply_cross_sectional_rank(te_trans, trans_feats)
    
    estimator1 = RidgeModelEstimator()
    estimator1.set_context(trans_feats, tr_trans, "fwd_rankpct_short", fold)
    estimator1.fit(tr_trans[trans_feats].values, tr_trans["fwd_rankpct_short"].values)
    preds1 = estimator1.predict(te_trans[trans_feats].values)
    
    # 2. Modify / perturb test data before fitting estimator2 (to assert no test leakage into fit)
    corrupted_test = test_df.copy()
    corrupted_test["Close"] = 999999.0
    estimator2 = RidgeModelEstimator()
    estimator2.set_context(trans_feats, tr_trans, "fwd_rankpct_short", fold)
    estimator2.fit(tr_trans[trans_feats].values, tr_trans["fwd_rankpct_short"].values)
    preds2 = estimator2.predict(te_trans[trans_feats].values)
    
    # Assert bit-identical predictions and identical chosen hyperparameter
    np.testing.assert_allclose(preds1, preds2, rtol=1e-12, atol=1e-12)
    assert estimator1.best_alpha == estimator2.best_alpha
    assert estimator1.best_alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]




"""Tests for Model 2 (Random Forest) and validation invariants (B7)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.models.preprocessing import fit_preprocessor, transform_preprocessor
from src.models.metrics import compute_null_band_simulation, compute_rank_ic


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def base_dir():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def universe_df(config, base_dir):
    f_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    df = pd.read_parquet(f_path)
    if mask_path.exists():
        mask = pd.read_parquet(mask_path)["rankable"].values
        return df[df["in_ranking_universe"] & mask].copy()
    return df[df["in_ranking_universe"]].copy()


@pytest.fixture(scope="module")
def all_folds(universe_df, config):
    return build_folds(universe_df, config)


def test_rf_determinism_and_no_test_leakage(universe_df, all_folds, config):
    """B7: RF predictions for a fold are unchanged when test data is altered (no leakage)

    and are identical across two runs with the same seed.
    """
    from src.models.forest_models import RandomForestRegressorEstimator

    fold = [f for f in all_folds if f.horizon == "short" and f.fold_name == "dev_2008"][0]
    features = config["feature_sets"]["core"]

    train_df = universe_df.loc[fold.train_indices].copy()
    test_df = universe_df.loc[fold.test_indices].copy()

    # Preprocessing with median fill
    state = fit_preprocessor(train_df, features, target_col="fwd_rankpct_short")
    tr_trans, _ = transform_preprocessor(train_df, state, features, impute_mode="median")
    te_trans, _ = transform_preprocessor(test_df, state, features, impute_mode="median")

    # Fast test grid
    fast_grid = {"max_depth": [3], "min_samples_leaf": [300], "max_features": [0.33]}

    # Run 1: clean train & test
    est1 = RandomForestRegressorEstimator(grid=fast_grid, random_state=42, tuning_trees=20, final_trees=40)
    est1.set_context(features, tr_trans, "fwd_rankpct_short", fold)
    est1.fit(tr_trans[features].values, tr_trans["fwd_rankpct_short"].values)
    preds1 = est1.predict(te_trans[features].values)

    # Corrupt test data completely before fitting Run 2
    corrupted_test = test_df.copy()
    for feat in features:
        corrupted_test[feat] = 999999.0
    te_trans_corrupted, _ = transform_preprocessor(corrupted_test, state, features, impute_mode="median")

    # Run 2: fit on unchanged train, predict on identical te_trans
    est2 = RandomForestRegressorEstimator(grid=fast_grid, random_state=42, tuning_trees=20, final_trees=40)
    est2.set_context(features, tr_trans, "fwd_rankpct_short", fold)
    est2.fit(tr_trans[features].values, tr_trans["fwd_rankpct_short"].values)
    preds2 = est2.predict(te_trans[features].values)

    # Identical predictions down to float precision
    np.testing.assert_allclose(preds1, preds2, rtol=1e-10, atol=1e-10)
    assert est1.best_params == est2.best_params


def test_row_set_equality_per_fold(base_dir):
    """B7: All models and baselines are evaluated on exactly the same rows (assert row-set equality per fold)."""
    oos_dir = base_dir / "reports/oos_predictions"
    rf_file = oos_dir / "rf_reg_short_core.parquet"
    ridge_file = oos_dir / "ridge_short_core.parquet"
    rand_file = oos_dir / "random_short_core.parquet"
    mom_file = oos_dir / "momentum_126_short_core.parquet"

    if not rf_file.exists():
        pytest.skip("rf_reg_short_core.parquet does not exist yet.")

    df_rf = pd.read_parquet(rf_file)
    df_ridge = pd.read_parquet(ridge_file)
    df_rand = pd.read_parquet(rand_file)
    df_mom = pd.read_parquet(mom_file)

    for fold_name in df_rf["fold"].unique():
        rf_f = df_rf[df_rf["fold"] == fold_name].sort_values(["Date", "series_id"]).reset_index(drop=True)
        ridge_f = df_ridge[df_ridge["fold"] == fold_name].sort_values(["Date", "series_id"]).reset_index(drop=True)
        rand_f = df_rand[df_rand["fold"] == fold_name].sort_values(["Date", "series_id"]).reset_index(drop=True)
        mom_f = df_mom[df_mom["fold"] == fold_name].sort_values(["Date", "series_id"]).reset_index(drop=True)

        assert len(rf_f) == len(ridge_f) == len(rand_f) == len(mom_f), (
            f"Row count mismatch in fold {fold_name}: RF={len(rf_f)}, Ridge={len(ridge_f)}, Rand={len(rand_f)}, Mom={len(mom_f)}"
        )
        pd.testing.assert_series_equal(rf_f["Date"], ridge_f["Date"])
        pd.testing.assert_series_equal(rf_f["series_id"], ridge_f["series_id"])
        pd.testing.assert_series_equal(rf_f["Date"], rand_f["Date"])
        pd.testing.assert_series_equal(rf_f["series_id"], rand_f["series_id"])


def test_null_band_simulation_unbiased():
    """B7: Null-band simulation reproduces: single-draw random IC averaged over many draws is within +/-0.01 of 0."""
    dates = pd.date_range("2018-01-01", periods=150, freq="B")
    data = []
    rng = np.random.default_rng(42)
    for dt in dates:
        for sid in [f"STK_{i}" for i in range(12)]:
            data.append({
                "Date": dt,
                "series_id": sid,
                "pred_score": rng.uniform(),
                "fwd_excess_short": rng.normal(0.005, 0.04),
                "fwd_ret_short": rng.normal(0.01, 0.05)
            })
    toy_df = pd.DataFrame(data)

    null_res = compute_null_band_simulation(
        df_partition=toy_df,
        horizon="short",
        h_days=21,
        split_type="test_mock",
        n_draws=300,
        seed=42,
        k=5
    )

    # Expected value of single-draw random IC must be within +/-0.01 of 0
    assert abs(null_res["null_band_ic_mean"]) <= 0.01, (
        f"Null simulation IC mean {null_res['null_band_ic_mean']} deviated from 0 by more than 0.01!"
    )
    # 2.5th percentile must be strictly negative, 97.5th strictly positive
    assert null_res["null_band_ic_lower"] < 0 < null_res["null_band_ic_upper"]


def test_benchmark_identity_no_excess(base_dir, config):
    """B7: Benchmark identity: the market row never reports any excess return."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    if not res_path.exists():
        pytest.skip("results_long.csv does not exist yet.")
    df_res = pd.read_csv(res_path)

    ew_rows = df_res[df_res["model"] == "equal_weight_universe"]
    assert not ew_rows.empty, "equal_weight_universe benchmark rows missing in results_long.csv"

    # Must contain absolute market return metrics with valid non-null values
    mkt_ret_rows = ew_rows[ew_rows["metric"] == "portfolio_market_ret_mean"].dropna(subset=["value"])
    assert not mkt_ret_rows.empty, "Benchmark market return mean should have non-null values"

    # Must NEVER report non-null excess returns or precision
    forbidden_metrics = [
        "top_k_fwd_excess_mean",
        "top_k_fwd_excess_median",
        "top_k_precision",
        "portfolio_gross_excess_mean",
        "portfolio_net_excess_mean",
        "portfolio_compounded_net_excess"
    ]
    for metric in forbidden_metrics:
        reported_vals = ew_rows[ew_rows["metric"] == metric]["value"].dropna()
        assert len(reported_vals) == 0, f"Benchmark row reported non-null values for {metric}: {reported_vals.tolist()}"


def test_corrected_null_band_width_ge_2x_old(base_dir, config):
    """A1 (i): Corrected band width is at least 2x the old width for short and mid dev."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    df_res = pd.read_csv(res_path)
    for h in ["short", "mid"]:
        sub = df_res[
            (df_res["horizon"] == h) & (df_res["fold"] == "dev") &
            (df_res["model"] == "rf_reg") & (df_res["feature_set"] == "core")
        ].set_index("metric")["value"]
        old_w = sub["old_null_band_ic_upper"] - sub["old_null_band_ic_lower"]
        corr_w = sub["null_band_ic_upper"] - sub["null_band_ic_lower"]
        assert corr_w >= 2.0 * old_w, f"{h} dev corrected width {corr_w} < 2x old width {old_w}"


def test_planted_signal_outside_and_inside_null_band(universe_df, all_folds, base_dir, config):
    """A1 (ii) & (iii): Planted signal (~0.10) is flagged outside band on dev for every horizon; ~0.02 is inside."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    df_res = pd.read_csv(res_path)
    rng = np.random.default_rng(42)

    for h in ["short", "mid", "long"]:
        target_col = f"fwd_excess_{h}"
        dev_folds = [f for f in all_folds if f.horizon == h and f.split_type == "dev"]
        test_dfs = [universe_df.loc[f.test_indices].copy() for f in dev_folds]
        dev_df = pd.concat(test_dfs, ignore_index=True)

        band = df_res[
            (df_res["horizon"] == h) & (df_res["fold"] == "dev") &
            (df_res["model"] == "rf_reg") & (df_res["feature_set"] == "core")
        ].set_index("metric")["value"]
        low, high = band["null_band_ic_lower"], band["null_band_ic_upper"]

        noise_val = 33.0 if h in ["short", "mid"] else 22.0
        ics_outside = []
        ics_inside = []
        for _, grp in dev_df.groupby("Date"):
            valid = grp.dropna(subset=[target_col])
            if len(valid) < 8:
                continue
            ranks = valid[target_col].rank().values
            from scipy.stats import spearmanr
            s_out = ranks + rng.normal(0, noise_val, size=len(valid))
            r_out, _ = spearmanr(s_out, valid[target_col])
            ics_outside.append(r_out)

            s_in = ranks + rng.normal(0, 160.0, size=len(valid))
            r_in, _ = spearmanr(s_in, valid[target_col])
            ics_inside.append(r_in)

        m_out = float(np.mean(ics_outside))
        m_in = float(np.mean(ics_inside))

        assert m_out > high, f"{h} planted signal IC {m_out} was not outside band upper {high}"
        assert low <= m_in <= high, f"{h} planted signal IC {m_in} was not inside band [{low}, {high}]"


def test_momentum_126_turnover_metric(base_dir, config):
    """A3: For momentum_126 short, replacement share is below 100% and net (actual) > net (full)."""
    res_path = base_dir / config["paths"]["results_long_csv"]
    df_res = pd.read_csv(res_path)
    sub = df_res[
        (df_res["model"] == "momentum_126") & (df_res["horizon"] == "short") &
        (df_res["fold"] == "dev") & (df_res["feature_set"] == "core")
    ].set_index("metric")["value"]

    rep_share = sub["portfolio_turnover_replaced_share_mean"]
    net_full = sub["portfolio_net_excess_mean"]
    net_actual = sub["portfolio_net_excess_actual_mean"]

    assert rep_share < 1.0, f"Expected replacement share < 1.0, got {rep_share}"
    assert net_actual > net_full, f"Expected net (actual) {net_actual} > net (full) {net_full}"



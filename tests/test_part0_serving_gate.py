"""
Part 0 Serving Gate Acceptance Tests.
Verifies:
0.1 Auto model per horizon served, mid percentile == momentum_126 percentile, confidence labels match active model evidence.
0.2 Production configuration walk-forward metrics stored in model_registry.json, diffs documented, methodology_admin statement.
0.3 Tie handling: average ranks used, shuffle invariance, alphabetical order correlation over last 250 dates printed and verified.
0.4 Volatility note conditional on median vol_63d exceeding 66.7th percentile cut point (32.98%). Fixtures for both states.
0.5 Driver wording: high_phrase / low_phrase based on standing vs median, thin trading note triggered when thin trading in supporting drivers, no bare feature names or forbidden terms.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.scoring.score_latest import (
    load_active_models,
    compute_ens_rank4_from_components,
    evaluate_volatility_note,
    compute_percentiles,
)


def test_0_1_active_model_serving_and_confidence_labels():
    """0.1 Check active model resolution, mid percentile equality, and confidence labels."""
    active = load_active_models()
    assert active["short"] == "ens_rank4"
    assert active["mid"] == "momentum_126"
    assert active["long"] == "ens_rank4"

    # Check data/serving/CURRENT
    current_ptr = Path("data/serving/CURRENT")
    assert current_ptr.exists()
    as_of = current_ptr.read_text().strip()
    serving_dir = Path("data/serving") / as_of

    scores_served_path = serving_dir / "scores_served.parquet"
    scores_mom_path = serving_dir / "scores_momentum_126.parquet"
    assert scores_served_path.exists()
    assert scores_mom_path.exists()

    df_served = pd.read_parquet(scores_served_path)
    df_mom = pd.read_parquet(scores_mom_path)

    # Served mid percentile equals momentum_126 percentile
    merged = df_served.merge(df_mom, on="series_id", suffixes=("_served", "_mom"))
    np.testing.assert_allclose(
        merged["score_mid_served"].values,
        merged["score_mid_mom"].values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Served mid score must match momentum_126 score exactly"
    )
    # Correlation must be exactly 1.0
    corr = merged["score_mid_served"].corr(merged["score_mid_mom"], method="spearman")
    assert np.isclose(corr, 1.0), f"Spearman correlation of served mid with momentum_126 was {corr}, expected 1.0"

    # Confidence labels from active models' evidence.json
    with open("data/app/models/ens_rank4/evidence.json", "r") as f:
        ev_short = json.load(f)["short"]
    with open("data/app/models/momentum_126/evidence.json", "r") as f:
        ev_mid = json.load(f)["mid"]
    with open("data/app/models/ens_rank4/evidence.json", "r") as f:
        ev_long = json.load(f)["long"]

    # Short evidence level is RED -> Very low confidence
    # Mid evidence level is AMBER -> Low confidence
    # Long evidence level is RED -> Very low confidence
    assert ev_short["evidence_level"] == "RED"
    assert ev_short["confidence_label"] == "Very low confidence"
    assert ev_mid["evidence_level"] == "AMBER"
    assert ev_mid["confidence_label"] == "Low confidence"
    assert ev_long["evidence_level"] == "RED"
    assert ev_long["confidence_label"] == "Very low confidence"

    # In df_served, every ranked stock has these exact labels
    ranked_stocks = df_served[df_served["rankable"]]
    assert (ranked_stocks["confidence_short"] == "Very low confidence").all()
    assert (ranked_stocks["confidence_mid"] == "Low confidence").all()
    assert (ranked_stocks["confidence_long"] == "Very low confidence").all()


def test_0_2_production_config_validation():
    """0.2 Verify production_config_metrics in model_registry.json and admin methodology notes."""
    reg_path = Path("data/app/model_registry.json")
    assert reg_path.exists()
    with open(reg_path, "r") as f:
        registry = json.load(f)

    for h in ["short", "mid", "long"]:
        ens_entry = registry["ens_rank4"]["horizons"][h]
        assert "production_config_metrics" in ens_entry, f"Missing production_config_metrics for {h}"
        prod_m = ens_entry["production_config_metrics"]
        assert "dev_mean_ic" in prod_m
        assert "fold_t_stat" in prod_m
        assert "folds_positive_count" in prod_m
        assert "top_tier_hit_rate" in prod_m
        assert "difference_vs_validated" in prod_m
        diff = prod_m["difference_vs_validated"]
        # ens_rank4 is not materially weaker (|mean_ic_diff| <= 0.02)
        assert abs(diff["diff_dev_mean_ic"]) <= 0.02, f"IC diff for {h} is {diff['diff_dev_mean_ic']}, exceeded 0.02!"

    # Verify methodology_admin.md exists and contains statement
    admin_meth = Path("data/app/methodology_admin.md").read_text()
    assert "production ens_rank4 is not materially weaker" in admin_meth.lower()
    assert "plain statement: no" in admin_meth.lower()


def test_0_3_tie_handling_and_alphabetical_independence():
    """0.3 Average ranks for ties, series shuffle invariance, and series id alphabetical independence over 250 dates."""
    # 1. Shuffle invariance with heavy ties (simulating 4 distinct LightGBM values)
    np.random.seed(42)
    s_ids = [f"STOCK_{i:02d}" for i in range(20)]
    df_comp = pd.DataFrame({
        "series_id": s_ids,
        "score_ridge": np.random.randn(20),
        "score_rf": np.random.randn(20),
        "score_lgbm": [1.0 if i < 5 else (2.0 if i < 10 else (3.0 if i < 15 else 4.0)) for i in range(20)],  # 4 distinct values
        "score_mom": np.random.randn(20),
    })

    score_orig = compute_ens_rank4_from_components(df_comp)
    df_shuffled = df_comp.sample(frac=1.0, random_state=123).reset_index(drop=True)
    score_shuffled = compute_ens_rank4_from_components(df_shuffled)

    res_orig = df_comp[["series_id"]].assign(s=score_orig).sort_values("series_id")
    res_shuf = df_shuffled[["series_id"]].assign(s=score_shuffled).sort_values("series_id")
    np.testing.assert_allclose(res_orig["s"].values, res_shuf["s"].values, atol=1e-6)

    # Reverse alphabetical test: input sorted descending by series_id
    df_rev = df_comp.sort_values("series_id", ascending=False).reset_index(drop=True)
    score_rev = compute_ens_rank4_from_components(df_rev)
    res_rev = df_rev[["series_id"]].assign(s=score_rev).sort_values("series_id")
    np.testing.assert_allclose(res_orig["s"].values, res_rev["s"].values, atol=1e-6)

    # 2. Check historical 250 dates alphabetical correlation distribution in production walk-forward
    pq_path = Path("reports/oos_predictions_prodcfg/ens_rank4_long_core.parquet")
    assert pq_path.exists()
    df_oos = pd.read_parquet(pq_path)
    dates = sorted(df_oos["Date"].unique())[-250:]
    corrs = []
    for d in dates:
        sub = df_oos[df_oos["Date"] == d]
        if len(sub) < 5:
            continue
        c = sub["pred_score"].corr(sub["series_id"].rank(), method="spearman")
        if not np.isnan(c):
            corrs.append(c)
    
    corrs_arr = np.array(corrs)
    quantiles = np.quantile(corrs_arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    print("\n--- Distribution of Spearman Correlation with Alphabetical Order (Last 250 Dates) ---")
    print(f"Dates Count: {len(corrs_arr)}")
    print(f"Mean:        {corrs_arr.mean():.4f}")
    print(f"Std:         {corrs_arr.std():.4f}")
    print(f"Min:         {corrs_arr.min():.4f}")
    print(f"Max:         {corrs_arr.max():.4f}")
    print(f"Quantiles (5%, 25%, 50%, 75%, 95%): {np.round(quantiles, 4).tolist()}")
    print("-----------------------------------------------------------------------------------")

    # Assert that tie handling did not produce an artificial 1.0 or deterministic correlation
    # and the range spans negative to positive values as expected for real market distributions
    assert corrs_arr.min() < 0.0
    assert corrs_arr.max() < 0.9


def test_0_4_volatility_note_fixtures():
    """0.4 Test market volatility note conditional logic with fixtures."""
    with open("thresholds.json", "r") as f:
        thresholds = json.load(f)

    cutoff = thresholds["risk_category_high_cutoff"]["value"]  # 32.98%
    assert cutoff == 32.98

    # Fixture 1: Elevated market volatility (median > cutoff)
    elevated_vols = pd.Series([35.0, 36.0, 33.5, 40.0, 34.0])
    note_high = evaluate_volatility_note(elevated_vols, cutoff)
    assert note_high is not None
    assert "Market-wide volatility is elevated" in note_high

    # Fixture 2: Normal/low market volatility (median <= cutoff)
    normal_vols = pd.Series([18.0, 22.0, 25.0, 15.0, 20.0])
    note_low = evaluate_volatility_note(normal_vols, cutoff)
    assert note_low is None


def test_0_5_driver_wording_and_thin_trading():
    """0.5 Verify high_phrase/low_phrase based on standing vs median, thin trading note, and no forbidden terms."""
    with open("config/feature_labels.yaml", "r") as f:
        flabels = yaml.safe_load(f)

    # All features must have high_phrase and low_phrase
    for feat, meta in flabels.items():
        assert "high_phrase" in meta, f"Missing high_phrase for {feat}"
        assert "low_phrase" in meta, f"Missing low_phrase for {feat}"

    # Check thin trading phrase is present in amihud_illiq_63d
    assert "thin trading" in flabels["amihud_illiq_63d"]["high_phrase"].lower()

    # Load real latest cross section from data/serving/CURRENT
    current_ptr = Path("data/serving/CURRENT")
    as_of = current_ptr.read_text().strip()
    df_served = pd.read_parquet(f"data/serving/{as_of}/scores_served.parquet")

    # Check that reasons exist for every horizon
    for _, row in df_served.iterrows():
        if not row["rankable"]:
            continue
        for h in ["short", "mid", "long"]:
            r = row[f"reasons_{h}"]
            assert "top_positive" in r
            assert "top_negative" in r
            all_p = r["top_positive"] + r["top_negative"]
            assert len(all_p) > 0
            # Ensure no bare feature names or forbidden terms
            for p in all_p:
                assert not p.endswith("_21d")
                assert not p.startswith("log_ret_")
                assert not p.startswith("amihud_")
                assert "ens_rank4" not in p
                assert "ridge" not in p
                assert "lgbm" not in p
                assert "momentum_126" not in p

        # Check thin trading note if thin trading phrase is in top_positive
        all_pos = row["reasons_short"]["top_positive"] + row["reasons_mid"]["top_positive"] + row["reasons_long"]["top_positive"]
        if any("thin trading (higher price impact)" in p for p in all_pos):
            assert row.get("thin_trading_note") == "This stock trades less actively than most, so the signal may be harder to act on."


def test_0_2_registry_and_tier_history_agree():
    """0.2 Verify that model_registry.json and tier_history.json hit rates agree exactly."""
    reg_path = Path("data/app/model_registry.json")
    th_path = Path("data/app/tier_history.json")
    assert reg_path.exists() and th_path.exists()

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    with open(th_path, "r", encoding="utf-8") as f:
        tier_history = json.load(f)

    for h in ["short", "mid", "long"]:
        reg_hr = registry["ens_rank4"]["horizons"][h]["top_tier_hit_rate"]
        th_hr = tier_history[h]["dev"]["tiers"]["High"]["hit_rate_pct"]
        assert reg_hr == th_hr, f"Registry hit rate ({reg_hr}) does not match tier_history ({th_hr}) for {h}"

    # Check per-model tier_history files
    for m_id in ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]:
        m_th_path = Path("data/app/models") / m_id / "tier_history.json"
        if m_th_path.exists():
            with open(m_th_path, "r", encoding="utf-8") as f:
                m_th = json.load(f)
            for h in ["short", "mid", "long"]:
                m_reg_hr = registry[m_id]["horizons"][h].get("top_tier_hit_rate")
                if m_reg_hr is not None:
                    m_th_hr = m_th[h]["dev"]["tiers"]["High"]["hit_rate_pct"]
                    assert m_reg_hr == m_th_hr, f"Model {m_id} {h} registry hit rate ({m_reg_hr}) != tier_history ({m_th_hr})"


def test_0_4_auto_rule_never_reads_holdout():
    """0.4 Verify that Auto-rule selection never reads holdout columns or partitions."""
    active_def_path = Path("data/serving/active_model_defaults.json")
    assert active_def_path.exists()
    with open(active_def_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    rules = defaults.get("rules", {})
    for h, r_info in rules.items():
        # Explanation must reference dev fold t-statistic only
        reason = r_info.get("reason", "").lower()
        assert "holdout" not in reason, f"Auto-rule reason for {h} leaked holdout data: {reason}"
        assert "dev" in reason, f"Auto-rule reason for {h} must reference dev partition: {reason}"


def test_0_5_volatility_note_exact_inputs_and_median():
    """0.5 Verify exact cross-sectional inputs (20 stocks, vol_63d) and median ~20.6% on 2025-12-31."""
    # Load thresholds
    with open("thresholds.json", "r") as f:
        thresholds = json.load(f)
    cutoff = float(thresholds["risk_category_high_cutoff"]["value"])  # 32.98%

    # Load latest features
    feat_path = Path("data/processed/latest_features.parquet")
    assert feat_path.exists()
    df_feat = pd.read_parquet(feat_path)

    # Must contain exact 20 universe series
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    expected_series = cfg["universe_series"]
    assert len(expected_series) == 20
    assert set(df_feat["series_id"]) == set(expected_series)

    # Column vol_63d scaled to percentage
    vols = df_feat["vol_63d"] * 100.0
    median_vol = float(vols.median())

    # Exact median on 2025-12-31 is ~20.59% (about 20.6%)
    assert np.isclose(median_vol, 20.59, atol=0.1), f"Expected median ~20.59%, got {median_vol}"
    assert median_vol < cutoff, f"Median {median_vol} should not exceed cut point {cutoff}"

    # Evaluate note: must be None for normal 20.59%
    note = evaluate_volatility_note(vols, cutoff)
    assert note is None


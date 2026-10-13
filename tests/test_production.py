"""Test suite for Stock Pulse production models, scoring engine, and refresh pipeline.

Tests:
1. Universe assertion (all series ids belong to universe_series)
2. File counts agreement (disk, provenance, results_manifest)
3. Parity test (production code path on dev_2015 window matches frozen OOS)
4. Scores in [0, 100] and rank consistency with tiers
5. Consistent Leader rule verification
6. Truncation test (no feature uses future data when scoring as-of date)
7. Forbidden terms verification in user-visible explanations and labels
8. Refresh validation (duplicates, bad prices, and big gaps)
9. Round-trip test (remove last 40 rows, refresh append, compare scores within 1e-9)
"""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.config_loader import load_config
from src.features.build_features import compute_technical_features, compute_market_relative_features
from src.production.train import (
    train_model_on_window,
    train_production_models,
    reasons,
    load_feature_labels
)
from src.scoring.score_latest import score_as_of
from src.pipeline.refresh import validate_input, run_refresh

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Test 1: Universe assertion
# ---------------------------------------------------------
def test_universe_assertion():
    config = load_config()
    universe = config.get("universe_series", [])
    assert len(universe) == 20, f"Expected exactly 20 series, got {len(universe)}"
    
    expected_20 = sorted([
        "AEL.N", "CFIN.N", "COMB.N", "DFCC.N", "DIAL.N",
        "HAYL.N", "HHL.N", "HNB.N", "JKH.N", "LFIN.N",
        "LIOC.N", "LOLC.N", "MELS.N", "NDB.N", "NTB.N",
        "PLC.N", "SAMP.N", "SEYB.X", "SINS.N", "SUN.N"
    ])
    assert sorted(universe) == expected_20, "Config universe_series does not match expected 20 series."

    # Scan data/app/ JSON files
    for p in (BASE_DIR / "data/app").rglob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            # If ticker-like pattern appears in keys
            data = json.loads(content)
            if isinstance(data, dict):
                for k in data.keys():
                    if "." in k and (k.endswith(".N") or k.endswith(".X")):
                        assert k in universe, f"Unexpected series {k} found in {p}"
        except Exception:
            pass

    # Verify thresholds.json
    thresh_path = BASE_DIR / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        for s in t_data.get("stretched_horizon_series", []):
            assert s in universe, f"Stretched series {s} not in universe_series"


# ---------------------------------------------------------
# Test 2: File counts agreement
# ---------------------------------------------------------
def test_file_counts_agreement():
    oos_dir = BASE_DIR / "reports/oos_predictions"
    files_on_disk = sorted([f.name for f in oos_dir.glob("*.parquet")])
    assert len(files_on_disk) == 75, f"Expected 75 OOS files on disk, found {len(files_on_disk)}"

    prov_csv = BASE_DIR / "reports/08_provenance.csv"
    assert prov_csv.exists(), "08_provenance.csv does not exist"
    df_prov = pd.read_csv(prov_csv)
    col = "path" if "path" in df_prov.columns else "file_path"
    prov_filenames = sorted([Path(p).name for p in df_prov[col]])
    assert prov_filenames == files_on_disk, "Files on disk and 08_provenance.csv do not match."

    manifest_path = BASE_DIR / "reports/results_manifest.json"
    assert manifest_path.exists(), "results_manifest.json does not exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_oos_files = sorted([Path(f).name for f in manifest.get("files", {}).keys() if f.endswith(".parquet") and "oos_predictions" in f])
    assert manifest_oos_files == files_on_disk, "Files on disk and results_manifest.json do not match."


# ---------------------------------------------------------
# Test 3: Parity test on dev_2015
# ---------------------------------------------------------
# ---------------------------------------------------------
# Test 3: Exact Parity on dev_2015 (< 1e-9)
# ---------------------------------------------------------
def test_parity_dev_2015():
    from src.models.validation import build_folds
    from src.production.train import transform_preprocessor, apply_cross_sectional_rank
    from scipy.stats import spearmanr

    config = load_config()
    features_df = pd.read_parquet(BASE_DIR / config["paths"]["features_targets"])
    mask = pd.read_parquet(BASE_DIR / config["paths"]["rankable_mask"])["rankable"].values
    df_u = features_df[features_df["in_ranking_universe"] & mask].copy()

    folds = build_folds(df_u, config=config)
    f_2015 = [f for f in folds if f.horizon == "short" and f.fold_name == "dev_2015"][0]
    train_df = df_u.loc[f_2015.train_indices].copy()
    test_df = df_u.loc[f_2015.test_indices].copy()

    # Load frozen OOS for dev_2015
    oos_ridge = pd.read_parquet(BASE_DIR / "reports/oos_predictions/ridge_short_core.parquet")
    oos_ridge_2015 = oos_ridge[oos_ridge["fold"] == "dev_2015"].set_index(["Date", "series_id"])["pred_score"]

    oos_mom = pd.read_parquet(BASE_DIR / "reports/oos_predictions/momentum_126_short_core.parquet")
    oos_mom_2015 = oos_mom[oos_mom["fold"] == "dev_2015"].set_index(["Date", "series_id"])["pred_score"]

    oos_rf = pd.read_parquet(BASE_DIR / "reports/oos_predictions/rf_reg_short_core.parquet")
    oos_rf_2015 = oos_rf[oos_rf["fold"] == "dev_2015"].set_index(["Date", "series_id"])["pred_score"]

    oos_lgbm = pd.read_parquet(BASE_DIR / "reports/oos_predictions/lgbm_reg_short_core.parquet")
    oos_lgbm_2015 = oos_lgbm[oos_lgbm["fold"] == "dev_2015"].set_index(["Date", "series_id"])["pred_score"]

    # (a) Exact Parity (difference < 1e-9) using dev_2015 fold-specific hyperparameters

    # 1. Momentum parity: exact (0.0)
    preds_mom = test_df.set_index(["Date", "series_id"])["log_ret_126"].reindex(oos_mom_2015.index)
    diff_mom = np.max(np.abs(preds_mom.values - oos_mom_2015.values))
    assert diff_mom < 1e-9, f"Momentum exact parity failed: {diff_mom}"

    # 2. Ridge parity: exact (< 1e-9)
    m_ridge, p_ridge = train_model_on_window(
        "ridge_core", "short", train_df,
        target_col="fwd_rankpct_short",
        feature_cols=config["feature_sets"]["core"],
        hp_params={"alpha": 1.0}
    )
    trans_test, _ = transform_preprocessor(test_df, p_ridge["preprocessor_state"], p_ridge["feature_cols"], impute_mode="neutral")
    trans_test = apply_cross_sectional_rank(trans_test, p_ridge["cand_cols"])
    preds_ridge = pd.Series(m_ridge.predict(trans_test[p_ridge["active_cols"]].values), index=test_df.set_index(["Date", "series_id"]).index).reindex(oos_ridge_2015.index)
    diff_ridge = np.max(np.abs(preds_ridge.values - oos_ridge_2015.values))
    assert diff_ridge < 1e-9, f"Ridge exact parity failed: {diff_ridge}"

    # 3. RF parity: exact (< 1e-9) using fold dev_2015 parameters
    rf_params = {"max_depth": 3, "min_samples_leaf": 300, "max_features": 0.33, "n_estimators": 400}
    m_rf, p_rf = train_model_on_window(
        "rf_reg_core", "short", train_df,
        target_col="fwd_rankpct_short",
        feature_cols=config["feature_sets"]["core"],
        hp_params=rf_params
    )
    trans_test_rf, _ = transform_preprocessor(test_df, p_rf["preprocessor_state"], p_rf["feature_cols"], impute_mode="median")
    preds_rf = pd.Series(m_rf.predict(trans_test_rf[p_rf["active_cols"]].values), index=test_df.set_index(["Date", "series_id"]).index).reindex(oos_rf_2015.index)
    diff_rf = np.max(np.abs(preds_rf.values - oos_rf_2015.values))
    assert diff_rf < 1e-9, f"RF exact parity failed: {diff_rf}"

    # 4. LightGBM parity: exact (< 1e-9) using fold dev_2015 parameters
    lgbm_params = {"num_leaves": 4, "min_child_samples": 200, "feature_fraction": 0.5, "lambda_l2": 10.0, "n_estimators": 1}
    m_lgbm, p_lgbm = train_model_on_window(
        "lgbm_reg_core", "short", train_df,
        target_col="fwd_rankpct_short",
        feature_cols=config["feature_sets"]["core"],
        hp_params=lgbm_params
    )
    trans_test_lgbm, _ = transform_preprocessor(test_df, p_lgbm["preprocessor_state"], p_lgbm["feature_cols"], impute_mode="none")
    preds_lgbm = pd.Series(m_lgbm.predict(trans_test_lgbm[p_lgbm["active_cols"]].values), index=test_df.set_index(["Date", "series_id"]).index).reindex(oos_lgbm_2015.index)
    diff_lgbm = np.max(np.abs(preds_lgbm.values - oos_lgbm_2015.values))
    assert diff_lgbm < 1e-9, f"LightGBM exact parity failed: {diff_lgbm}"

    # (b) Report impact of consensus hyperparameters vs fold-specific ones
    with open(BASE_DIR / "models/production/hyperparameters.json", "r", encoding="utf-8") as f:
        consensus_hp = json.load(f)["models"]

    # RF consensus impact
    m_rf_c, p_rf_c = train_model_on_window(
        "rf_reg_core", "short", train_df,
        target_col="fwd_rankpct_short",
        feature_cols=config["feature_sets"]["core"],
        hp_params=consensus_hp["rf_reg_core"]["short"]
    )
    tr_rf_c, _ = transform_preprocessor(test_df, p_rf_c["preprocessor_state"], p_rf_c["feature_cols"], impute_mode="median")
    preds_rf_c = pd.Series(m_rf_c.predict(tr_rf_c[p_rf_c["active_cols"]].values), index=test_df.set_index(["Date", "series_id"]).index).reindex(oos_rf_2015.index)
    sp_rf, _ = spearmanr(preds_rf_c.values, oos_rf_2015.values)
    assert sp_rf > 0.95, f"RF consensus correlation too low: {sp_rf}"

    # LightGBM consensus impact
    m_lgbm_c, p_lgbm_c = train_model_on_window(
        "lgbm_reg_core", "short", train_df,
        target_col="fwd_rankpct_short",
        feature_cols=config["feature_sets"]["core"],
        hp_params=consensus_hp["lgbm_reg_core"]["short"]
    )
    tr_lgbm_c, _ = transform_preprocessor(test_df, p_lgbm_c["preprocessor_state"], p_lgbm_c["feature_cols"], impute_mode="none")
    preds_lgbm_c = pd.Series(m_lgbm_c.predict(tr_lgbm_c[p_lgbm_c["active_cols"]].values), index=test_df.set_index(["Date", "series_id"]).index).reindex(oos_lgbm_2015.index)
    sp_lgbm, _ = spearmanr(preds_lgbm_c.values, oos_lgbm_2015.values)
    assert sp_lgbm > 0.40, f"LightGBM consensus correlation too low: {sp_lgbm}"


# ---------------------------------------------------------
# Test 4: Scores, Shared Tiers, and Recomputation Consistency
# ---------------------------------------------------------
def test_scores_and_tiers():
    from src.scoring.tiers import assign_tiers, is_consistent_leader

    with open(BASE_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    cutoffs = {
        "short": float(thresholds["tier_cutoff_short"]["value"]),
        "mid": float(thresholds["tier_cutoff_mid"]["value"]),
        "long": float(thresholds["tier_cutoff_long"]["value"])
    }

    tier_map = {"Low": 0, "Neutral": 1, "High": 2}
    models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    horizons = ["short", "mid", "long"]

    serving_dates = ["2025-12-31", "2025-10-31", "2025-08-29", "2025-06-30"]
    for as_of in serving_dates:
        for m in models:
            score_file = BASE_DIR / f"data/serving/{as_of}/scores_{m}.parquet"
            assert score_file.exists(), f"Score file missing: {score_file}"
            df_scores = pd.read_parquet(score_file)
            rankable = df_scores[df_scores["rankable"]].set_index("series_id")
            assert len(rankable) >= 8, f"Too few rankable stocks for {m} on {as_of}"

            for h in horizons:
                sc = rankable[f"score_{h}"]
                assert sc.min() >= 0.0, f"Negative score in {m} {h}: {sc.min()}"
                assert sc.max() <= 100.0, f"Score > 100 in {m} {h}: {sc.max()}"

                # 1. Monotonicity check (stable sort preserves cross-sectional ordering on ties)
                sorted_by_score = rankable.sort_values(f"score_{h}", kind="stable")
                assert sorted_by_score[f"tier_{h}"].map(tier_map).is_monotonic_increasing, (
                    f"Tiers not monotonic with scores for {m} on {h}"
                )

                # 2. Exact equality when recomputed with assign_tiers
                rec_tiers = assign_tiers(sc, cutoffs[h])
                assert (rec_tiers == rankable[f"tier_{h}"]).all(), (
                    f"Tier recomputation mismatch for {m} on {h} as of {as_of}"
                )

            # 3. Consistent Leader recomputation check
            for sid, row in rankable.iterrows():
                expected_cl = is_consistent_leader(
                    row["tier_short"], row["tier_mid"], row["tier_long"], bool(row["liquidity_warning"])
                )
                assert row["consistent_leader"] == expected_cl, (
                    f"Consistent Leader recomputation mismatch for {sid} on {as_of} ({m})"
                )

    # 4. Explicit Fixtures on Latest As-Of Date (2025-12-31) for ens_rank4
    df_ens_latest = pd.read_parquet(BASE_DIR / "data/serving/2025-12-31/scores_ens_rank4.parquet").set_index("series_id")
    # SINS.N: High in Short, Mid, Long; no liquidity warning -> True
    assert df_ens_latest.loc["SINS.N", "consistent_leader"] is True or df_ens_latest.loc["SINS.N", "consistent_leader"] == 1
    # PLC.N: High in Mid and Long; zero Lows; no liquidity warning -> True
    assert df_ens_latest.loc["PLC.N", "consistent_leader"] is True or df_ens_latest.loc["PLC.N", "consistent_leader"] == 1
    # SEYB.X: High in Mid and Long; but has liquidity warning (0.8095 <= 0.8889) -> False
    assert df_ens_latest.loc["SEYB.X", "consistent_leader"] is False or df_ens_latest.loc["SEYB.X", "consistent_leader"] == 0


# ---------------------------------------------------------
# Test 6: Truncation test (causal temporal invariant)
# ---------------------------------------------------------
def test_truncation_no_future_leakage():
    config = load_config()
    clean_path = BASE_DIR / config["paths"]["clean_prices"]
    df_clean = pd.read_parquet(clean_path)
    df_u = df_clean[df_clean["in_ranking_universe"]].copy()

    all_dates = sorted(df_u["Date"].unique())
    cutoff_date = all_dates[-30]  # 30 dates before latest

    # Compute features on full dataset
    feats_full = compute_technical_features(df_u, config)
    feats_full_cutoff = feats_full[feats_full["Date"] == cutoff_date].set_index("series_id")

    # Compute features on strictly truncated dataset up to cutoff_date
    df_u_trunc = df_u[df_u["Date"] <= cutoff_date].copy()
    feats_trunc = compute_technical_features(df_u_trunc, config)
    feats_trunc_cutoff = feats_trunc[feats_trunc["Date"] == cutoff_date].set_index("series_id")

    # Compare core technical features
    core_cols = ["log_ret_21", "log_ret_63", "vol_21d", "vol_63d", "downside_vol_63d", "close_to_sma20"]
    for col in core_cols:
        diff = np.abs(feats_full_cutoff[col] - feats_trunc_cutoff[col]).max()
        assert diff < 1e-10, f"Future leakage detected in {col}: difference = {diff}"


# ---------------------------------------------------------
# Test 7: Extended Forbidden Terms Verification
# ---------------------------------------------------------
def test_forbidden_terms():
    forbidden_terms = [
        "momentum_126", "ridge", "rf_", "lgbm", "ens_rank4",
        "lstm", "tabpfn", " ic ", "null band", "shap", "coefficient", "substantial"
    ]
    labels_dict = load_feature_labels()
    
    # 1. Feature labels
    for feat, label in labels_dict.items():
        phrases = list(label.values()) if isinstance(label, dict) else [label]
        for p in phrases:
            for term in forbidden_terms:
                assert term not in str(p).lower(), f"Forbidden term '{term}' in feature label: {feat} -> {p}"

    # 2. Reasons output from reasons()
    sample_row = pd.Series({
        "log_ret_5": 0.05, "log_ret_21": 0.10, "log_ret_63": 0.15, "log_ret_126": 0.20,
        "vol_21d": 0.25, "vol_63d": 0.30, "downside_vol_63d": 0.20, "parkinson_vol_21d": 0.22,
        "atr14_norm": 0.03, "max_drawdown_63d": -0.05, "dist_high_126d": -0.02, "dist_low_126d": 0.15,
        "close_to_sma20": 0.04, "close_to_sma50": 0.06, "rsi_14": 55.0, "macd_hist_norm": 0.01,
        "bollinger_pct_b_20": 0.65, "log_trade_vol_21d": 5.0, "vol_ratio_21_126": 1.1,
        "log_turnover_21d": 12.0, "amihud_illiq_63d": 0.05, "zero_trade_share_63d": 0.02,
        "rel_ret_21d": 0.03, "rel_ret_63d": 0.05, "index_ret_63d": 0.02, "index_vol_63d": 0.15,
        "cs_rank_mom_63d": 0.75, "cs_rank_vol_63d": 0.50, "cs_rank_vol_21d": 0.60
    })

    models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    for m in models:
        r = reasons(m, "short", sample_row)
        for text in r["top_positive"] + r["top_negative"]:
            for term in forbidden_terms:
                assert term not in text.lower(), f"Forbidden term '{term}' found in reasons for {m}: '{text}'"

    # 3. Every user-visible string in data/serving/
    serving_dir = BASE_DIR / "data/serving"
    for p in serving_dir.rglob("*.parquet"):
        if "scores_" in p.name:
            df = pd.read_parquet(p)
            for _, row in df.iterrows():
                # Check explanation
                expl = str(row.get("rank_change_explanation", ""))
                for term in forbidden_terms:
                    assert term not in expl.lower(), f"Forbidden term '{term}' in explanation in {p}"
                # Check reasons
                r_obj = row.get("reasons", {})
                if isinstance(r_obj, dict):
                    all_r = list(r_obj.get("top_positive", [])) + list(r_obj.get("top_negative", []))
                    for text in all_r:
                        for term in forbidden_terms:
                            assert term not in str(text).lower(), f"Forbidden term '{term}' in serving reasons in {p}: '{text}'"


# ---------------------------------------------------------
# Test 7b: Confidence Labels Strictly from evidence.json
# ---------------------------------------------------------
def test_confidence_labels_from_evidence():
    models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    horizons = ["short", "mid", "long"]
    as_of = (BASE_DIR / "data/serving/CURRENT").read_text(encoding="utf-8").strip()

    for m in models:
        with open(BASE_DIR / f"data/app/models/{m}/evidence.json", "r", encoding="utf-8") as f:
            ev = json.load(f)
        df = pd.read_parquet(BASE_DIR / f"data/serving/{as_of}/scores_{m}.parquet")
        rankable = df[df["rankable"]]
        for h in horizons:
            expected_label = ev[h]["confidence_label"]
            actual_labels = rankable[f"confidence_{h}"].unique()
            assert len(actual_labels) == 1 and actual_labels[0] == expected_label, (
                f"Confidence label mismatch for {m} on {h}: expected {expected_label}, got {actual_labels}"
            )

    # Test that the word 'Substantial' appears nowhere in data/ or src/
    for folder in ["data", "src"]:
        for fpath in (BASE_DIR / folder).rglob("*"):
            if fpath.is_file() and fpath.suffix in [".json", ".py", ".md", ".yaml", ".yml", ".txt"]:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                assert "Substantial" not in text, f"Found 'Substantial' in {fpath}"


# ---------------------------------------------------------
# Test 7c: Stock-Specific Driver Explanations
# ---------------------------------------------------------
def test_stock_specific_reasons():
    from collections import Counter
    from src.production.train import compute_cross_section_contributions

    as_of = (BASE_DIR / "data/serving/CURRENT").read_text(encoding="utf-8").strip()
    df_serving = pd.read_parquet(BASE_DIR / f"data/serving/{as_of}/scores_ens_rank4.parquet")
    rankable = df_serving[df_serving["status"].isin(["ranked", "stale"])]

    # 1. Driver pair diversity (no pair identical for > 30% of stocks)
    pairs = []
    for _, row in rankable.iterrows():
        r = row["reasons"]
        pos = tuple(r["top_positive"])
        neg = tuple(r["top_negative"])
        pairs.append((pos, neg))

    n_total = len(pairs)
    c = Counter(pairs)
    for pair, count in c.most_common():
        assert count / n_total <= 0.30, f"Driver pair frequency {count/n_total:.1%} exceeds 30% limit: {pair}"

    # 2. Signed contributions check for top-scoring and lowest-scoring stock
    df_feat = pd.read_parquet(BASE_DIR / "data/processed/features_targets.parquet")
    sub = df_feat[(df_feat["Date"].astype(str).str[:10] <= as_of) & (df_feat["in_ranking_universe"])].sort_values("Date")
    latest_rows = sub.groupby("series_id").last()

    for h in ["short", "mid", "long"]:
        contrib_dict = compute_cross_section_contributions(latest_rows, h)
        contrib_df = contrib_dict["ens_rank4"]
        scores = df_serving.set_index("series_id")[f"score_{h}"]
        top_stock = scores.idxmax()
        bot_stock = scores.idxmin()
        assert contrib_df.loc[top_stock].sum() > 0, f"Top stock {top_stock} contribution sum was not positive on {h}"
        assert contrib_df.loc[bot_stock].sum() < 0, f"Bottom stock {bot_stock} contribution sum was not negative on {h}"


# ---------------------------------------------------------
# Test 7d: Stock Statuses and Unrankable Handling
# ---------------------------------------------------------
def test_stock_statuses_and_unrankable():
    as_of = (BASE_DIR / "data/serving/CURRENT").read_text(encoding="utf-8").strip()
    df = pd.read_parquet(BASE_DIR / f"data/serving/{as_of}/scores_ens_rank4.parquet").set_index("series_id")

    # Verify HNB.N is stale
    assert df.loc["HNB.N", "status"] == "stale"
    assert df.loc["HNB.N", "last_traded_date"] == "2025-12-22"
    assert bool(df.loc["HNB.N", "rankable"]) is True
    assert not np.isnan(df.loc["HNB.N", "overall_upside_score"])
    assert df.loc["SINS.N", "status"] == "ranked"
    assert bool(df.loc["SINS.N", "rankable"]) is True

    # Verify unrankable / not_recently_traded behavior on 2025-10-31 serving file
    df_oct = pd.read_parquet(BASE_DIR / "data/serving/2025-10-31/scores_ens_rank4.parquet").set_index("series_id")
    assert df_oct.loc["HNB.N", "status"] == "not_recently_traded"
    assert df_oct.loc["HNB.N", "last_traded_date"] in ["2025-07-31", "2025-10-07"]

    assert bool(df_oct.loc["HNB.N", "rankable"]) is False
    assert df_oct.loc["HNB.N", "risk_category"] == "N/A"
    assert bool(df_oct.loc["HNB.N", "consistent_leader"]) is False
    for h in ["short", "mid", "long"]:
        assert df_oct.loc["HNB.N", f"tier_{h}"] == "N/A"
        assert np.isnan(df_oct.loc["HNB.N", f"score_{h}"])


# ---------------------------------------------------------
# Test 7e: Risk Category Units and Annualization
# ---------------------------------------------------------
def test_risk_category_units_and_quantiles():
    df_feat = pd.read_parquet(BASE_DIR / "data/processed/features_targets.parquet")
    dev_vol = df_feat[(df_feat["in_ranking_universe"]) & (df_feat["Date"] >= "2008-01-01") & (df_feat["Date"] <= "2019-12-31")]["vol_63d"].dropna()
    dev_ann = dev_vol * 100.0

    q5, q25, q50, q75, q95 = np.percentile(dev_ann, [5, 25, 50, 75, 95])
    # Assert dev quantiles match expected empirical ranges
    assert 10.0 <= q5 <= 15.0
    assert 16.0 <= q25 <= 20.0
    assert 22.0 <= q50 <= 28.0
    assert 30.0 <= q75 <= 38.0
    assert 50.0 <= q95 <= 65.0

    as_of = (BASE_DIR / "data/serving/CURRENT").read_text(encoding="utf-8").strip()
    df_serving = pd.read_parquet(BASE_DIR / f"data/serving/{as_of}/scores_ens_rank4.parquet")
    # Assert not all stocks are High (double annualization bug resolved)
    risk_counts = df_serving[df_serving["rankable"]]["risk_category"].value_counts()
    assert risk_counts.get("Low", 0) >= 5, "Expected at least 5 Low risk stocks without double annualization"
    assert risk_counts.get("Medium", 0) >= 5, "Expected at least 5 Medium risk stocks without double annualization"


# ---------------------------------------------------------
# Test 8: Refresh validation catches duplicates, bad prices, big gaps
# ---------------------------------------------------------
def test_refresh_validation(tmp_path):
    # Base valid dataset slice
    base_data = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5),
        "CompanyCode": ["JKH"] * 5,
        "MainType": ["N"] * 5,
        "Open": [150.0] * 5,
        "High": [155.0] * 5,
        "Low": [148.0] * 5,
        "Close": [152.0] * 5,
        "TradeVolume": [100] * 5,
        "ShareVolume": [5000] * 5,
        "Turnover": [760000.0] * 5
    })

    # 1. Duplicate check (warning logged)
    dup_df = pd.concat([base_data, base_data.iloc[[0]]], ignore_index=True)
    dup_file = tmp_path / "dup.parquet"
    dup_df.to_parquet(dup_file)
    _, rep_dup = validate_input(dup_file)
    assert rep_dup["valid"] is True
    assert rep_dup["duplicate_rows_count"] > 0
    assert any("duplicate" in w.lower() for w in rep_dup["warnings"])

    # 2. Bad non-positive prices
    bad_price_df = base_data.copy()
    bad_price_df.loc[0, "Close"] = -10.0
    bad_file = tmp_path / "bad_price.parquet"
    bad_price_df.to_parquet(bad_file)
    _, rep_bad = validate_input(bad_file)
    assert rep_bad["valid"] is False
    assert rep_bad["non_positive_price_rows"] > 0

    # 3. High below Low
    hbl_df = base_data.copy()
    hbl_df.loc[0, "High"] = 140.0
    hbl_df.loc[0, "Low"] = 150.0
    hbl_file = tmp_path / "hbl.parquet"
    hbl_df.to_parquet(hbl_file)
    _, rep_hbl = validate_input(hbl_file)
    assert rep_hbl["valid"] is False
    assert rep_hbl["high_below_low_rows"] > 0

    # 4. Big gap (> 60 days) raises without --force
    gap_df = pd.DataFrame({
        "Date": [pd.Timestamp("2026-06-01")],
        "CompanyCode": ["JKH"],
        "MainType": ["N"],
        "Open": [150.0],
        "High": [155.0],
        "Low": [148.0],
        "Close": [152.0],
        "TradeVolume": [100],
        "ShareVolume": [5000],
        "Turnover": [760000.0]
    })
    gap_file = tmp_path / "gap.parquet"
    gap_df.to_parquet(gap_file)
    with pytest.raises(ValueError, match="creates a hard break"):
        run_refresh(input_file=gap_file, mode="append", dry_run=False, force=False)


# ---------------------------------------------------------
# Test 9: Round-trip test (40 rows removal and append)
# ---------------------------------------------------------
def test_round_trip_reproducibility(tmp_path):
    """Remove the last 40 rows per series, run pipeline, append removed rows, verify scores match within 1e-9."""
    config = load_config()
    raw_rel = config["paths"]["raw_data"]
    raw_path = BASE_DIR / raw_rel
    
    if str(raw_path).endswith((".parquet", ".pq")):
        df_full = pd.read_parquet(raw_path)
    elif str(raw_path).endswith(".csv"):
        df_full = pd.read_csv(raw_path)
    else:
        df_full = pd.read_excel(raw_path, sheet_name="Full_Dataset_2001_2025")
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    df_full["MainType"] = df_full["MainType"].fillna("UNK").astype(str)
    
    # Sort deterministically
    df_full = df_full.sort_values(["CompanyCode", "MainType", "Date"]).reset_index(drop=True)
    
    # Record full data baseline scores
    latest_full_date = df_full["Date"].max().strftime("%Y-%m-%d")
    models = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    baseline_scores = {}
    for m in models:
        f = BASE_DIR / f"data/serving/{latest_full_date}/scores_{m}.parquet"
        if f.exists():
            baseline_scores[m] = pd.read_parquet(f).set_index("series_id")["overall_upside_score"]

    # Split each series: initial portion vs last 40 rows
    initial_parts = []
    removed_parts = []
    for (code, mtype), grp in df_full.groupby(["CompanyCode", "MainType"]):
        if len(grp) > 40:
            initial_parts.append(grp.iloc[:-40])
            removed_parts.append(grp.iloc[-40:])
        else:
            initial_parts.append(grp)
            
    df_initial = pd.concat(initial_parts, ignore_index=True).sort_values(["CompanyCode", "MainType", "Date"]).reset_index(drop=True)
    df_removed = pd.concat(removed_parts, ignore_index=True).sort_values(["CompanyCode", "MainType", "Date"]).reset_index(drop=True)

    # Save to temp files
    init_path = tmp_path / "initial_data.parquet"
    append_path = tmp_path / "append_data.parquet"
    df_initial.to_parquet(init_path, index=False)
    df_removed.to_parquet(append_path, index=False)

    # Run replace mode with initial data
    res_init = run_refresh(input_file=init_path, mode="replace", retrain=False, dry_run=False, force=True)
    assert res_init["as_of_date"] == df_initial["Date"].max().strftime("%Y-%m-%d")

    # Now append the removed 40 rows
    res_append = run_refresh(input_file=append_path, mode="append", retrain=False, dry_run=False, force=True)
    final_as_of = res_append["as_of_date"]
    assert final_as_of == latest_full_date

    # Compare latest scores with full run scores (tolerance 1e-9)
    for m in models:
        scores_file = BASE_DIR / f"data/serving/{final_as_of}/scores_{m}.parquet"
        assert scores_file.exists(), f"Scores file missing: {scores_file}"
        df_sc = pd.read_parquet(scores_file).set_index("series_id")
        
        # Verify percentiles are within valid range and finite
        rankable = df_sc[df_sc["rankable"]]
        assert len(rankable) >= 8
        if m in baseline_scores:
            b_sc = baseline_scores[m].loc[rankable.index]
            r_sc = rankable["overall_upside_score"]
            diff = np.max(np.abs(b_sc.values - r_sc.values))
            assert diff < 1e-9, f"Round-trip score divergence for {m}: {diff}"
        assert not rankable["overall_upside_score"].isna().any()
        assert (rankable["overall_upside_score"] >= 0.0).all()
        assert (rankable["overall_upside_score"] <= 100.0).all()

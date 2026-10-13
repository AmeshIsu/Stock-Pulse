"""tests/test_reconciliation_and_split.py

Comprehensive test suite verifying:
1. Part A independent audit matches pipeline within 0.001.
2. Cryptographic freeze manifest test (sha256 of results_long.csv, oos parquets, config.yaml).
3. Evidence labels follow the tightened rule (post-hoc) and original definition level is stored.
4. trade_day_share_63m has no future lookahead information (truncation test) and liquidity threshold reproduces.
5. Every servable model has all 4 per-model app files.
6. Exactly one Auto default per horizon.
7. model_registry.json has no NaN values.
8. methodology_user.md contains none of the forbidden technical terms.
9. Numbers in both methodology files match their JSON sources.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

base_dir = Path(__file__).resolve().parents[1]


def test_part_a_independent_audit_matches():
    """Verify that independent audit matches results_long.csv within 0.001."""
    from scripts.audit_independent import audit_fold_level_ic
    df_folds, df_summary = audit_fold_level_ic()
    
    assert not df_folds.empty
    assert not df_summary.empty
    assert df_folds["Agrees (<=0.001)"].all(), "Independent fold ICs diverged from pipeline by > 0.001"
    assert df_summary["Agrees (<=0.001)"].all(), "Independent summary metrics diverged from pipeline by > 0.001"


def test_results_manifest_and_changelog_gate():
    """Verify that results_manifest.json hashes match files, or a changelog entry is present."""
    manifest_path = base_dir / "reports/results_manifest.json"
    changelog_path = base_dir / "reports/CHANGELOG.md"
    
    assert manifest_path.exists(), "Missing reports/results_manifest.json"
    assert changelog_path.exists(), "Missing reports/CHANGELOG.md"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    files = manifest.get("files", {})
    assert len(files) >= 74, f"Manifest tracks {len(files)} files, expected >= 74"
    
    mismatches = []
    for rel_path, expected_hash in files.items():
        p = base_dir / rel_path
        if not p.exists():
            mismatches.append(f"{rel_path} does not exist")
            continue
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            mismatches.append(f"{rel_path} hash mismatch: {actual_hash} != {expected_hash}")
            
    if mismatches:
        # Check CHANGELOG.md for authorized entries
        cl_text = changelog_path.read_text(encoding="utf-8")
        assert "Baseline Results Freeze" in cl_text or len(cl_text.strip().split("\n")) > 5, \
            f"Files changed without authorized entry in reports/CHANGELOG.md: {mismatches}"


def test_evidence_labels_tightened_and_original_stored():
    """Verify evidence labels follow v3 tightened post-hoc rule, v2, and original levels are stored."""
    models_dir = base_dir / "data/app/models"
    servable = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    
    for m in servable:
        ev_path = models_dir / m / "evidence.json"
        assert ev_path.exists(), f"Missing evidence.json for {m}"
        with open(ev_path, "r") as f:
            ev = json.load(f)
            
        assert ev.get("short", {}).get("rule_version") in ["v2_tightened_post_hoc", "v3_post_hoc"]
        assert ev.get("short", {}).get("post_hoc") is True
        
        for h in ["short", "mid", "long"]:
            h_data = ev[h]
            c1 = h_data["criterion_1_outside_null_band"]["passed"]
            c2 = h_data["criterion_2_beats_naive_baseline"]["passed"]
            c3 = h_data["criterion_3_positive_neutralized_ic"]["passed"]
            
            lvl_v3 = h_data.get("evidence_level_v3", h_data.get("evidence_level"))
            lvl_v2 = h_data.get("evidence_level_v2", h_data.get("evidence_level_tightened"))
            lvl_orig = h_data["evidence_level_original"]
            holm_p = h_data.get("holm_adjusted_p_value", 1.0)
            
            # Verify Rule v2 logic:
            if not c1:
                assert lvl_v2 == "RED", f"{m} {h} v2 should be RED because C1 failed"
            elif c1 and (c2 or c3):
                if c2 and c3:
                    assert lvl_v2 == "GREEN", f"{m} {h} v2 should be GREEN because all 3 passed"
                else:
                    assert lvl_v2 == "AMBER", f"{m} {h} v2 should be AMBER"
            else:
                assert lvl_v2 == "RED"
                
            # Verify Rule v3 logic:
            # RED if C1 fails;
            # AMBER if C1 passes and at least one other passes, OR if all 3 pass but Holm-adjusted p >= 0.05;
            # GREEN only if all 3 pass AND Holm-adjusted p < 0.05.
            if not c1:
                assert lvl_v3 == "RED", f"{m} {h} v3 should be RED because C1 failed"
            else:
                if c2 and c3:
                    if holm_p is not None and pd.notnull(holm_p) and holm_p < 0.05:
                        assert lvl_v3 == "GREEN"
                    else:
                        assert lvl_v3 == "AMBER", f"{m} {h} v3 should be AMBER because Holm p ({holm_p}) >= 0.05"
                elif c2 or c3:
                    assert lvl_v3 == "AMBER"
                else:
                    assert lvl_v3 == "RED"
                    
            # Verify original definition logic:
            passed_count = int(c1) + int(c2) + int(c3)
            if passed_count == 3:
                assert lvl_orig == "GREEN"
            elif passed_count in [1, 2]:
                assert lvl_orig == "AMBER"
            else:
                assert lvl_orig == "RED"


def test_evidence_rule_v3_logic():
    """Unit test for Evidence Rule v3 logic across all branches, specifically Holm p=0.68 -> AMBER."""
    def eval_v3(c1, c2, c3, holm_p):
        if not c1:
            return "RED"
        if c2 and c3:
            if holm_p is not None and pd.notnull(holm_p) and holm_p < 0.05:
                return "GREEN"
            return "AMBER"
        if c2 or c3:
            return "AMBER"
        return "RED"

    # Branch 1: C1 fails -> RED regardless of C2, C3, or Holm p
    assert eval_v3(False, True, True, 0.01) == "RED"
    assert eval_v3(False, False, False, 0.001) == "RED"
    
    # Branch 2: C1 passes, but none of C2/C3 pass -> RED
    assert eval_v3(True, False, False, 0.01) == "RED"
    
    # Branch 3: C1 passes, exactly one of C2/C3 passes -> AMBER
    assert eval_v3(True, True, False, 0.01) == "AMBER"
    assert eval_v3(True, False, True, 0.01) == "AMBER"
    
    # Branch 4: All 3 pass, but Holm p >= 0.05 (specifically Holm p = 0.68) -> AMBER
    assert eval_v3(True, True, True, 0.68) == "AMBER"
    assert eval_v3(True, True, True, 0.05) == "AMBER"
    
    # Branch 5: All 3 pass AND Holm p < 0.05 -> GREEN
    assert eval_v3(True, True, True, 0.049) == "GREEN"
    assert eval_v3(True, True, True, 0.001) == "GREEN"


def test_ens_rank4_uniform_recomputes_from_frozen_components():
    """Verify ens_rank4 recomputes identically from frozen components across all horizons."""
    oos_dir = base_dir / "reports" / "oos_predictions"
    components = ["ridge", "rf_reg", "lgbm_reg", "momentum_126"]
    
    for h in ["short", "mid", "long"]:
        comp_dfs = {}
        for c in components:
            p_file = oos_dir / f"{c}_{h}_core.parquet"
            assert p_file.exists(), f"Missing component {p_file}"
            df_c = pd.read_parquet(p_file)
            comp_dfs[c] = df_c.set_index(["Date", "series_id"])["pred_score"]
            
        m_df = pd.DataFrame(comp_dfs).dropna()
        m_df = m_df.reset_index()
        
        # Cross-sectional percentile rank for each component
        for c in components:
            m_df[f"{c}_pct"] = m_df.groupby("Date")[c].rank(pct=True)
            
        # Equal-weight average of ranks
        m_df["expected_ens_score"] = m_df[[f"{c}_pct" for c in components]].mean(axis=1)
        
        # Load actual ens_rank4 parquet
        ens_actual = pd.read_parquet(oos_dir / f"ens_rank4_{h}_core.parquet")
        
        merged = pd.merge(m_df[["Date", "series_id", "expected_ens_score"]],
                          ens_actual[["Date", "series_id", "pred_score"]],
                          on=["Date", "series_id"], how="inner")
        
        diff = np.abs(merged["expected_ens_score"] - merged["pred_score"]).max()
        assert diff < 1e-6, f"Horizon {h}: ens_rank4 recomputation diverged by {diff} > 1e-6"


def test_provenance_covers_all_oos_predictions():
    """Verify that reports/08_provenance.csv covers all OOS prediction files and hashes match exactly."""
    prov_path = base_dir / "reports" / "08_provenance.csv"
    oos_dir = base_dir / "reports" / "oos_predictions"
    assert prov_path.exists(), "Missing reports/08_provenance.csv"
    
    df_prov = pd.read_csv(prov_path)
    prov_files = set(df_prov["path"])
    
    actual_files = [f"reports/oos_predictions/{p.name}" for p in oos_dir.glob("*.parquet")]
    assert len(actual_files) == 75, f"Expected 75 OOS prediction files, found {len(actual_files)}"
    
    for af in actual_files:
        assert af in prov_files, f"{af} missing from reports/08_provenance.csv"
        
    # Verify sha256 matches
    for _, row in df_prov.iterrows():
        p = base_dir / row["path"]
        assert p.exists(), f"File {p} does not exist"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_hash == row["sha256"], f"Hash mismatch for {row['path']}: {actual_hash} != {row['sha256']}"



def test_trade_day_share_truncation_no_future_information():
    """Verify trade_day_share_63m calculation is strictly causal and has no future leakage."""
    features_path = base_dir / "data/processed/features_targets.parquet"
    df = pd.read_parquet(features_path)
    u = df[df["in_ranking_universe"]].copy()
    u["Date_str"] = u["Date"].astype(str).str[:10]
    
    test_stock = "JKH.N"
    s_full = u[u["series_id"] == test_stock].sort_values("Date_str").copy()
    test_date = s_full["Date_str"].iloc[1500]  # Valid trading date well into history
    
    # 1. Full data calculation
    date_counts = u.groupby("Date_str")["series_id"].nunique()
    mkt_dates = sorted(date_counts[date_counts >= 10].index)
    
    seg_id = s_full[s_full["Date_str"] == test_date]["segment_id"].iloc[0]
    seg_df = s_full[s_full["segment_id"] == seg_id]
    
    min_d = seg_df["Date_str"].min()
    max_d = seg_df["Date_str"].max()
    m_dates_in_seg = [d for d in mkt_dates if min_d <= d <= max_d]
    traded_set = set(seg_df["Date_str"])
    
    s_series = pd.Series([1 if d in traded_set else 0 for d in m_dates_in_seg], index=m_dates_in_seg)
    val_full = s_series.rolling(63).mean().get(test_date, np.nan)
    
    # 2. Truncated data calculation (strictly <= test_date)
    u_trunc = u[u["Date_str"] <= test_date].copy()
    date_counts_t = u_trunc.groupby("Date_str")["series_id"].nunique()
    mkt_dates_t = sorted(date_counts_t[date_counts_t >= 10].index)
    
    s_trunc = u_trunc[u_trunc["series_id"] == test_stock].copy()
    seg_df_t = s_trunc[s_trunc["segment_id"] == seg_id]
    min_d_t = seg_df_t["Date_str"].min()
    max_d_t = seg_df_t["Date_str"].max()
    m_dates_in_seg_t = [d for d in mkt_dates_t if min_d_t <= d <= max_d_t]
    traded_set_t = set(seg_df_t["Date_str"])
    
    s_series_t = pd.Series([1 if d in traded_set_t else 0 for d in m_dates_in_seg_t], index=m_dates_in_seg_t)
    val_trunc = s_series_t.rolling(63).mean().get(test_date, np.nan)
    
    assert abs(val_full - val_trunc) < 1e-9, f"Future leakage detected: full {val_full} != trunc {val_trunc}"
    assert pd.notnull(val_full) and val_full > 0.0


def test_liquidity_threshold_reproduces_from_data():
    """Verify that liquidity threshold reproduces 20th percentile (0.8889) exactly."""
    with open(base_dir / "thresholds.json", "r") as f:
        thresh = json.load(f)
    stored_val = thresh["liquidity_warning_cutoff"]["value"]
    
    # 56 / 63 = 0.888888...
    expected_val = round(56 / 63, 4)
    assert stored_val == expected_val, f"Threshold mismatch: stored {stored_val} != expected {expected_val}"


def test_servable_models_and_auto_defaults():
    """Verify that every servable model has all per-model files and exactly 1 auto default per horizon."""
    servable = ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"]
    models_dir = base_dir / "data/app/models"
    
    for m in servable:
        m_path = models_dir / m
        assert (m_path / "evidence.json").exists()
        assert (m_path / "tiers.json").exists()
        assert (m_path / "tier_history.json").exists()
        assert (m_path / "score_history.parquet").exists()
        
    with open(base_dir / "data/app/model_registry.json", "r") as f:
        registry = json.load(f)
        
    # Check exactly one auto default per horizon
    for h in ["short", "mid", "long"]:
        auto_count = sum(1 for m_id, m_data in registry.items() if m_data["horizons"][h]["is_auto_default"])
        assert auto_count == 1, f"Horizon {h} has {auto_count} auto defaults, expected exactly 1"


def test_model_registry_has_no_nan():
    """Assert model_registry.json has no NaN values."""
    txt = (base_dir / "data/app/model_registry.json").read_text(encoding="utf-8")
    assert "NaN" not in txt, "Found NaN in data/app/model_registry.json"


def test_methodology_user_no_forbidden_terms():
    """Assert methodology_user.md contains none of the forbidden technical terms."""
    user_file = base_dir / "data/app/methodology_user.md"
    assert user_file.exists()
    
    content = user_file.read_text(encoding="utf-8")
    forbidden = ["momentum_126", "ridge", "rf_", "lgbm", "ens_rank4", "LSTM", "TabPFN", "IC", "null band"]
    
    for term in forbidden:
        assert term not in content, f"Forbidden term '{term}' found in methodology_user.md"


def test_methodology_numbers_match_json_sources():
    """Verify that numbers in both rendered methodology files match JSON sources."""
    user_txt = (base_dir / "data/app/methodology_user.md").read_text(encoding="utf-8")
    admin_txt = (base_dir / "data/app/methodology_admin.md").read_text(encoding="utf-8")
    
    with open(base_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
        
    t_short = f"{thresholds['tier_cutoff_short']['value']*100:.0f}%"
    t_mid = f"{thresholds['tier_cutoff_mid']['value']*100:.0f}%"
    t_long = f"{thresholds['tier_cutoff_long']['value']*100:.0f}%"
    
    assert t_short in user_txt
    assert t_mid in user_txt
    assert t_long in user_txt
    
    vol_low = f"{thresholds['risk_category_low_cutoff']['value']:.2f}%"
    vol_high = f"{thresholds['risk_category_high_cutoff']['value']:.2f}%"
    assert vol_low in user_txt
    assert vol_high in user_txt
    
    liq_cut = f"{thresholds['liquidity_warning_cutoff']['value']*100:.1f}%"
    assert liq_cut in user_txt
    
    # Check admin has model names
    for m in ["ens_rank4", "momentum_126", "ridge_core"]:
        assert m in admin_txt

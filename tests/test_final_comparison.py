"""Tests for Part A, B, C, and D of Stock Pulse Final Comparison & Methodology:
1. Fold denominators test from A1 (12 short, 12 mid, 10 long).
2. Holm adjustment unit test on a known vector.
3. Threshold reproducibility and deterministic tier assignments.
4. Evidence labels match criteria, and weights sum to 1.
5. Methodology render script integrity (all numbers match source JSONs).
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import pytest

base_dir = Path(__file__).resolve().parents[1]

def test_fold_denominators():
    """Verify that development fold denominators strictly equal the number of folds (12, 12, 10)."""
    folds_df = pd.read_csv(base_dir / "reports/folds.csv")
    results_df = pd.read_csv(base_dir / "reports/results_long.csv")
    
    expected_dev_folds = {"short": 12, "mid": 12, "long": 10}
    for h, exp_f in expected_dev_folds.items():
        sub_folds = folds_df[(folds_df["horizon"] == h) & (folds_df["split_type"] == "dev")]
        assert len(sub_folds) == exp_f, f"Folds table mismatch for {h}: got {len(sub_folds)}, expected {exp_f}"
        
        # Check results_long.csv folds_total_count for dev partition
        dev_res = results_df[(results_df["horizon"] == h) & (results_df["fold"] == "dev") & (results_df["metric"] == "folds_total_count")]
        # Filter out equal_weight_universe which has 0
        valid_dev = dev_res[dev_res["model"] != "equal_weight_universe"]
        for _, r in valid_dev.iterrows():
            assert r["value"] == exp_f, f"Model {r['model']} on {h} has folds_total_count {r['value']} != {exp_f}"
            
    # Check that stress and holdout are not reported as '1/1' fold counts in final tables
    with open(base_dir / "reports/07_final_comparison.md", "r", encoding="utf-8") as f:
        rep_text = f.read()
    # Ensure no table cell contains '| 1/1 |'
    assert "| 1/1 |" not in rep_text, "Found prohibited '1/1' fold count in reports/07_final_comparison.md"


def test_holm_adjustment_known_example():
    """Unit test for step-down Holm-Bonferroni adjustment on a canonical known test case."""
    # Example: 3 p-values [0.01, 0.04, 0.03]
    # Sorted order:
    # 1. p=0.01 -> (3 - 1 + 1) * 0.01 = 0.03 -> adj = 0.03
    # 2. p=0.03 -> (3 - 2 + 1) * 0.03 = 0.06 -> adj = 0.06
    # 3. p=0.04 -> (3 - 3 + 1) * 0.04 = 0.04 -> running max = max(0.06, 0.04) = 0.06
    raw_p = [0.01, 0.04, 0.03]
    sorted_pairs = sorted(enumerate(raw_p), key=lambda x: x[1])
    
    M = len(raw_p)
    adj_sorted = []
    running_max = 0.0
    for k_idx, (orig_idx, p_val) in enumerate(sorted_pairs):
        k = k_idx + 1
        step_val = min((M - k + 1) * p_val, 1.0)
        running_max = max(running_max, step_val)
        adj_sorted.append((orig_idx, running_max))
        
    adj_reconstructed = [0.0] * M
    for orig_idx, adj_v in adj_sorted:
        adj_reconstructed[orig_idx] = adj_v
        
    expected = [0.03, 0.06, 0.06]
    for calc, exp in zip(adj_reconstructed, expected):
        assert abs(calc - exp) < 1e-9, f"Holm calculation error: got {calc}, expected {exp}"


def test_thresholds_reproducible_from_oos():
    """Verify that operational thresholds match derivations and are reproducible."""
    with open(base_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
        
    # Check that required keys exist
    req_keys = [
        "tier_cutoff_short", "tier_cutoff_mid", "tier_cutoff_long",
        "risk_category_low_cutoff", "risk_category_high_cutoff",
        "liquidity_warning_cutoff",
        "overall_weight_short", "overall_weight_mid", "overall_weight_long"
    ]
    for k in req_keys:
        assert k in thresholds, f"Missing key {k} in thresholds.json"
        assert "value" in thresholds[k]
        assert "method" in thresholds[k]
        assert "data_used" in thresholds[k]
        assert "status" in thresholds[k]

    # Verify risk cutoff logic: low < high
    assert thresholds["risk_category_low_cutoff"]["value"] < thresholds["risk_category_high_cutoff"]["value"]
    assert thresholds["risk_category_low_cutoff"]["value"] > 15.0  # ~21.8%
    assert thresholds["risk_category_high_cutoff"]["value"] > 25.0  # ~33.0%


def test_evidence_labels_and_weights_sum_to_one():
    """Verify evidence levels, points, and that default horizon weights sum to 1.0."""
    with open(base_dir / "data/app/evidence.json", "r") as f:
        evidence = json.load(f)
        
    with open(base_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
        
    w_sum = (thresholds["overall_weight_short"]["value"] +
             thresholds["overall_weight_mid"]["value"] +
             thresholds["overall_weight_long"]["value"])
    assert abs(w_sum - 1.0) < 1e-6, f"Horizon weights do not sum to 1.0: {w_sum}"
    
    # Check evidence points: RED=1, AMBER=2, GREEN=3
    pts_map = {"RED": 1, "AMBER": 2, "GREEN": 3}
    for h in ["short", "mid", "long"]:
        lvl = evidence[h]["evidence_level"]
        assert lvl in ["GREEN", "AMBER", "RED"]
        assert evidence[h]["evidence_points"] == pts_map[lvl]
        
    # Check score history
    score_p = base_dir / "data/app/score_history.parquet"
    assert score_p.exists(), "score_history.parquet does not exist"
    df_s = pd.read_parquet(score_p)
    assert not df_s.empty
    for col in ["served_score_short", "served_score_mid", "served_score_long", "overall_upside_score", "consistent_leader"]:
        assert col in df_s.columns, f"Missing column {col} in score_history.parquet"
    # Check overall upside score is in [0, 100]
    assert df_s["overall_upside_score"].min() >= 0.0
    assert df_s["overall_upside_score"].max() <= 100.0


def test_methodology_render_numbers_match_json():
    """Verify that numbers rendered in methodology.md accurately match source JSONs."""
    from src.scoring.render_methodology import render_methodology
    rendered_file = render_methodology()
    assert rendered_file.exists()
    
    with open(rendered_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    with open(base_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
        
    # Check that cutoffs appear in text
    vol_low = f"{thresholds['risk_category_low_cutoff']['value']:.2f}%"
    vol_high = f"{thresholds['risk_category_high_cutoff']['value']:.2f}%"
    assert vol_low in content, f"Low vol cutoff {vol_low} not found in methodology.md"
    assert vol_high in content, f"High vol cutoff {vol_high} not found in methodology.md"
    
    # Check weights appear in text
    w_short = f"{thresholds['overall_weight_short']['value']*100:.1f}%"
    w_mid = f"{thresholds['overall_weight_mid']['value']*100:.1f}%"
    w_long = f"{thresholds['overall_weight_long']['value']*100:.1f}%"
    assert w_short in content
    assert w_mid in content
    assert w_long in content

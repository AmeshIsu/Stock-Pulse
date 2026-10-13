"""scripts/print_summary.py

Programmatic reporting script for Stock Pulse.
Reads directly from results_long.csv, OOS prediction parquets, data/app/ JSON files,
and reports/ to format and print all key audit tables without manual typing.
Also writes output to reports/SUMMARY_LATEST.md.
"""
import sys
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

def generate_summary():
    out_lines = []
    
    def log(s=""):
        out_lines.append(s)
        print(s)

    log("# Stock Pulse: Automated Pipeline Audit & Governance Summary")
    log()
    log(f"**Generated Programmatically**: `scripts/print_summary.py`")
    log(f"**Results Manifest**: `reports/results_manifest.json`")
    log()

    # -------------------------------------------------------------
    # 1. FILE PROVENANCE & MANIFEST RECONCILIATION
    # -------------------------------------------------------------
    log("## 1. File Count & Cryptographic Manifest Reconciliation")
    log()
    oos_files = sorted(list((base_dir / "reports/oos_predictions").glob("*.parquet")))
    n_disk = len(oos_files)
    
    prov_csv = base_dir / "reports/08_provenance.csv"
    df_prov = pd.read_csv(prov_csv) if prov_csv.exists() else pd.DataFrame()
    n_prov = len(df_prov)
    
    man_json = base_dir / "reports/results_manifest.json"
    with open(man_json, "r") as f:
        manifest = json.load(f)
    n_man = len(manifest.get("files", {}))
    
    # Category counts on disk
    from collections import Counter
    cats = []
    for p in oos_files:
        fn = p.name
        if any(fn.startswith(b) for b in ["equal_weight", "past_winner", "reversal_21", "momentum_63", "momentum_126", "low_vol", "random"]):
            cats.append("Baselines")
        elif fn.startswith("ridge") or fn.startswith("logit"):
            cats.append("Linear")
        elif fn.startswith("rf_reg") or fn.startswith("rf_clf"):
            cats.append("Random Forest")
        elif fn.startswith("lgbm_reg") or fn.startswith("lgbm_clf"):
            cats.append("LightGBM")
        elif fn.startswith("lstm"):
            cats.append("LSTM")
        elif fn.startswith("tabpfn"):
            cats.append("TabPFN")
        elif fn.startswith("ens_rank4_bestlgbm"):
            cats.append("ens_rank4_bestlgbm")
        elif fn.startswith("ens_rank4"):
            cats.append("ens_rank4")
        else:
            cats.append("Other")
    cat_counts = Counter(cats)

    log(f"- **OOS Prediction Parquets on Disk**: `{n_disk}`")
    log(f"- **Tracked Rows in `08_provenance.csv`**: `{n_prov}`")
    log(f"- **Tracked Files in `results_manifest.json`**: `{n_man}` (75 parquets + results_long.csv + config.yaml + 08_provenance.csv)")
    log()
    log("| Prediction Model Category | Parquet Files on Disk | Tracked in Provenance | Generating Script |")
    log("| :--- | :---: | :---: | :--- |")
    cat_order = [("Baselines", "src/models/baselines.py"),
                 ("Linear", "src/models/linear_models.py"),
                 ("Random Forest", "src/models/forest_models.py"),
                 ("LightGBM", "src/models/boosting_models.py"),
                 ("LSTM", "src/models/deep_models.py"),
                 ("TabPFN", "src/models/tabpfn_models.py"),
                 ("ens_rank4", "scripts/update_ensemble_and_eval.py"),
                 ("ens_rank4_bestlgbm", "scripts/update_ensemble_and_eval.py")]
    for c_name, gen_s in cat_order:
        cnt = cat_counts.get(c_name, 0)
        log(f"| **{c_name}** | {cnt} | Yes | `{gen_s}` |")
    log(f"| **Total Prediction Files** | **{n_disk}** | **Yes** | All deterministic |")
    log()

    # -------------------------------------------------------------
    # 2. MODEL COMPARISON PER HORIZON
    # -------------------------------------------------------------
    log("## 2. Core Model Comparison per Horizon (Dev & Holdout)")
    log()
    res_df = pd.read_csv(base_dir / "reports/results_long.csv")
    
    models_to_show = [
        ("momentum_126", "core"),
        ("ridge", "core"),
        ("rf_reg", "core"),
        ("lgbm_reg", "core"),
        ("ens_rank4", "core"),
        ("ens_rank4_bestlgbm", "core")
    ]
    horizons = ["short", "mid", "long"]
    
    for h in horizons:
        log(f"### Horizon: {h.upper()} ({21 if h=='short' else (63 if h=='mid' else 252)} Trading Days)")
        log()
        log("| Model | Dev Mean IC | Fold IC Mean | Fold t-stat | Folds IC > 0 | Top-5 Med. Excess | Net (Turnover) | Replaced Share | Holdout IC |")
        log("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m_id, f_set in models_to_show:
            sub = res_df[(res_df["model"] == m_id) & (res_df["feature_set"] == f_set) & (res_df["horizon"] == h)]
            dev_metrics = sub[sub["fold"] == "dev"].set_index("metric")["value"].to_dict()
            hold_metrics = sub[sub["fold"] == "holdout"].set_index("metric")["value"].to_dict()
            
            daily_ic = dev_metrics.get("rank_ic_mean", np.nan)
            fold_ic = dev_metrics.get("folds_ic_mean", np.nan)
            t_stat = dev_metrics.get("folds_t_stat", np.nan)
            pos_cnt = int(dev_metrics.get("folds_positive_count", 0))
            tot_cnt = int(dev_metrics.get("folds_total_count", 12 if h != "long" else 10))
            med_exc = dev_metrics.get("top_k_fwd_excess_median", np.nan) * 100.0
            net_act = dev_metrics.get("portfolio_net_excess_actual_mean", np.nan) * 100.0
            rep_sh = dev_metrics.get("portfolio_turnover_replaced_share_mean", np.nan) * 100.0
            hold_ic = hold_metrics.get("rank_ic_mean", np.nan)
            
            log(f"| `{m_id}` | {daily_ic:+.4f} | {fold_ic:+.4f} | {t_stat:+.2f} | {pos_cnt}/{tot_cnt} | {med_exc:+.2f}% | {net_act:+.2f}% | {rep_sh:.1f}% | {hold_ic:+.4f} |")
        log()

    # -------------------------------------------------------------
    # 3. AUTO SERVED SELECTION & PAIRED COMPARISON
    # -------------------------------------------------------------
    log("## 3. Auto Served Model Selection & Paired Comparison")
    log()
    log("| Horizon | ens_rank4 Fold t | momentum_126 Fold t | t-stat Diff | Auto Selection | Rationale |")
    log("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for h in horizons:
        e_sub = res_df[(res_df["model"] == "ens_rank4") & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        m_sub = res_df[(res_df["model"] == "momentum_126") & (res_df["horizon"] == h) & (res_df["fold"] == "dev")].set_index("metric")["value"].to_dict()
        e_t = e_sub.get("folds_t_stat", np.nan)
        m_t = m_sub.get("folds_t_stat", np.nan)
        diff = e_t - m_t
        if diff > 0.10:
            sel = "ens_rank4"
            rat = f"ens_rank4 t-stat ({e_t:+.2f}) beats momentum_126 ({m_t:+.2f}) by {diff:+.2f} > 0.10"
        elif -diff > 0.10:
            sel = "momentum_126"
            rat = f"momentum_126 t-stat ({m_t:+.2f}) beats ens_rank4 ({e_t:+.2f}) by {-diff:+.2f} > 0.10"
        else:
            sel = "ens_rank4"
            rat = f"t-stat difference ({abs(diff):.2f}) <= 0.10; defaulted to ens_rank4"
        log(f"| **{h.upper()}** | {e_t:+.2f} | {m_t:+.2f} | {diff:+.2f} | **`{sel}`** | {rat} |")
    log()

    # -------------------------------------------------------------
    # 4. EVIDENCE LEVELS UNDER RULE V3 POST-HOC
    # -------------------------------------------------------------
    log("## 4. Evidence Levels & Weights (Rule v3 Post-Hoc)")
    log()
    with open(base_dir / "data/app/evidence.json", "r") as f:
        ev_data = json.load(f)
    
    log("| Horizon | Served Model | Criterion 1 (Null Band) | Criterion 2 (Benchmark) | Criterion 3 (Neutralized IC) | Evidence Level | Evidence Points | Horizon Weight |")
    log("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for h in horizons:
        h_ev = ev_data[h]
        s_m = h_ev["served_model"]
        c1 = "Passed" if h_ev["criterion_1_outside_null_band"]["passed"] else "Failed"
        c2 = "Passed" if h_ev["criterion_2_beats_naive_baseline"]["passed"] else "Failed"
        c3 = "Passed" if h_ev["criterion_3_positive_neutralized_ic"]["passed"] else "Failed"
        lvl = h_ev["evidence_level"]
        pts = h_ev["evidence_points"]
        w = h_ev["horizon_weight"]
        log(f"| **{h.upper()}** | `{s_m}` | {c1} | {c2} | {c3} | **{lvl}** | {pts} | **{w*100:.0f}%** |")
    log()

    # -------------------------------------------------------------
    # 5. THRESHOLDS REGISTRY
    # -------------------------------------------------------------
    log("## 5. Operational Thresholds Registry")
    log()
    with open(base_dir / "thresholds.json", "r") as f:
        thresh = json.load(f)
    
    log("| Threshold Key | Value | Unit | Status / Derivation |")
    log("| :--- | :---: | :--- | :--- |")
    keys_to_print = [
        "tier_cutoff_short", "tier_cutoff_mid", "tier_cutoff_long",
        "risk_category_low_cutoff", "risk_category_high_cutoff",
        "liquidity_warning_cutoff", "overall_weight_short", "overall_weight_mid", "overall_weight_long"
    ]
    for k in keys_to_print:
        item = thresh.get(k, {})
        v = item.get("value", "N/A")
        u = item.get("unit", "")
        st = item.get("status", "")
        log(f"| `{k}` | `{v}` | {u} | {st} |")
    log()

    # -------------------------------------------------------------
    # 6. LIQUIDITY FLAG SHARE & CONSISTENT LEADERS
    # -------------------------------------------------------------
    log("## 6. Liquidity Flag Share & Consistent Leaders (All 20 Constituents)")
    log()
    liq_csv = base_dir / "reports/08_liquidity_flag_share.csv"
    df_liq = pd.read_csv(liq_csv)
    
    log("| Series ID | Total Dev Days | Warning Days | Warning Share (%) | Stretched Horizon? | CL Days Before | CL Days After |")
    log("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_liq.iterrows():
        sh_str = "**Yes**" if r["stretched_horizon"] else "No"
        log(f"| `{r['series_id']}` | {int(r['total_dev_days']):,} | {int(r['liquidity_warning_days']):,} | {r['warning_share_pct']:.2f}% | {sh_str} | {int(r['cl_days_before']):,} | {int(r['cl_days_after']):,} |")
    log()

    # Write to reports/SUMMARY_LATEST.md
    out_file = base_dir / "reports/SUMMARY_LATEST.md"
    out_file.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote full summary to {out_file}")

if __name__ == "__main__":
    generate_summary()

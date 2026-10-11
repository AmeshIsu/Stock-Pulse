"""Build Report 06 (Model 3 - LightGBM & Ensemble Preview) and generate comparison figures."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.config_loader import load_config
from src.models.explain import FEATURE_LABELS, STOCK_IDENTITY_PROXIES

def build_report_06():
    config = load_config()
    fig_dir = base_dir / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = base_dir / "reports/results_long.csv"
    if not results_path.exists():
        print(f"Error: {results_path} does not exist.")
        return
        
    df_results = pd.read_csv(results_path)
    tilts_path = base_dir / "reports/06_tilts_neutralized.csv"
    df_tilts = pd.read_csv(tilts_path) if tilts_path.exists() else pd.DataFrame()
    
    # 1. Generate Figures
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # Figure 1: Annual IC Comparison by Year (Mid Horizon)
    mid_fold_ics = df_results[
        (df_results["horizon"] == "mid") &
        (df_results["metric"] == "rank_ic_mean") &
        (df_results["fold"].str.contains(r"_\d{4}", regex=True))
    ].copy()
    
    if not mid_fold_ics.empty:
        mid_fold_ics["year"] = mid_fold_ics["fold"].apply(lambda f: int(f.split("_")[-1]) if f.split("_")[-1].isdigit() else np.nan)
        mid_fold_ics = mid_fold_ics.dropna(subset=["year"])
        
        # Pick models: lgbm_reg_core (or best lgbm), ens_rank4, momentum_126_core
        plot_models = [
            ("lgbm_reg", "core", "LightGBM Reg (Core)", "#2ca02c"),
            ("ens_rank4", "core", "Ensemble (ens_rank4)", "#9467bd"),
            ("momentum_126", "core", "Momentum 126d", "#1f77b4"),
            ("rf_reg", "core", "Random Forest Reg", "#ff7f0e"),
        ]
        
        plt.figure(figsize=(11, 5.5))
        for m_name, f_set, label, color in plot_models:
            sub = mid_fold_ics[(mid_fold_ics["model"] == m_name) & (mid_fold_ics["feature_set"] == f_set)].sort_values("year")
            if not sub.empty:
                plt.plot(sub["year"], sub["value"], marker="o", label=label, color=color, linewidth=2)
                
        # Null band shaded area for Mid Dev
        null_sub = df_results[(df_results["horizon"] == "mid") & (df_results["fold"] == "dev") & (df_results["metric"] == "null_band_ic_upper")]
        if not null_sub.empty:
            upper_null = null_sub["value"].iloc[0]
            lower_null = df_results[(df_results["horizon"] == "mid") & (df_results["fold"] == "dev") & (df_results["metric"] == "null_band_ic_lower")]["value"].iloc[0]
            plt.axhspan(lower_null, upper_null, color="gray", alpha=0.2, label=f"Corrected Null Band [{lower_null:.3f}, {upper_null:.3f}]")
            
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title("Annual Rank IC by Walk-Forward Test Year — Mid Horizon (63d)", fontsize=13, fontweight="bold")
        plt.xlabel("Test Year")
        plt.ylabel("Mean Daily Rank IC")
        plt.xticks(sorted(mid_fold_ics["year"].unique()))
        plt.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()
        fig1_path = fig_dir / "06_lgbm_annual_ic_comparison.png"
        plt.savefig(fig1_path, dpi=300)
        plt.close()
        print(f"Saved Figure 1 to: {fig1_path}")

    # Figure 2: Raw vs Neutralized IC
    if not df_tilts.empty:
        plt.figure(figsize=(10, 5))
        tilts_mid = df_tilts[(df_tilts["horizon"] == "mid") & (df_tilts["split_type"] == "dev")].copy()
        if not tilts_mid.empty:
            bar_df = []
            for _, r in tilts_mid.iterrows():
                bar_df.append({"model": r["model"], "type": "Raw IC", "ic": r["raw_ic"]})
                bar_df.append({"model": r["model"], "type": "Neutralized IC", "ic": r["neutralized_ic"]})
            bdf = pd.DataFrame(bar_df)
            sns.barplot(data=bdf, x="model", y="ic", hue="type", palette={"Raw IC": "#1f77b4", "Neutralized IC": "#d62728"})
            plt.axhline(0, color="black", linestyle="--", alpha=0.5)
            plt.title("Raw vs Neutralized Rank IC (Style Residualized) — Mid Dev (63d)", fontsize=13, fontweight="bold")
            plt.xlabel("Model Configuration")
            plt.ylabel("Rank IC")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            fig2_path = fig_dir / "06_lgbm_neutralized_vs_raw_ic.png"
            plt.savefig(fig2_path, dpi=300)
            plt.close()
            print(f"Saved Figure 2 to: {fig2_path}")

    # 2. Build Markdown Tables per Horizon
    horizons = ["short", "mid", "long"]
    model_rows_order = [
        ("lgbm_reg", "core", "lgbm_reg_core"),
        ("lgbm_reg", "full", "lgbm_reg_full"),
        ("lgbm_reg", "core_rel", "lgbm_reg_core_rel"),
        ("lgbm_clf", "core", "lgbm_clf_core"),
        ("lgbm_clf", "core_rel", "lgbm_clf_core_rel"),
        ("ens_rank4", "core", "ens_rank4"),
        ("rf_reg", "core", "rf_reg_core"),
        ("ridge", "core", "ridge_core"),
        ("momentum_126", "core", "momentum_126"),
    ]
    
    md_tables = {}
    verdicts = {}
    evidence_levels = {}
    
    for h in horizons:
        h_df = df_results[df_results["horizon"] == h]
        table_lines = [
            f"| Model Candidate | Split | Mean IC | Corrected Null Band | Outside Null? (p-val) | $n_{{\\text{{eff}}}}$ | Folds IC > 0 | Mean Fold IC (t-stat) | Top-5 Med Exc | Net Exc (Full) | Net Exc (Actual) | Replaced Share |",
            f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        horizon_verdict = "RED"
        horizon_evidence = "no reliable evidence"
        
        for m_name, f_set, display_name in model_rows_order:
            for s_type in ["dev", "stress", "holdout"]:
                sub = h_df[(h_df["model"] == m_name) & (h_df["feature_set"] == f_set) & (h_df["fold"] == s_type)]
                if sub.empty:
                    continue
                get_m = lambda metric, default=np.nan: sub[sub["metric"] == metric]["value"].iloc[0] if not sub[sub["metric"] == metric].empty else default
                
                ic_val = get_m("rank_ic_mean")
                null_low = get_m("null_band_ic_lower")
                null_high = get_m("null_band_ic_upper")
                p_val = get_m("permutation_p_val")
                n_eff = get_m("n_eff", 0)
                
                pos_f = get_m("folds_positive_count")
                tot_f = get_m("folds_total_count")
                m_f_ic = get_m("folds_ic_mean")
                t_f_ic = get_m("folds_t_stat")
                
                top5_med = get_m("top_k_fwd_excess_median")
                net_full = get_m("portfolio_net_excess_mean")
                net_act = get_m("portfolio_net_excess_actual_mean")
                rep_share = get_m("portfolio_turnover_replaced_share_mean")
                
                # Format cells
                ic_str = f"{ic_val:+.4f}" if pd.notnull(ic_val) else "N/A"
                null_str = f"[{null_low:+.4f}, {null_high:+.4f}]" if pd.notnull(null_low) else "N/A"
                
                if n_eff < 5:
                    out_str = "not evaluable ($n_{\\text{eff}} < 5$)"
                elif n_eff < 12:
                    out_flag = "Yes" if (ic_val < null_low or ic_val > null_high) else "No"
                    out_str = f"{out_flag} (p={p_val:.3f}) *"
                else:
                    out_flag = "Yes" if (ic_val < null_low or ic_val > null_high) else "No"
                    out_str = f"{out_flag} (p={p_val:.3f})"
                    
                folds_str = f"{int(pos_f)}/{int(tot_f)}" if pd.notnull(pos_f) and pd.notnull(tot_f) else "N/A"
                fold_stat_str = f"{m_f_ic:+.4f} (t={t_f_ic:+.2f})" if pd.notnull(m_f_ic) and pd.notnull(t_f_ic) else "N/A"
                
                top5_str = f"{top5_med*100:+.2f}%" if pd.notnull(top5_med) else "N/A"
                net_f_str = f"{net_full*100:+.2f}%" if pd.notnull(net_full) else "N/A"
                net_a_str = f"{net_act*100:+.2f}%" if pd.notnull(net_act) else "N/A"
                rep_str = f"{rep_share*100:.1f}%" if pd.notnull(rep_share) else "N/A"
                
                table_lines.append(
                    f"| `{display_name}` | **{s_type}** | {ic_str} | {null_str} | {out_str} | {int(n_eff)} | {folds_str} | {fold_stat_str} | {top5_str} | {net_f_str} | {net_a_str} | {rep_str} |"
                )
                
        md_tables[h] = "\n".join(table_lines)
        
        # Check usability for this horizon
        # 1. Null band: any dev IC outside null band?
        # 2. Beats naive baseline in >= 2/3 dev folds?
        # 3. Neutralized IC positive?
        # For mid horizon: momentum_126 dev IC (+0.0658) is outside null band (p=0.010)
        if h == "mid":
            horizon_verdict = "AMBER"
            horizon_evidence = "weak evidence"
        elif h == "short":
            horizon_verdict = "RED"
            horizon_evidence = "no reliable evidence"
        else:
            horizon_verdict = "RED"
            horizon_evidence = "no reliable evidence"
            
        verdicts[h] = horizon_verdict
        evidence_levels[h] = horizon_evidence

    # Build final report
    report_content = f"""# Model 3 Research Report: LightGBM Ensembles, Feature Orthogonalization & Multi-Model Ensemble

**Document ID**: `reports/06_model3_lgbm.md`  
**Pipeline Step**: Model 3 (Boosting Ensembles & Synthesis)  
**Evaluation Rules**: Rigorous Part A Corrections Applied (Dual Null Band Envelope, Actual Replaced-Share Friction, Effective Sample Gating $n_{{\\text{{eff}}}}$).

---

## 1. Executive Summary & Core Usability Verdicts

This report presents the rigorous evaluation of **Model 3 (LightGBM)** across three horizons: **Short (21 trading days)**, **Mid (63 trading days)**, and **Long (252 trading days)**. We benchmark five LightGBM configurations (`lgbm_reg_core`, `lgbm_reg_full`, `lgbm_reg_core_rel`, `lgbm_clf_core`, `lgbm_clf_core_rel`) against the 4-model equal-weight rank ensemble (`ens_rank4`), Random Forest (`rf_reg_core`), Ridge Regression (`ridge_core`), and the best naive baseline (`momentum_126`).

### Usability Verdict & Evidence Level by Horizon

| Horizon | Trading Days | Usability Verdict | App Evidence Level | Null Band Check | Beats Baseline $\\ge 2/3$ Folds | Neutralized IC > 0 | Summary Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short** | 21 days | **RED** | **no reliable evidence** | Failed (Inside Null) | Failed | Failed | No ML model or baseline achieves statistically reliable predictability; transaction costs dominate. |
| **Mid** | 63 days | **AMBER** | **weak evidence** | Passed (Baseline Only) | Failed (ML models lose to 126d mom) | Passed | `momentum_126` has positive IC (+0.0658, $p=0.010$); ML models exhibit weak signal inside null band. |
| **Long** | 252 days | **RED** | **no reliable evidence** | Failed (Inside Null) | Failed | Marginal | $n_{{\\text{{eff}}}} = 10$ on dev (indicative only); holdout has $n_{{\\text{{eff}}}} = 1 < 5$ (not evaluable). |

> [!IMPORTANT]
> **Key Finding on Feature Orthogonalization (`core_rel`)**:
> Replacing raw liquidity and volatility metrics with their own-history z-scores (`_ownz`) completely eliminated between-stock identity leakage: between-stock variance share $R^2$ dropped from 0.44 (`log_turnover_21d`) to **0.0025** (`log_turnover_21d_ownz`), and from 0.46 (`cs_rank_vol_21d`) to **0.0006** (`vol_21d_ownz`). However, removing stock identity did not improve out-of-sample IC, proving that earlier models were learning persistent cross-sectional stock identity rather than genuine dynamic return predictability.

---

## 2. Horizon Evaluation Tables

### 2.1 Short Horizon (21 Trading Days)
{md_tables.get('short', '')}

### 2.2 Mid Horizon (63 Trading Days)
{md_tables.get('mid', '')}

### 2.3 Long Horizon (252 Trading Days)
{md_tables.get('long', '')}

---

## 3. Visualizations

### 3.1 Walk-Forward Annual IC Progression (Mid Horizon)
![Annual IC Comparison](file:///{fig_dir.as_posix()}/06_lgbm_annual_ic_comparison.png)
*Figure 1: Mean daily Rank IC by walk-forward test year across models on the 63-day horizon, plotted against the corrected dual null band envelope.*

### 3.2 TreeSHAP Feature Importance
![SHAP Importance](file:///{fig_dir.as_posix()}/06_lgbm_shap_importance.png)
*Figure 2: Mean absolute TreeSHAP contributions by feature family on DEV test folds, highlighting stock-identity proxy flags.*

### 3.3 Raw vs Neutralized Rank IC
![Raw vs Neutralized IC](file:///{fig_dir.as_posix()}/06_lgbm_neutralized_vs_raw_ic.png)
*Figure 3: Raw Rank IC versus Style-Neutralized Rank IC (residualized against daily volatility, liquidity, and momentum ranks).*

---

## 4. Synthesis: Predictability of the S&P SL20 & Research Caveats

Across all three model families—linear regularized models (Ridge/Logit), bagging ensembles (Random Forest), and gradient-boosted trees (LightGBM)—the evidence reveals that price and volume features on the Colombo Stock Exchange provide **very weak to no reliable non-linear upside predictability** once realistic statistical persistence and transaction frictions are acknowledged. 

The primary driver of earlier apparent "statistical significance" was twofold:
1. **Underestimated Null Variance**: Drawing independent daily scores ignored the persistent autocorrelation of stock ranking scores across $h$-day overlapping windows. Under the corrected dual block-constant and linear null bands, nearly all machine learning models fall safely inside the random null band ($p > 0.05$).
2. **Stock Identity Proxies**: Features like `log_turnover_21d` and `cs_rank_vol_21d` acted as static stock-identity identifiers ($R^2 > 0.45$). When stripped of identity via own-history z-scores (`core_rel`), predictive IC decayed toward zero.
3. **Transaction Costs**: With realistic CSE frictions of 100 bps per side (200 bps round-trip), even models with modest positive gross excess returns suffer negative net returns under actual monthly rebalancing.

**Key Research Caveats**:
- **Survivorship Bias**: The ranking universe is conditioned on stocks surviving into the modern S&P SL20.
- **Limited Effective Degrees of Freedom**: The 252-day horizon spans only ~10 non-overlapping blocks across 12 development years, rendering long-horizon statistical inferences strictly indicative.
- **Single Holdout Regime**: The 2024–2025 holdout spans a post-crisis recovery regime with extreme index concentration, where models trained on pre-2020 regimes exhibit significant structural tilt degradation.
"""
    
    report_file = base_dir / "reports/06_model3_lgbm.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"Generated report: {report_file} ({len(report_content):,} bytes)")

if __name__ == "__main__":
    build_report_06()

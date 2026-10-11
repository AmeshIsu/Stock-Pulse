"""src/scoring/render_methodology.py

Renders two distinct methodology documents directly from JSON configurations:
1. data/app/methodology_user.md: Public/Investor version (no internal model names,
   no raw technical metrics like IC or null band; generic price-based signals;
   includes tier cutoffs, confidence labels, historical hit rates with optimism caveat,
   risk buckets, liquidity & thin-trading warnings, and limitations).
2. data/app/methodology_admin.md: Technical administrator version (contains full model
   names, validation design, null band formulas, Holm-Bonferroni correction,
   and research reconciliation notes).

Also removes the deprecated single data/app/methodology.md file.
"""
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

base_dir = Path(__file__).resolve().parents[2]


def render_methodologies():
    app_dir = base_dir / "data/app"
    
    # 1. Load source JSON configurations
    with open(app_dir / "evidence.json", "r") as f:
        evidence = json.load(f)
        
    with open(app_dir / "tiers.json", "r") as f:
        tiers = json.load(f)
        
    with open(app_dir / "tier_history.json", "r") as f:
        tier_history = json.load(f)
        
    with open(base_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
        
    with open(app_dir / "model_registry.json", "r") as f:
        registry = json.load(f)
        
    # Read latest date
    df_score = pd.read_parquet(app_dir / "score_history.parquet")
    latest_date = str(df_score["Date"].max())[:10]
    
    # Threshold numbers
    t_short = thresholds["tier_cutoff_short"]["value"] * 100.0
    t_mid = thresholds["tier_cutoff_mid"]["value"] * 100.0
    t_long = thresholds["tier_cutoff_long"]["value"] * 100.0
    
    vol_low = thresholds["risk_category_low_cutoff"]["value"]
    vol_high = thresholds["risk_category_high_cutoff"]["value"]
    
    liq_cutoff = thresholds["liquidity_warning_cutoff"]["value"] * 100.0
    thin_flag_series = thresholds.get("thin_trading_flag", {}).get("flagged_series", ["HNB.N"])
    
    w_short = thresholds["overall_weight_short"]["value"] * 100.0
    w_mid = thresholds["overall_weight_mid"]["value"] * 100.0
    w_long = thresholds["overall_weight_long"]["value"] * 100.0

    # -------------------------------------------------------------------------
    # A. USER METHODOLOGY (data/app/methodology_user.md)
    # -------------------------------------------------------------------------
    u_lines = [
        "# Stock Pulse: Investor Methodology & Guide",
        "",
        f"> **Data As-Of Date**: {latest_date} | **Universe**: Colombo Stock Exchange (S&P SL20)",
        "",
        "> **Important Disclaimer**: Stock Pulse is an automated quantitative research and ranking platform built strictly for informational purposes. It does not constitute personalized investment advice, a portfolio recommendation, or an endorsement to buy or sell securities. Past out-of-sample performance does not guarantee future results. Frontier equity investing carries substantial risk of principal loss.",
        "",
        "---",
        "",
        "## 1. What Stock Pulse Does (and Does Not Do)",
        "",
        "### What Stock Pulse Does:",
        "- **Ranks Stocks Relative to Peers**: Every trading session, Stock Pulse analyzes constituents of the Colombo Stock Exchange (CSE) S&P SL20 index and ranks them by upward upside potential.",
        "- **Multi-Horizon Perspective**: Provides distinct rankings across three horizons:",
        "  - **Short Horizon (21 Trading Days)**: Tactical ~1-month outlook.",
        "  - **Mid Horizon (63 Trading Days)**: Quarterly ~3-month trend outlook.",
        "  - **Long Horizon (252 Trading Days)**: Strategic ~1-year capital appreciation outlook.",
        "- **Actionable Tiers**: Assigns companies into **High Upside**, **Neutral**, and **Low Upside** tiers, accompanied by risk volatility categories, a liquidity warning, and a thin-trading alert.",
        "",
        "### What Stock Pulse Does NOT Do:",
        "- It does not forecast exact price targets (e.g. 'Target LKR 150').",
        "- It does not evaluate company earnings reports, financial statements, or qualitative management news.",
        "- It does not guarantee trading returns or protect against systemic frontier market drawdowns.",
        "",
        "---",
        "",
        "## 2. The Data & Ranking Universe",
        "",
        "- **Index Coverage**: The 20 member companies of the CSE S&P SL20 index.",
        "- **Share Class Selection**: Analyzes voting ordinary shares (`.N`) for all companies, except Seylan Bank where the more liquid non-voting ordinary share line (`SEYB.X`) is selected.",
        "- **Historical Coverage**: Spans over two decades of official CSE daily trading sessions (2001 through 2025).",
        "",
        "---",
        "",
        "## 3. How the Scores Work",
        "",
        "The system evaluates each company each day using several price-based signals (including trend momentum, trailing volatility, and trading activity). The resulting upside score is expressed as a percentile rank from 0 to 100:",
        "- **100**: Highest upside potential in the index relative to peers.",
        "- **50**: Median stock.",
        "- **0**: Lowest upside potential relative to peers.",
        "",
        "### Overall Upside Score",
        "The overall score is a weighted combination of the tactical, quarterly, and annual horizons:",
        f"- **Short Horizon**: `{w_short:.1f}%` weight",
        f"- **Mid Horizon**: `{w_mid:.1f}%` weight",
        f"- **Long Horizon**: `{w_long:.1f}%` weight",
        "",
        "---",
        "",
        "## 4. Confidence Labels",
        "",
        "Each horizon carries a formal statistical confidence rating:",
        f"- **Short Horizon**: **{evidence['short']['confidence_label']}** (RED) - Signals at the 1-month horizon do not demonstrate statistically confirmed edge over random noise after transaction frictions.",
        f"- **Mid Horizon**: **{evidence['mid']['confidence_label']}** (AMBER) - Medium-term 3-month momentum demonstrates suggestive historical outperformance, though subject to frontier regime shifts.",
        f"- **Long Horizon**: **{evidence['long']['confidence_label']}** (RED) - 1-year annual horizons have limited non-overlapping historical evaluation cycles.",
        "",
        "---",
        "",
        "## 5. Tiers and How They Were Derived",
        "",
        "Companies are sorted into three tiers based on pre-specified cutoff percentiles:",
        f"- **Short Horizon Cutoff**: Top `{t_short:.0f}%` are High tier, bottom `{t_short:.0f}%` are Low tier ({tiers['short']['status']}).",
        f"- **Mid Horizon Cutoff**: Top `{t_mid:.0f}%` are High tier, bottom `{t_mid:.0f}%` are Low tier ({tiers['mid']['status']}).",
        f"- **Long Horizon Cutoff**: Top `{t_long:.0f}%` are High tier, bottom `{t_long:.0f}%` are Low tier ({tiers['long']['status']}).",
        "",
        "---",
        "",
        "## 6. Historical Performance of Tiers",
        "",
        "> [!WARNING]",
        "> **Optimism Warning**: The cutoffs and the score were chosen using the same historical years shown here, so these hit rates are optimistic. Realized future performance is likely to be lower.",
        "",
        "Performance is reported as **hit rate** (the percentage of selections that beat the median stock in the index):",
        "",
        "| Horizon | Partition | High Tier Hit Rate (90% CI) | Neutral Tier Hit Rate | Low Tier Hit Rate | High Tier Median Excess | Evaluation Note |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for h in ["short", "mid", "long"]:
        h_name = "Short (21d)" if h == "short" else ("Mid (63d)" if h == "mid" else "Long (252d)")
        # Dev
        dev_h = tier_history[h]["dev"]["High"]
        dev_n = tier_history[h]["dev"]["Neutral"]
        dev_l = tier_history[h]["dev"]["Low"]
        ci_str = f"[{dev_h['hit_rate_90_ci'][0]:.1f}%, {dev_h['hit_rate_90_ci'][1]:.1f}%]"
        u_lines.append(f"| **{h_name}** | **Historical Dev** | **{dev_h['hit_rate_pct']:.1f}%** ({ci_str}) | {dev_n['hit_rate_pct']:.1f}% | {dev_l['hit_rate_pct']:.1f}% | {dev_h['median_excess_pct']:+.2f}% | Selected on Dev (n={dev_h['n_periods']}) |")
        
        # Holdout
        if "holdout" in tier_history[h] and "High" in tier_history[h]["holdout"]:
            hold_h = tier_history[h]["holdout"]["High"]
            hold_n = tier_history[h]["holdout"].get("Neutral", {"hit_rate_pct": 0.0})
            hold_l = tier_history[h]["holdout"].get("Low", {"hit_rate_pct": 0.0})
            u_lines.append(f"| **{h_name}** | **Holdout** | **{hold_h['hit_rate_pct']:.1f}%** (small sample) | {hold_n['hit_rate_pct']:.1f}% | {hold_l['hit_rate_pct']:.1f}% | {hold_h['median_excess_pct']:+.2f}% | Holdout (n={hold_h['n_periods']}) |")
            
    u_lines.extend([
        "",
        "---",
        "",
        "## 7. Volatility Categories & Warnings",
        "",
        "### Realized Volatility Categories",
        "Calculated from trailing 63-day annualized return volatility:",
        f"- **Low Volatility**: Annualized volatility $\le$ `{vol_low:.2f}%`.",
        f"- **Medium Volatility**: Annualized volatility between `{vol_low:.2f}%` and `{vol_high:.2f}%`.",
        f"- **High Volatility**: Annualized volatility $>$ `{vol_high:.2f}%`.",
        "",
        "### Liquidity Warning",
        f"- A stock receives a **Liquidity Warning** when its trailing 63 market-calendar trading day participation is at or below `{liq_cutoff:.1f}%` (the 20th percentile of market participation). This warns investors that the stock has experienced frequent non-trading sessions.",
        "",
        "### Thin Trading Warning (Stretched Horizon)",
        f"- Stocks flagged for thin trading: **{', '.join(thin_flag_series)}**.",
        "- For thinly traded stocks, 21 consecutive trading rows span significantly more calendar days than the market median (> 1.5x ratio). All operational windows in Stock Pulse count the stock's own trading observations, meaning calendar holding periods are longer for these companies.",
        "",
        "---",
        "",
        "## 8. Limitations & Key Assumptions",
        "",
        "1. **Survivorship Conditioning**: The historical universe comprises companies surviving into the modern S&P SL20 index.",
        "2. **No Fundamental or Sentiment Data**: The engine evaluates only historical price and volume. Quarterly financial statements, dividend announcements, macroeconomic indicators, and news events are omitted.",
        "3. **Frontier Market Transaction Frictions**: We assume an institutional round-trip trading cost of **200 basis points** (100 bps buy, 100 bps sell) covering brokerage commissions, CSE statutory charges, and bid-ask spread impact.",
        "4. **Own-Trading-Day Horizons**: Observation windows count each stock's own active trading days rather than fixed calendar days.",
        "5. **Frontier Liquidity Concentration**: Trading volume on the Colombo Stock Exchange is heavily concentrated in a small subset of blue-chip counters.",
        "",
        "---",
        "",
        "## 9. Regulatory & Legal Disclaimer",
        "",
        "This platform is provided solely for academic research, education, and technology demonstration. Under no circumstances should any score, tier, or metric produced by Stock Pulse be interpreted as investment recommendations or portfolio advice."
    ])
    
    user_content = "\n".join(u_lines)
    user_file = app_dir / "methodology_user.md"
    user_file.write_text(user_content, encoding="utf-8")
    print(f"Saved {user_file} ({len(user_content)} bytes)")

    # -------------------------------------------------------------------------
    # B. ADMIN METHODOLOGY (data/app/methodology_admin.md)
    # -------------------------------------------------------------------------
    a_lines = [
        "# Stock Pulse: Administrator & Technical Methodology Guide",
        "",
        f"**Document**: `data/app/methodology_admin.md` | **Generated**: {datetime.now().strftime('%Y-%m-%d')} | **Latest Data**: {latest_date}",
        "",
        "---",
        "",
        "## 1. System Architecture & Model Serving Governance",
        "",
        "Stock Pulse implements a walk-forward purged cross-sectional ranking framework over the Colombo Stock Exchange (S&P SL20).",
        "",
        "### Default Served Models (Auto Rule):",
        "- **Short Horizon (21d)**: `ens_rank4` (Equal-weight rank blend of Ridge, Random Forest, LightGBM, and 126d Momentum)",
        "- **Mid Horizon (63d)**: `momentum_126` (Pure 6-month cross-sectional price momentum benchmark)",
        "- **Long Horizon (252d)**: `ens_rank4` (Equal-weight rank blend)",
        "",
        "### Servable Model Registry:",
        "The following 5 models are configured as servable and can be selected manually in the administrative console:",
        "1. `momentum_126` (Baseline)",
        "2. `ridge_core` (L2 Linear)",
        "3. `rf_reg_core` (Bagging Ensemble)",
        "4. `lgbm_reg_core` (Gradient Boosted Trees)",
        "5. `ens_rank4` (Multi-Family Rank Ensemble)",
        "",
        "Research models (`logit_core`, `rf_clf_core`, `lgbm_clf_core`, `lstm_reg_core`, `tabpfn_reg_core`, `tabpfn_clf_core`) are retained in the registry with `servable: false` for governance auditing.",
        "",
        "---",
        "",
        "## 2. Statistical Validation Design & Multiple Testing",
        "",
        "### Walk-Forward Folds",
        "- **Short & Mid Horizons**: 12 development folds (`dev_2008` through `dev_2019`).",
        "- **Long Horizon**: 10 development folds (`dev_2010` through `dev_2019`). Denominator is strictly 10; long horizon requires 2-year feature formation, so 2008-2009 were never evaluated.",
        "- **Stress Partition**: 2020 COVID shock ($n_{\\text{eff}}$ displayed).",
        "- **Holdout Partition**: 2024–2025 out-of-sample evaluation ($n_{\\text{eff}}$ displayed).",
        "",
        "### Dual Random Null Band & Significance",
        "- Null bands are constructed using 1,000 Monte Carlo block-constant and linear scorer simulations.",
        "- **Short Null Band**: `[-0.0374, +0.0368]`",
        "- **Mid Null Band**: `[-0.0532, +0.0558]`",
        "- **Long Null Band**: `[-0.1252, +0.1231]`",
        "",
        "### Holm-Bonferroni Correction",
        "- Across all 69 family evaluations, after step-down Holm correction, no model retains statistical significance at $\\alpha = 0.05$.",
        "- All raw significant results are classified as **suggestive, not confirmed**.",
        "",
        "---",
        "",
        "## 3. Evidence-Level Rules (Post-Hoc Tightening)",
        "",
        "### Evaluated Usability Criteria:",
        "1. **Criterion 1**: Dev Mean Rank IC outside corrected null band ($p \\le 0.05$).",
        "2. **Criterion 2**: Beats best naive baseline in $\\ge 2/3$ dev folds.",
        "3. **Criterion 3**: Style-neutralized Rank IC $> 0.000$.",
        "",
        "### Tightened Conservative Rule (`post_hoc: true`):",
        "- **RED**: Criterion 1 fails (inside null band).",
        "- **AMBER**: Criterion 1 passes and $\\ge 1$ other criterion passes.",
        "- **GREEN**: All 3 criteria pass.",
        "",
        "### Empirical Justification:",
        "- Random linear scorer draws (null type b) produce positive style-neutralized IC in **47.5%** (Short), **54.0%** (Mid), and **52.5%** (Long) of simulations.",
        "- Criterion 3 alone represents weak random baseline noise (~50% chance of passing) and cannot lift a candidate out of 'no reliable evidence'.",
        "",
        "---",
        "",
        "## 4. Liquidity & Thin-Trading Governance",
        "",
        "- **`zero_trade_share_63d`**: Identified as uninformative (constant 0.0) due to trade-day filtering in cleaned parquet. Marked as constant; models not retrained.",
        "- **Market Calendar**: Constructed requiring $\\ge 10$ active ranking series.",
        "- **`trade_day_share_63m`**: Evaluated as trailing 63 market-calendar days traded within segment.",
        "- **Liquidity Warning Threshold**: Set to 20th percentile on development rankable rows (`0.8889` or 88.89%).",
        "- **Thin Trading Flag**: `HNB.N` has 21-row median span ratio of 1.533x (> 1.5x threshold) and is flagged as stretched horizon.",
        "",
        "---",
        "",
        "## 5. Model Registry Summary",
        "",
        "| Model ID | Family | Servable | Short IC (t) | Mid IC (t) | Long IC (t) | Evidence (S/M/L) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for m_id, m_data in registry.items():
        s_ic = m_data["horizons"]["short"]["dev_mean_ic"]
        s_t = m_data["horizons"]["short"]["fold_t_stat"]
        m_ic = m_data["horizons"]["mid"]["dev_mean_ic"]
        m_t = m_data["horizons"]["mid"]["fold_t_stat"]
        l_ic = m_data["horizons"]["long"]["dev_mean_ic"]
        l_t = m_data["horizons"]["long"]["fold_t_stat"]
        
        s_str = f"{s_ic:+.4f} ({s_t:+.2f})" if s_ic is not None and s_t is not None else "N/A"
        m_str = f"{m_ic:+.4f} ({m_t:+.2f})" if m_ic is not None and m_t is not None else "N/A"
        l_str = f"{l_ic:+.4f} ({l_t:+.2f})" if l_ic is not None and l_t is not None else "N/A"
        
        ev_s = m_data["horizons"]["short"]["evidence_level"] or "N/A"
        ev_m = m_data["horizons"]["mid"]["evidence_level"] or "N/A"
        ev_l = m_data["horizons"]["long"]["evidence_level"] or "N/A"
        
        a_lines.append(f"| `{m_id}` | {m_data['family']} | {'Yes' if m_data['servable'] else 'No'} | {s_str} | {m_str} | {l_str} | {ev_s}/{ev_m}/{ev_l} |")
        
    admin_content = "\n".join(a_lines)
    admin_file = app_dir / "methodology_admin.md"
    admin_file.write_text(admin_content, encoding="utf-8")
    print(f"Saved {admin_file} ({len(admin_content)} bytes)")
    
    # -------------------------------------------------------------------------
    # C. Remove deprecated single methodology.md
    # -------------------------------------------------------------------------
    old_methodology = app_dir / "methodology.md"
    if old_methodology.exists():
        old_methodology.unlink()
        print(f"Removed deprecated {old_methodology}")


def render_methodology():
    render_methodologies()
    return base_dir / "data/app/methodology_user.md"


if __name__ == "__main__":
    render_methodologies()

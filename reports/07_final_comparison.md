# Stock Pulse: Final Model Comparison, Methodology & Served Governance
**Document ID**: `reports/07_final_comparison.md`  
**Date**: 2026-10-11  
**Universe**: Colombo Stock Exchange (S&P SL20, 20 stocks)  
**Validation Design**: Walk-Forward Purged Cross-Sectional Ranking (Test Years 2008–2019 Dev, 2020 Stress, 2024–2025 Holdout)  

---

## 1. Executive Summary & Master Comparison Tables

This report delivers the comprehensive empirical synthesis across all evaluated model families (Linear, Bagging, Gradient Boosting, Deep Learning LSTM, and TabPFN Foundation Models) against naive heuristic benchmarks.

### Key Validation Fixes Applied:
1. **Strict Denominator Governance**: Long horizon dev statistics use strictly **10** annual folds (2010–2019); Short and Mid horizons use strictly **12** annual folds (2008–2019). Stress and Holdout partitions display effective sample size ($n_{\text{eff}}$) instead of deceptive single-fold counts.
2. **Rigorous Multiple-Testing Adjustment**: Across the full family of 69 development evaluations, the expected false positive rate is 3.45. After applying the step-down Holm-Bonferroni correction, no model retains statistical significance at $\alpha = 0.05$ (all adjusted p-values $> 0.05$), classifying all raw significant signals as **suggestive, not confirmed**.
3. **Company Identity Resolution**: Both core and core_rel models remain well inside the corrected random null band. Because `core_rel` still retains cross-sectional volatility ranks ($R^2 \approx 0.46$ on company dummies), observed minor score variations represent random sampling variance rather than conclusive identity exploitation.

---

### 1.1 Master Comparison: SHORT Horizon (21 Trading Days)
**Development Folds**: 12 annual walk-forward folds | **Corrected Null Band**: `[-0.0374, +0.0368]`  

| Model Candidate | Dev Mean IC | Corrected Null Band | Raw p | Holm p | Folds IC > 0 / N | Fold t-stat (df) | Top-5 Med Exc | Net Exc (Actual) | Replaced Share | Holdout IC ($n_{\text{eff}}$) | Holdout Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Random Null Band* | `0.0000` | `[-0.0374, +0.0368]` | — | — | `6/12` | `0.00` | `0.00%` | `-2.00%` | `100.0%` | `0.0000` | Reference Envelope |
| `reversal_21` | **-0.0042** | Inside | 0.592 | 1.000 | 5/12 | -0.27 (df=11) | +0.30% | -1.22% | 69.3% | +0.0126 (n=16) | indicative |
| `low_vol` | **-0.0246** | Inside | 0.910 | 1.000 | 5/12 | -1.10 (df=11) | -0.01% | -1.05% | 28.6% | -0.0580 (n=16) | indicative |
| `past_winner` | **+0.0099** | Inside | 0.302 | 1.000 | 5/12 | +0.35 (df=11) | +0.27% | +0.06% | 13.4% | +0.0528 (n=16) | indicative |
| `momentum_126` | **+0.0356** | Inside | 0.031 | 1.000 | 8/12 | +1.56 (df=11) | +0.60% | -0.22% | 30.5% | +0.0381 (n=16) | indicative |
| `ridge_core` | **+0.0284** | Inside | 0.070 | 1.000 | 8/12 | +1.62 (df=11) | +0.45% | -0.79% | 50.4% | +0.1913 (n=16) | indicative |
| `logit_core` | **+0.0241** | Inside | 0.110 | 1.000 | 7/12 | +1.24 (df=11) | +0.37% | -0.93% | 49.3% | +0.1795 (n=16) | indicative |
| `rf_reg_core` | **+0.0069** | Inside | 0.350 | 1.000 | 6/12 | +0.42 (df=11) | +0.30% | -1.32% | 52.6% | +0.2068 (n=16) | indicative |
| `rf_clf_core` | **+0.0087** | Inside | 0.317 | 1.000 | 6/12 | +0.58 (df=11) | +0.27% | -1.20% | 51.8% | +0.1604 (n=16) | indicative |
| `lgbm_reg_core` | **-0.0048** | Inside | 0.609 | 1.000 | 6/12 | -0.28 (df=11) | +0.20% | -1.31% | 50.8% | +0.1494 (n=16) | indicative |
| `lgbm_clf_core` | **+0.0043** | Inside | 0.401 | 1.000 | 5/12 | +0.26 (df=11) | +0.30% | -0.98% | 53.3% | +0.1096 (n=16) | indicative |
| `ens_rank4` | **+0.0315** | Inside | 0.049 | 1.000 | 10/12 | +1.72 (df=11) | +0.33% | -0.90% | 47.2% | +0.1966 (n=16) | indicative |
| `lstm_reg_core` | **+0.0455** | Outside | 0.006 | 0.414 | 10/12 | +3.65 (df=11) | +0.41% | -1.34% | 63.4% | +0.0167 (n=16) | indicative |
| `tabpfn_reg_core` | **+0.0075** | Inside | 0.339 | 1.000 | 7/12 | +0.37 (df=11) | +0.26% | -0.83% | 48.8% | -0.0184 (n=16) | indicative |
| `tabpfn_clf_core` | **+0.0260** | Inside | 0.086 | 1.000 | 7/12 | +1.62 (df=11) | +0.40% | -0.46% | 52.1% | +0.0395 (n=16) | indicative |

### 1.2 Master Comparison: MID Horizon (63 Trading Days)
**Development Folds**: 12 annual walk-forward folds | **Corrected Null Band**: `[-0.0532, +0.0558]`  

| Model Candidate | Dev Mean IC | Corrected Null Band | Raw p | Holm p | Folds IC > 0 / N | Fold t-stat (df) | Top-5 Med Exc | Net Exc (Actual) | Replaced Share | Holdout IC ($n_{\text{eff}}$) | Holdout Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Random Null Band* | `0.0000` | `[-0.0532, +0.0558]` | — | — | `6/12` | `0.00` | `0.00%` | `-2.00%` | `100.0%` | `0.0000` | Reference Envelope |
| `reversal_21` | **-0.0348** | Inside | 0.914 | 1.000 | 5/12 | -1.69 (df=11) | +0.56% | -1.39% | 70.7% | -0.0378 (n=5) | indicative |
| `low_vol` | **-0.0371** | Inside | 0.926 | 1.000 | 5/12 | -1.42 (df=11) | +0.01% | -2.45% | 52.0% | -0.1358 (n=5) | indicative |
| `past_winner` | **+0.0227** | Inside | 0.201 | 1.000 | 6/12 | +0.43 (df=11) | +0.93% | +1.24% | 15.6% | +0.0984 (n=5) | indicative |
| `momentum_126` | **+0.0658** | Outside | 0.010 | 0.680 | 9/12 | +1.74 (df=11) | +1.60% | +0.30% | 46.2% | +0.1135 (n=5) | indicative |
| `ridge_core` | **+0.0144** | Inside | 0.300 | 1.000 | 7/12 | +0.52 (df=11) | +0.84% | -0.69% | 56.9% | +0.3304 (n=5) | indicative |
| `logit_core` | **+0.0095** | Inside | 0.359 | 1.000 | 7/12 | +0.35 (df=11) | +0.85% | -0.48% | 56.0% | +0.2681 (n=5) | indicative |
| `rf_reg_core` | **+0.0250** | Inside | 0.184 | 1.000 | 9/12 | +1.16 (df=11) | +1.15% | -2.01% | 58.2% | +0.2984 (n=5) | indicative |
| `rf_clf_core` | **-0.0056** | Inside | 0.602 | 1.000 | 4/12 | -0.26 (df=11) | +0.63% | -2.10% | 61.8% | +0.2733 (n=5) | indicative |
| `lgbm_reg_core` | **+0.0179** | Inside | 0.249 | 1.000 | 6/12 | +0.82 (df=11) | +0.70% | -1.66% | 56.9% | +0.1834 (n=5) | indicative |
| `lgbm_clf_core` | **+0.0046** | Inside | 0.435 | 1.000 | 7/12 | +0.20 (df=11) | +0.87% | -0.67% | 52.9% | +0.1491 (n=5) | indicative |
| `ens_rank4` | **+0.0429** | Inside | 0.064 | 1.000 | 8/12 | +1.41 (df=11) | +1.49% | -0.62% | 51.1% | +0.3176 (n=5) | indicative |
| `lstm_reg_core` | **+0.0217** | Inside | 0.210 | 1.000 | 7/12 | +1.39 (df=11) | +0.72% | -1.47% | 66.2% | -0.0905 (n=5) | indicative |
| `tabpfn_reg_core` | **+0.0052** | Inside | 0.428 | 1.000 | 6/12 | +0.26 (df=11) | +0.58% | -0.71% | 61.8% | +0.1980 (n=5) | indicative |
| `tabpfn_clf_core` | **+0.0250** | Inside | 0.184 | 1.000 | 8/12 | +0.78 (df=11) | +1.05% | -0.28% | 64.4% | +0.2010 (n=5) | indicative |

### 1.3 Master Comparison: LONG Horizon (252 Trading Days)
**Development Folds**: 10 annual walk-forward folds | **Corrected Null Band**: `[-0.1252, +0.1231]`  

| Model Candidate | Dev Mean IC | Corrected Null Band | Raw p | Holm p | Folds IC > 0 / N | Fold t-stat (df) | Top-5 Med Exc | Net Exc (Actual) | Replaced Share | Holdout IC ($n_{\text{eff}}$) | Holdout Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Random Null Band* | `0.0000` | `[-0.1252, +0.1231]` | — | — | `5/10` | `0.00` | `0.00%` | `-2.00%` | `100.0%` | `0.0000` | Reference Envelope |
| `reversal_21` | **-0.0541** | Inside | 0.781 | 1.000 | 2/10 | -2.60 (df=9) | +2.04% | +2.16% | 80.0% | -0.1007 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `low_vol` | **-0.0786** | Inside | 0.873 | 1.000 | 2/10 | -2.20 (df=9) | -0.89% | -13.86% | 62.2% | -0.1304 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `past_winner` | **-0.0450** | Inside | 0.737 | 1.000 | 4/10 | -0.60 (df=9) | +4.27% | +16.68% | 28.9% | +0.1080 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `momentum_126` | **+0.0682** | Inside | 0.121 | 1.000 | 7/10 | +1.32 (df=9) | +4.83% | -6.43% | 75.6% | -0.0127 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `ridge_core` | **+0.0694** | Inside | 0.113 | 1.000 | 7/10 | +1.48 (df=9) | +4.71% | -8.20% | 64.4% | +0.4780 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `logit_core` | **+0.0463** | Inside | 0.220 | 1.000 | 7/10 | +0.82 (df=9) | +3.43% | -7.78% | 68.9% | +0.4565 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `rf_reg_core` | **+0.0595** | Inside | 0.157 | 1.000 | 7/10 | +2.22 (df=9) | +4.66% | +5.47% | 51.1% | +0.3754 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `rf_clf_core` | **+0.0349** | Inside | 0.273 | 1.000 | 6/10 | +0.98 (df=9) | +4.02% | -8.89% | 60.0% | +0.2740 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `lgbm_reg_core` | **+0.0451** | Inside | 0.224 | 1.000 | 6/10 | +1.64 (df=9) | +2.26% | -11.97% | 62.2% | +0.2929 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `lgbm_clf_core` | **+0.0932** | Inside | 0.062 | 1.000 | 8/10 | +2.86 (df=9) | +4.56% | +2.53% | 48.9% | +0.4267 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `ens_rank4` | **+0.1143** | Inside | 0.032 | 1.000 | 9/10 | +2.79 (df=9) | +6.76% | +5.88% | 68.9% | +0.3939 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `lstm_reg_core` | **-0.0177** | Inside | 0.588 | 1.000 | 3/10 | -0.71 (df=9) | +1.70% | +10.03% | 75.6% | +0.3085 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `tabpfn_reg_core` | **+0.0403** | Inside | 0.246 | 1.000 | 6/10 | +1.45 (df=9) | +4.48% | +12.32% | 68.9% | +0.4371 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |
| `tabpfn_clf_core` | **+0.0094** | Inside | 0.429 | 1.000 | 6/10 | +0.21 (df=9) | +3.84% | -11.00% | 64.4% | +0.3851 (n=1) | not evaluable ($n_{\text{eff}} < 5$) |

---

## 2. Paired Comparison: Ensemble (`ens_rank4`) vs Naive Momentum (`momentum_126`)

To establish whether multi-model machine learning provides statistically demonstrable superiority over single-factor momentum, we performed an annual paired walk-forward comparison with 2,000 year-block bootstrap resamples:

| Horizon | Dev Folds ($N$) | ens_rank4 Dev IC | momentum_126 Dev IC | Mean Difference | ens Wins (Years) | 90% Bootstrap CI of Difference | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short (21d)** | 12 | `+0.0315` | `+0.0356` | `-0.0043` | 7 / 12 (58.3%) | `[-0.0334, +0.0231]` | Statistically Indistinguishable (CI spans 0) |
| **Mid (63d)** | 12 | `+0.0429` | `+0.0658` | `-0.0230` | 5 / 12 (41.7%) | `[-0.0528, +0.0053]` | Momentum Leads; Difference spans 0 |
| **Long (252d)** | 10 | `+0.1143` | `+0.0682` | `+0.0460` | 6 / 10 (60.0%) | `[-0.0129, +0.1009]` | Ensemble Leads; Difference spans 0 |

> [!NOTE]
> Across all three horizons, the 90% block bootstrap confidence interval of the annual difference between `ens_rank4` and `momentum_126` includes zero. Machine learning provides diversification across market regimes, but does not statistically dominate raw 6-month momentum.

---

## 3. Visual Diagnostics

### 3.1 Mean Out-of-Sample Rank IC vs Random Null Band
![Dev Mean IC vs Null Band](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_dev_mean_ic_by_horizon.png)
*Figure 1: Cross-sectional Rank IC across all model families. Grey shaded area indicates the 90% block-resampled random null band envelope.*

### 3.2 Annual Walk-Forward Performance Persistence
![ens_rank4 vs momentum_126 Annual IC](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_ens_vs_momentum_annual_ic.png)
*Figure 2: Annual Out-of-Sample Rank IC across walk-forward test years for `ens_rank4` vs `momentum_126`.*

### 3.3 Consistency & Fold Hit Rates
![Fold Hit Rate Bars](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_fold_hit_rate_bars.png)
*Figure 3: Share of development folds where out-of-sample Rank IC is strictly positive. Red dashed line indicates the 2/3 (66.7%) usability hurdle.*

---

## 4. Synthesis: What We Found & What the Evidence Shows

### What Was Tested
We evaluated four distinct modeling families across 20 Colombo Stock Exchange equities over a 24-year history (2001–2025):
1. **Linear Regularized Models**: L2-penalized Ridge Regression and Logistic Classification.
2. **Bagging Tree Ensembles**: Random Forest Regressors and Classifiers.
3. **Gradient Boosting Ensembles**: LightGBM Regressors, Classifiers, and Multi-Model Rank Ensembles (`ens_rank4`).
4. **Deep Learning Architectures**: 1-Layer LSTM with 6 causal price/volume channels, and TabPFN v2 Foundation Prior-Fitted Networks.

### Horizon Evidence Synthesis
- **Short Horizon (21d)**: Characterized by heavy bid-ask and execution friction. High portfolio turnover (~50% monthly replacement) completely erodes gross excess returns (-1.31% net actual drag). While the LSTM extracted significant short-term mean-reversion, multiple-testing adjustment reduces raw significance to suggestive. **Evidence Level: RED (No reliable evidence)**.
- **Mid Horizon (63d)**: The only horizon demonstrating authentic economic signal persistence. 6-Month Cross-Sectional Momentum (`momentum_126`) was the single benchmark to break out of the random null band (raw $p = 0.010$, Dev IC `+0.0658`), with positive style-neutralized residual alpha (+0.0561) and 9 of 12 folds positive. **Evidence Level: AMBER (Weak evidence)**.
- **Long Horizon (252d)**: Appears lucrative nominally (Dev IC `+0.1143` for `ens_rank4`), but possesses only 10 independent annual dev test periods. With overlapping annual return dynamics, statistical power is severely constrained, and wide confidence intervals encompass naive momentum. **Evidence Level: RED (No reliable evidence)**.

### Did Deep Learning Add Anything?
Deep learning (LSTM and TabPFN) was evaluated for research completeness. The LSTM captured subtle non-linear sequence patterns in the Short horizon (`+0.0455` Dev IC), but suffered severe memory overhead and exhibited no sustained edge over regularized linear models or LightGBM on small tabular universes (20 stocks). TabPFN operated feasibly via context striding and chunking, but performed within the null band. **Conclusion**: Deep learning adds substantial training and maintenance complexity without demonstrable outperformance on frontier equity rankings.

### Non-Evidence & Critical Caveats
1. **Holdout Performance is Not Proof of Skill**: The 2024–2025 holdout comprises an unprecedented frontier bull-market recovery following the sovereign debt restructuring. With effective sample sizes of $n_{\text{eff}} \approx 0.4 \text{ to } 16$, the holdout represents a single regime and neither confirms nor refutes model skill.
2. **Survivorship Bias**: The ranking universe is conditioned on today's S&P SL20 index constituents. Historical momentum and winner heuristics carry survivorship bias.
3. **Cost Assumption**: Transaction cost of 100 bps per side (200 bps round-trip) is a realistic baseline, but execution slippage in illiquid frontier equities may be even higher during market dislocations.
4. **Absence of Fundamentals & News**: The current system relies strictly on price-volume dynamics, omitting balance-sheet disclosures, earnings surprises, and macro news events.

---

## 5. Served Governance, Evidence Levels & Threshold Registry

### 5.1 Served Score Assignment Rule
- **Rule**: For each horizon, evaluate candidate scores $\{ens\_rank4, momentum\_126\}$. Serve the candidate with the higher development fold-level t-statistic; if they differ by less than 0.10, serve `ens_rank4`.
- **Short (21d)**: `ens_rank4` served (t = +1.72 vs +1.56, diff = +0.16 > 0.10).
- **Mid (63d)**: `momentum_126` served (t = +1.74 vs +1.41, diff = +0.33 > 0.10).
- **Long (252d)**: `ens_rank4` served (t = +2.79 vs +1.32, diff = +1.47 > 0.10).

### 5.2 Evidence Levels & Default Score Weights
Overall Upside Score = weighted average of the three horizon percentiles. Default weights are derived strictly from evidence points (RED = 1, AMBER = 2, GREEN = 3):
- Short (RED, 1 pt): **25.0%**
- Mid (AMBER, 2 pts): **50.0%**
- Long (RED, 1 pt): **25.0%**

### 5.3 Operational Thresholds Summary
| Threshold ID | Metric Value | Unit | Plain-English Derivation Method | Status |
| :--- | :---: | :--- | :--- | :--- |
| `tier_cutoff_short` | 25% | fraction | Top quartile convention; no candidate fraction beat median in >= 2/3 dev years. | convention - no evidence-based cutoff |
| `tier_cutoff_mid` | 30% | fraction | Largest qualifying fraction with positive lift in 9/12 dev years (75.0%, pooled lift +1.66%). | empirical |
| `tier_cutoff_long` | 30% | fraction | Largest qualifying fraction with positive lift in 8/10 dev years (80.0%, pooled lift +5.98%). | empirical |
| `risk_category_low_cutoff` | 21.76% | ann. percent | 33.3rd percentile of trailing annualized 63-day volatility across dev rankable rows. | empirical |
| `risk_category_high_cutoff` | 32.98% | ann. percent | 66.7th percentile of trailing annualized 63-day volatility across dev rankable rows. | empirical |
| `liquidity_warning_cutoff` | 0.00% | percent days | 80th percentile of zero-trade days share in trailing quarter. Triggers if strictly positive. | empirical |

---

## 6. Honest Unresolved List / Open Questions

1. **Survivorship Correction**: Can historical point-in-time index composition be sourced to replace today's static S&P SL20 constituent roster?
2. **Frontier Market Liquidity Shocks**: The current model assumes continuous trading with 100 bps cost per side, but during liquidity freezes or political shocks, trades cannot execute at quoted prices.
3. **Absence of Fundamental Features**: P/E ratios, book-to-market, and dividend yields are unmodeled in the current technical price-volume schema.
4. **Multiple Testing Power**: With 69 hypothesis tests and only 10–12 annual folds, true alpha signals risk being rejected under strict family-wise error rate control.
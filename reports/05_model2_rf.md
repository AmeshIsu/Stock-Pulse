# Stock Pulse: Model 2 — Random Forest Benchmark & Evaluation Repair

**Document**: `reports/05_model2_rf.md`  
**Purpose**: Rigorous walk-forward benchmark of non-linear tree ensembles (`RandomForestRegressor` and `RandomForestClassifier`), corrected random null band evaluation, effective sample size gating, factor tilt diagnostics, and realistic turnover-adjusted transaction frictions.

> [!WARNING]
> **Sample Limitations & Survivorship Caveat**:
> 1. **Index Survivorship Bias**: The S&P SL20 universe consists of companies that survived and prospered to be index members today. Backward evaluations—particularly over the long horizon—are inherently flattered by this survivorship selection.
> 2. **Macroeconomic Regime Shift**: The 2024–2025 holdout covers a single post-default recovery cycle characterized by aggressive disinflation and currency stabilization, comprising very few independent periods ($n_{	ext{eff}} = 1$ for long horizon; $n_{	ext{eff}} = 5$ for mid). It serves as a sanity check rather than conclusive proof of generalization.
> 3. **Statistical Sample Size**: With only ~10 independent years of development history on the Colombo Stock Exchange, long-horizon evidence is constrained by small sample degrees of freedom ($n_{	ext{eff}} = 10$, marked indicative only).

---

## 1. Evaluation Fixes & Null Band Repair (Part A Summary)

### Root-Cause of Old Null Band Narrowness:
The previous random null band drew fresh random permutations for every (draw, date) independently. Because real models produce persistent scores across days and forward targets overlap over $h$ trading days, the effective number of independent cross-sectional observations across a partition is approximately $(	ext{trading days} / h)$, not the total number of calendar days. This made the old null band 3x to 12x too narrow, generating spurious outside-null rejections ($p = 0.000$) that contradicted fold-level inconsistencies where models failed in multiple annual periods.

### Side-by-Side Random Null Band Comparison (1,000 Draws Each):

| Horizon | Partition | Old Band (Daily i.i.d.) | (a) Block-Constant Null | (b) Random Linear Null | Corrected Null Band (Wider Envelope) | Width Expansion Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SHORT (21d)** | **Dev** | `[-0.0090, +0.0100]` | `[-0.0305, +0.0321]` | `[-0.0374, +0.0368]` | **`[-0.0374, +0.0368]`** | **3.91x** |
| **SHORT (21d)** | *Stress* | `[-0.0335, +0.0365]` | `[-0.0931, +0.0911]` | `[-0.1476, +0.1573]` | **`[-0.1476, +0.1573]`** | **4.36x** |
| **SHORT (21d)** | *Holdout* | `[-0.0267, +0.0262]` | `[-0.0886, +0.0906]` | `[-0.1611, +0.1398]` | **`[-0.1611, +0.1398]`** | **5.69x** |
| **MID (63d)** | **Dev** | `[-0.0094, +0.0102]` | `[-0.0532, +0.0558]` | `[-0.0534, +0.0545]` | **`[-0.0532, +0.0558]`** | **5.57x** |
| **MID (63d)** | *Stress* | `[-0.0324, +0.0359]` | `[-0.2055, +0.1817]` | `[-0.1519, +0.1528]` | **`[-0.2055, +0.1817]`** | **5.67x** |
| **MID (63d)** | *Holdout* | `[-0.0263, +0.0273]` | `[-0.1650, +0.1658]` | `[-0.2772, +0.2602]` | **`[-0.2772, +0.2602]`** | **10.03x** |
| **LONG (252d)** | **Dev** | `[-0.0101, +0.0104]` | `[-0.1252, +0.1231]` | `[-0.0746, +0.0776]` | **`[-0.1252, +0.1231]`** | **12.10x** |
| **LONG (252d)** | *Holdout* | `[-0.0508, +0.0476]` | `[-0.4294, +0.4185]` | `[-0.3202, +0.3170]` | **`[-0.4294, +0.4185]`** | **8.62x** |

---

## 2. Usability Criteria & Revised Verdicts

A model is classified as **usable** only if all three mandatory criteria are satisfied:
1. **Dev Rank IC is outside the random null band** (empirical permutation $p < 0.05$ vs. wider persistence-corrected null band).
2. **Beats the best naive baseline (`momentum_126`) in at least two-thirds ($\ge 67\%$) of development folds**.
3. **Neutralized IC stays positive** when residualized against market volatility, liquidity, and momentum tilts.

### Summary of Verdict Changes:
- **SHORT Horizon**: `rf_reg_core`, `rf_clf_core`, `ridge_core`, and `logit_core` all changed from AMBER to **RED**. Under the corrected null band `[-0.0374, +0.0368]`, their Dev ICs (+0.007 to +0.028) fall entirely inside the random null.
- **MID Horizon**: `rf_reg_core`, `ridge_core`, and `logit_core` changed from AMBER to **RED**. Under the corrected null band `[-0.0532, +0.0558]`, none of the ML models exceed the null band. Only naive `momentum_126` (+0.0658) is outside the null band ($p = 0.010$).
- **LONG Horizon**: `rf_reg_core` changed from AMBER to **RED**. $n_{	ext{eff}} = 10 \implies$ marked *indicative only*. Dev IC (+0.0595) falls inside the null band `[-0.1252, +0.1231]`. Long holdout has $n_{	ext{eff}} = 1 < 5 \implies$ **Not Evaluable** (verdict and p-value suppressed).

### Horizon Evaluation Detail:
- **SHORT Horizon (21d)**: **RED** (0 of 3 criteria met)
  - *(1) Null Band*: **FAILED** (Dev IC `+0.007` lies well inside corrected null band `[-0.0374, +0.0368]`, $p = 0.350$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in only **5 / 12** dev folds = 41.7% < 66.7%).
  - *(3) Neutralized IC*: Residual IC `+0.012` > 0, but nominal model fails null band.
  - *Turnover & Friction*: Replaced share is 52.55%. Realized net excess is -1.32% under actual turnover (-2.27% under full).

- **MID Horizon (63d)**: **RED** (0 of 3 criteria met)
  - *(1) Null Band*: **FAILED** (Dev IC `+0.025` lies inside corrected null band `[-0.0532, +0.0558]`, $p = 0.184$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in **6 / 12** dev folds = 50.0% < 66.7%). Best naive baseline `momentum_126` achieves higher mean Dev IC (`+0.066` vs `+0.025`) and beats the null band ($p = 0.010$).
  - *(3) Neutralized IC*: Residual IC `+0.021` > 0, but nominal model fails null band.
  - *Turnover & Friction*: Replaced share is 58.22%. Realized net excess is -2.01% under actual turnover (-2.83% under full).

- **LONG Horizon (252d)**: **RED** (0 of 3 criteria met, indicative only)
  - *(1) Null Band*: **FAILED / INDICATIVE ONLY** ($n_{	ext{eff}} = 10$, Dev IC `+0.060` lies inside corrected null band `[-0.1252, +0.1231]`, $p = 0.157$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in **6 / 10** dev folds = 60.0% < 66.7%). `momentum_126` achieves mean Dev IC `+0.068` vs RF `+0.060`.
  - *(3) Neutralized IC*: Residual IC `+0.026` > 0.
  - *Holdout Gating*: $n_{	ext{eff}} = 1 < 5 \implies$ **Not Evaluable** (verdict and p-value suppressed).

---

## 3. Candidate Performance Tables with Turnover & Gating

### Realistic Friction Modeling:
Friction is evaluated under **100 bps per side (200 bps round-trip)**:
- **Full Turnover**: Conservative baseline assuming 100% portfolio liquidation every rebalance.
- **Actual Turnover**: Proportionate friction: `2 x 100 bps x (Share of Top-5 Replaced)`.
- **Gating**: $n_{	ext{eff}} < 5 \implies$ marked *not evaluable (n_eff < 5)* with IC verdict suppressed; $5 \le n_{	ext{eff}} < 12 \implies$ marked *indicative only*.

### Horizon: `SHORT` (21 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null? | Perm. p-val | $n_{	ext{eff}}$ | Gating Status | Folds IC > 0 | Net (Full) | Net (Actual) | Avg Replaced Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2008-19)** | +0.007 | No | 0.350 | 138 | evaluable | 6 / 12 | -2.27% | -1.32% | 52.6% |
| `rf_reg_core` | *Stress (2020)* | +0.018 | No | 0.453 | 10 | indicative only | 1 / 1 | -1.25% | -0.73% | 71.1% |
| `rf_reg_core` | *Holdout (2024-25)* | +0.207 | Yes | 0.000 | 16 | evaluable | 1 / 1 | -0.00% | +1.00% | 46.7% |
| `rf_reg_full` | **Dev (2008-19)** | -0.001 | No | 0.530 | 138 | evaluable | 5 / 12 | -2.28% | -1.26% | 48.5% |
| `rf_reg_full` | *Stress (2020)* | +0.036 | No | 0.384 | 10 | indicative only | 1 / 1 | +0.54% | +1.18% | 64.4% |
| `rf_reg_full` | *Holdout (2024-25)* | +0.050 | No | 0.252 | 10 | indicative only | 1 / 1 | -0.96% | +0.12% | 40.0% |
| `rf_clf_core` | **Dev (2008-19)** | +0.009 | No | 0.317 | 138 | evaluable | 6 / 12 | -2.16% | -1.20% | 51.8% |
| `rf_clf_core` | *Stress (2020)* | -0.050 | No | 0.739 | 10 | indicative only | 0 / 1 | -1.41% | -0.93% | 73.3% |
| `rf_clf_core` | *Holdout (2024-25)* | +0.160 | Yes | 0.015 | 16 | evaluable | 1 / 1 | -0.28% | +0.80% | 42.7% |
| `rf_clf_full` | **Dev (2008-19)** | -0.005 | No | 0.602 | 138 | evaluable | 6 / 12 | -2.29% | -1.30% | 50.2% |
| `rf_clf_full` | *Stress (2020)* | +0.017 | No | 0.455 | 10 | indicative only | 1 / 1 | +1.23% | +1.75% | 71.1% |
| `rf_clf_full` | *Holdout (2024-25)* | +0.146 | Yes | 0.022 | 10 | indicative only | 1 / 1 | +0.29% | +1.37% | 40.0% |
| `ridge_core` | **Dev (2008-19)** | +0.028 | No | 0.070 | 138 | evaluable | 8 / 12 | -1.77% | -0.79% | 50.4% |
| `ridge_core` | *Stress (2020)* | -0.027 | No | 0.645 | 10 | indicative only | 0 / 1 | -1.76% | -0.56% | 33.3% |
| `ridge_core` | *Holdout (2024-25)* | +0.191 | Yes | 0.001 | 16 | evaluable | 1 / 1 | -0.72% | +0.18% | 52.0% |
| `logit_core` | **Dev (2008-19)** | +0.024 | No | 0.110 | 138 | evaluable | 7 / 12 | -1.94% | -0.93% | 49.3% |
| `logit_core` | *Stress (2020)* | -0.032 | No | 0.668 | 10 | indicative only | 0 / 1 | -2.49% | -1.37% | 37.8% |
| `logit_core` | *Holdout (2024-25)* | +0.180 | Yes | 0.004 | 16 | evaluable | 1 / 1 | -0.52% | +0.33% | 54.7% |
| `momentum_126_core` | **Dev (2008-19)** | +0.036 | No | 0.031 | 138 | evaluable | 8 / 12 | -1.60% | -0.22% | 30.5% |
| `momentum_126_core` | *Stress (2020)* | +0.092 | No | 0.176 | 10 | indicative only | 1 / 1 | +1.38% | +2.70% | 26.7% |
| `momentum_126_core` | *Holdout (2024-25)* | +0.038 | No | 0.304 | 16 | evaluable | 1 / 1 | -1.79% | -0.54% | 33.3% |
| `momentum_63_core` | **Dev (2008-19)** | +0.019 | No | 0.166 | 138 | evaluable | 5 / 12 | -1.70% | -0.52% | 40.7% |
| `momentum_63_core` | *Stress (2020)* | +0.018 | No | 0.453 | 10 | indicative only | 1 / 1 | -2.19% | -1.23% | 46.7% |
| `momentum_63_core` | *Holdout (2024-25)* | -0.025 | No | 0.590 | 16 | evaluable | 0 / 1 | -2.82% | -1.80% | 45.3% |
| `reversal_21_core` | **Dev (2008-19)** | -0.004 | No | 0.592 | 138 | evaluable | 5 / 12 | -1.83% | -1.22% | 69.3% |
| `reversal_21_core` | *Stress (2020)* | -0.003 | No | 0.534 | 10 | indicative only | 0 / 1 | -2.04% | -1.64% | 77.8% |
| `reversal_21_core` | *Holdout (2024-25)* | +0.013 | No | 0.407 | 16 | evaluable | 1 / 1 | -1.07% | -0.47% | 68.0% |
| `past_winner_core` | **Dev (2008-19)** | +0.010 | No | 0.302 | 138 | evaluable | 5 / 12 | -1.66% | +0.06% | 13.4% |
| `past_winner_core` | *Stress (2020)* | -0.045 | No | 0.716 | 10 | indicative only | 0 / 1 | -2.65% | -0.93% | 4.4% |
| `past_winner_core` | *Holdout (2024-25)* | +0.053 | No | 0.244 | 16 | evaluable | 1 / 1 | -2.85% | -1.02% | 2.7% |

### Horizon: `MID` (63 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null? | Perm. p-val | $n_{	ext{eff}}$ | Gating Status | Folds IC > 0 | Net (Full) | Net (Actual) | Avg Replaced Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2008-19)** | +0.025 | No | 0.184 | 46 | evaluable | 9 / 12 | -2.83% | -2.01% | 58.2% |
| `rf_reg_core` | *Stress (2020)* | +0.092 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | -0.24% | +0.26% | 66.7% |
| `rf_reg_core` | *Holdout (2024-25)* | +0.298 | Yes | 0.005 | 5 | indicative only | 1 / 1 | +5.10% | +5.90% | 50.0% |
| `rf_reg_full` | **Dev (2008-19)** | -0.003 | No | 0.559 | 46 | evaluable | 5 / 12 | -2.50% | -1.65% | 56.9% |
| `rf_reg_full` | *Stress (2020)* | +0.095 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | +3.97% | +4.77% | 46.7% |
| `rf_reg_full` | *Holdout (2024-25)* | +0.135 | - | - | 3 | not evaluable (n_eff < 5) | 1 / 1 | -0.85% | -0.45% | 70.0% |
| `rf_clf_core` | **Dev (2008-19)** | -0.006 | No | 0.602 | 46 | evaluable | 4 / 12 | -2.85% | -2.10% | 61.8% |
| `rf_clf_core` | *Stress (2020)* | -0.032 | - | - | 4 | not evaluable (n_eff < 5) | 0 / 1 | +0.57% | +1.27% | 53.3% |
| `rf_clf_core` | *Holdout (2024-25)* | +0.273 | Yes | 0.017 | 5 | indicative only | 1 / 1 | +3.99% | +4.71% | 55.0% |
| `rf_clf_full` | **Dev (2008-19)** | -0.011 | No | 0.694 | 46 | evaluable | 4 / 12 | -2.40% | -1.53% | 55.6% |
| `rf_clf_full` | *Stress (2020)* | -0.041 | - | - | 4 | not evaluable (n_eff < 5) | 0 / 1 | +4.84% | +5.34% | 66.7% |
| `rf_clf_full` | *Holdout (2024-25)* | +0.321 | - | - | 3 | not evaluable (n_eff < 5) | 1 / 1 | +1.91% | +2.71% | 40.0% |
| `ridge_core` | **Dev (2008-19)** | +0.014 | No | 0.300 | 46 | evaluable | 7 / 12 | -1.54% | -0.69% | 56.9% |
| `ridge_core` | *Stress (2020)* | +0.071 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | +1.28% | +1.98% | 53.3% |
| `ridge_core` | *Holdout (2024-25)* | +0.330 | Yes | 0.001 | 5 | indicative only | 1 / 1 | +2.44% | +3.32% | 45.0% |
| `logit_core` | **Dev (2008-19)** | +0.009 | No | 0.359 | 46 | evaluable | 7 / 12 | -1.34% | -0.48% | 56.0% |
| `logit_core` | *Stress (2020)* | +0.038 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | +1.56% | +2.36% | 46.7% |
| `logit_core` | *Holdout (2024-25)* | +0.268 | Yes | 0.022 | 5 | indicative only | 1 / 1 | +3.64% | +4.68% | 35.0% |
| `momentum_126_core` | **Dev (2008-19)** | +0.066 | Yes | 0.010 | 46 | evaluable | 9 / 12 | -0.75% | +0.30% | 46.2% |
| `momentum_126_core` | *Stress (2020)* | +0.064 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | +2.46% | +3.26% | 46.7% |
| `momentum_126_core` | *Holdout (2024-25)* | +0.114 | No | 0.212 | 5 | indicative only | 1 / 1 | -0.46% | +0.10% | 65.0% |
| `momentum_63_core` | **Dev (2008-19)** | +0.049 | No | 0.044 | 46 | evaluable | 8 / 12 | -1.45% | -0.74% | 64.0% |
| `momentum_63_core` | *Stress (2020)* | +0.047 | - | - | 4 | not evaluable (n_eff < 5) | 1 / 1 | -2.82% | -2.52% | 80.0% |
| `momentum_63_core` | *Holdout (2024-25)* | +0.084 | No | 0.265 | 5 | indicative only | 1 / 1 | -1.68% | -1.20% | 70.0% |
| `reversal_21_core` | **Dev (2008-19)** | -0.035 | No | 0.914 | 46 | evaluable | 5 / 12 | -1.96% | -1.39% | 70.7% |
| `reversal_21_core` | *Stress (2020)* | -0.025 | - | - | 4 | not evaluable (n_eff < 5) | 0 / 1 | -0.49% | -0.09% | 73.3% |
| `reversal_21_core` | *Holdout (2024-25)* | -0.038 | No | 0.584 | 5 | indicative only | 0 / 1 | -2.07% | -1.67% | 75.0% |
| `past_winner_core` | **Dev (2008-19)** | +0.023 | No | 0.201 | 46 | evaluable | 6 / 12 | -0.41% | +1.24% | 15.6% |
| `past_winner_core` | *Stress (2020)* | -0.124 | - | - | 4 | not evaluable (n_eff < 5) | 0 / 1 | -0.36% | +0.94% | 13.3% |
| `past_winner_core` | *Holdout (2024-25)* | +0.098 | No | 0.239 | 5 | indicative only | 1 / 1 | -2.11% | -0.67% | 10.0% |

### Horizon: `LONG` (252 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null? | Perm. p-val | $n_{	ext{eff}}$ | Gating Status | Folds IC > 0 | Net (Full) | Net (Actual) | Avg Replaced Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2010-19)** | +0.060 | No | 0.157 | 10 | indicative only | 7 / 10 | +4.59% | +5.47% | 51.1% |
| `rf_reg_core` | *Holdout (2024-25)* | +0.375 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | +10.69% | +10.69% | 100.0% |
| `rf_reg_full` | **Dev (2010-19)** | +0.013 | No | 0.401 | 10 | indicative only | 5 / 10 | +5.81% | +6.53% | 60.0% |
| `rf_clf_core` | **Dev (2010-19)** | +0.035 | No | 0.273 | 10 | indicative only | 6 / 10 | -9.61% | -8.89% | 60.0% |
| `rf_clf_core` | *Holdout (2024-25)* | +0.274 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | +4.46% | +4.46% | 100.0% |
| `rf_clf_full` | **Dev (2010-19)** | +0.014 | No | 0.392 | 10 | indicative only | 6 / 10 | +4.63% | +5.35% | 60.0% |
| `ridge_core` | **Dev (2010-19)** | +0.069 | No | 0.113 | 10 | indicative only | 7 / 10 | -8.84% | -8.20% | 64.4% |
| `ridge_core` | *Holdout (2024-25)* | +0.478 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | +29.53% | +29.53% | 100.0% |
| `logit_core` | **Dev (2010-19)** | +0.046 | No | 0.220 | 10 | indicative only | 7 / 10 | -8.34% | -7.78% | 68.9% |
| `logit_core` | *Holdout (2024-25)* | +0.456 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | +29.53% | +29.53% | 100.0% |
| `momentum_126_core` | **Dev (2010-19)** | +0.068 | No | 0.121 | 10 | indicative only | 7 / 10 | -6.87% | -6.43% | 75.6% |
| `momentum_126_core` | *Holdout (2024-25)* | -0.013 | - | - | 1 | not evaluable (n_eff < 5) | 0 / 1 | -3.29% | -3.29% | 100.0% |
| `momentum_63_core` | **Dev (2010-19)** | +0.076 | No | 0.095 | 10 | indicative only | 8 / 10 | -3.96% | -3.60% | 80.0% |
| `momentum_63_core` | *Holdout (2024-25)* | +0.130 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | -13.79% | -13.79% | 100.0% |
| `reversal_21_core` | **Dev (2010-19)** | -0.054 | No | 0.781 | 10 | indicative only | 2 / 10 | +1.80% | +2.16% | 80.0% |
| `reversal_21_core` | *Holdout (2024-25)* | -0.101 | - | - | 1 | not evaluable (n_eff < 5) | 0 / 1 | -8.77% | -8.77% | 100.0% |
| `past_winner_core` | **Dev (2010-19)** | -0.045 | No | 0.737 | 10 | indicative only | 4 / 10 | +15.40% | +16.68% | 28.9% |
| `past_winner_core` | *Holdout (2024-25)* | +0.108 | - | - | 1 | not evaluable (n_eff < 5) | 1 / 1 | -12.57% | -12.57% | 100.0% |

---

## 4. Factor Tilt & Neutralization Diagnostics

Does model skill survive when controlling for market volatility, liquidity, and momentum tilts?

| Horizon | Model | Split Type | Volatility Tilt | Liquidity Tilt | Momentum Tilt | Raw Mean IC | Neutralized Mean IC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SHORT** | `ridge_core` | Dev | -0.238 | -0.344 | +0.090 | +0.028 | +0.040 |
| **SHORT** | `ridge_core` | Holdout | -0.111 | -0.464 | +0.148 | +0.191 | +0.122 |
| **SHORT** | `logit_core` | Dev | -0.326 | -0.361 | +0.122 | +0.024 | +0.049 |
| **SHORT** | `logit_core` | Holdout | -0.161 | -0.494 | +0.271 | +0.180 | +0.119 |
| **SHORT** | `rf_reg_core` | Dev | +0.001 | -0.484 | +0.005 | +0.007 | +0.012 |
| **SHORT** | `rf_reg_core` | Holdout | -0.021 | -0.557 | +0.101 | +0.207 | +0.083 |
| **SHORT** | `rf_clf_core` | Dev | -0.092 | -0.460 | +0.069 | +0.009 | +0.013 |
| **SHORT** | `rf_clf_core` | Holdout | -0.005 | -0.504 | +0.301 | +0.160 | +0.077 |
| **SHORT** | `momentum_126_core` | Dev | +0.065 | +0.078 | +0.615 | +0.036 | +0.026 |
| **SHORT** | `momentum_126_core` | Holdout | +0.355 | -0.136 | +0.603 | +0.038 | +0.078 |
| **MID** | `ridge_core` | Dev | -0.113 | -0.371 | +0.097 | +0.014 | +0.044 |
| **MID** | `ridge_core` | Holdout | -0.070 | -0.563 | +0.271 | +0.330 | +0.184 |
| **MID** | `logit_core` | Dev | -0.217 | -0.336 | +0.079 | +0.010 | +0.042 |
| **MID** | `logit_core` | Holdout | -0.239 | -0.472 | +0.166 | +0.268 | +0.156 |
| **MID** | `rf_reg_core` | Dev | +0.131 | -0.389 | +0.137 | +0.025 | +0.021 |
| **MID** | `rf_reg_core` | Holdout | -0.200 | -0.558 | +0.128 | +0.298 | +0.202 |
| **MID** | `rf_clf_core` | Dev | +0.033 | -0.404 | +0.080 | -0.006 | +0.007 |
| **MID** | `rf_clf_core` | Holdout | -0.344 | -0.485 | +0.033 | +0.273 | +0.210 |
| **MID** | `momentum_126_core` | Dev | +0.068 | +0.077 | +0.615 | +0.066 | +0.055 |
| **MID** | `momentum_126_core` | Holdout | +0.302 | -0.147 | +0.591 | +0.114 | +0.100 |
| **LONG** | `ridge_core` | Dev | -0.008 | -0.332 | +0.113 | +0.069 | +0.089 |
| **LONG** | `ridge_core` | Holdout | +0.022 | -0.682 | +0.313 | +0.478 | +0.220 |
| **LONG** | `logit_core` | Dev | -0.205 | -0.114 | -0.009 | +0.046 | +0.081 |
| **LONG** | `logit_core` | Holdout | +0.042 | -0.638 | +0.386 | +0.456 | +0.173 |
| **LONG** | `rf_reg_core` | Dev | +0.211 | -0.414 | +0.001 | +0.060 | +0.026 |
| **LONG** | `rf_reg_core` | Holdout | -0.075 | -0.707 | +0.322 | +0.375 | -0.012 |
| **LONG** | `rf_clf_core` | Dev | +0.092 | -0.200 | -0.023 | +0.035 | -0.014 |
| **LONG** | `rf_clf_core` | Holdout | +0.166 | -0.249 | +0.374 | +0.274 | +0.051 |
| **LONG** | `momentum_126_core` | Dev | +0.083 | +0.084 | +0.634 | +0.068 | +0.011 |
| **LONG** | `momentum_126_core` | Holdout | +0.001 | -0.101 | +0.417 | -0.013 | -0.099 |

> [!NOTE]
> **Illiquidity Tilts on Holdout**: Both linear and tree models exhibit extreme negative liquidity tilts (**-0.46 to -0.71**) during the 2024–2025 recovery holdout. When residualized against market volatility, liquidity, and momentum, raw holdout IC falls by 35% to 60%, showing that nominal holdout performance is largely driven by buying illiquid names that rebounded during post-crisis normalization.

---

## 5. Audit & Re-Implementation Reconciliation (Part A Findings)

1. **Independent Cross-Check Agreement**:
   - `scripts/audit_independent.py` recomputed daily cross-sectional Spearman IC using only basic pandas/numpy (zero imports from `src/models`).
   - All numbers agree with pipeline results to machine precision ($\le 0.00000000000000001$, well within the mandatory $\le 0.002$ tolerance).
2. **Reconciliation of `momentum_126` Long Dev IC (0.326 vs 0.068)**:
   - In v1, the reported Long Dev row mistakenly pulled only the very first annual fold `dev_2010` (where `Rebalance Periods = 1`), which captured the massive 2010 post-war rally where momentum spiked to +0.3262.
   - The pipeline in v2/v3 correctly aggregates all 10 annual development folds (2010–2019), averaging +0.0682. Target purging did not affect this non-learned baseline.
   - `momentum_63` Long Dev IC in v1 similarly reflected `dev_2010` (+0.2371), whereas the full 10-fold aggregate is +0.0764.
3. **Turnover Metric Correction**:
   - Previous versions reported identical values for `Net (Full)` and `Net (Actual)` due to cost wiring.
   - Fixed wiring accurately measures the fraction of top-5 positions replaced per rebalance. For `momentum_126` short dev, only 30.5% of names turn over each month, reducing friction from -1.60% down to -0.22%.
4. **Feature Hygiene & Dropped Column Accounting**:
   - `zero_trade_share_63d` is constant 0.0 across all stocks and dates in the cleaned dataset (std = 0.0). It was retained in linear models where cs-rank tied.
   - `index_ret_63d` and `index_vol_63d` are identical across stocks on each date. Their daily cs-ranks tie at $(N_t + 1)/(2 N_t)$, causing $|r| = 1.0$ collinearity with `zero_trade_share_63d`. They were dropped for collinearity, leaving exactly 25 core linear features ($30 - 3	ext{ redundant ranks} - 2	ext{ collinear index features} = 25$).
   - Raw tree models use 29 active features (excluding zero-variance `zero_trade_share_63d`).
5. **Universe Benchmark Name Clarification**:
   - Within the 20-company universe, the first 5 alphabetical stocks picked by the old tie-breaker were `['AEL.N', 'CFIN.N', 'COMB.N', 'DFCC.N', 'DIAL.N']`.
   - Non-universe tickers (`ACL`, `AHUN`, `BRWN`, `CARG`) were mentioned in an earlier draft in error; verification against the raw source confirms they do not exist anywhere in the raw data, which contains strictly the 20 S&P SL20 companies.
6. **2001 Coverage Clarification**:
   - While the raw Excel filename is titled `2001_2025`, trading records for the 20 S&P SL20 companies only begin in January 2002. There are 0 rows in 2001.

---
*Report generated by Stock Pulse Evaluation Engine (Part A Repair Complete).*

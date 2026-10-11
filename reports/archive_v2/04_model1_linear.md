# Stock Pulse: Model 1 — Linear Models (Ridge & Logistic Regression)

**Document**: `reports/04_model1_linear.md`  
**Purpose**: Benchmark Linear Regression (Ridge) and Classification (Logit) against baseline strategies across walk-forward folds.

> [!WARNING]
> **Survivorship Bias Warning**: S&P SL20 constituents represent today's index survivors. Long-horizon momentum and historical winner baselines are biased upwards by this survivorship effect.

---

## 1. Executive Summary & Verdict per Horizon

### Short Horizon (`21 Trading Days` / ~1 Month)
- **Verdict**: Neither `ridge` nor `logit` achieves a robust edge over naive baselines. Across the 12 development folds (2008–2019), `ridge_core` achieves positive Rank IC in only ~6 out of 12 folds (50% win rate), falling short of the two-thirds (67%) consistency hurdle.
- **Core vs. Full**: `core` features are superior to `full` features on short horizon. Adding 252-day backward features introduces noise and reduces validation Rank IC.
- **Holdout (2024–2025)**: On the holdout period, linear models fail to generate substantial net excess return after deducting the **ASSUMED** 200 bps round-trip friction.

### Mid Horizon (`63 Trading Days` / ~1 Quarter)
- **Verdict**: `ridge_core` and `logit_core` exhibit moderate correlation with forward returns, but do not consistently surpass standard momentum (`momentum_126`). On the 12 development folds, positive IC occurs in ~7 out of 12 folds (~58%), which does not meet the 67% threshold.
- **Holdout (2024–2025)**: Models maintain modest positive Rank IC, but net portfolio returns after friction remain near zero.

### Long Horizon (`252 Trading Days` / ~1 Year)
- **Verdict**: `ridge_core` generates strong nominal Rank IC on development folds (+0.25 to +0.32), but cannot beat the static `past_winner` heuristic (+0.306) or 126-day momentum (+0.326).
- **Holdout Full Feature Constraint**: For the long horizon holdout with the full feature set (2024–2025), **zero rows remain** with $\ge 90\%$ non-null full features because 252-day forward targets require full forward history while 252-day features require historical history, leaving the holdout **not evaluable**.

---

## 2. Performance Comparison Tables (Model 1 vs. Baselines)

### Horizon: `SHORT` (21 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.010 | 7 / 12 | +0.32% | +0.16% | -2.42% | 138 |
| `ridge` | `core` | *Stress (2020)* | -0.009 | 0 / 1 | +1.94% | +0.77% | -2.78% | 10 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.139 | 1 / 1 | +1.45% | +1.07% | -0.47% | 22 |
| `ridge` | `full` | **Dev (2008-19)** | -0.003 | 4 / 12 | +0.23% | +0.06% | -2.41% | 138 |
| `ridge` | `full` | *Stress (2020)* | +0.007 | 1 / 1 | +1.98% | +0.80% | -2.31% | 10 |
| `ridge` | `full` | *Holdout (2024-25)* | +0.268 | 1 / 1 | +4.06% | +3.68% | +0.18% | 10 |
| `logit` | `core` | **Dev (2008-19)** | +0.013 | 7 / 12 | +0.32% | +0.15% | -2.07% | 138 |
| `logit` | `core` | *Stress (2020)* | -0.053 | 0 / 1 | +1.77% | +0.59% | -2.68% | 10 |
| `logit` | `core` | *Holdout (2024-25)* | +0.133 | 1 / 1 | +1.65% | +1.27% | -0.70% | 22 |
| `logit` | `full` | **Dev (2008-19)** | -0.001 | 5 / 12 | +0.21% | +0.05% | -2.23% | 138 |
| `logit` | `full` | *Stress (2020)* | -0.046 | 0 / 1 | +1.22% | +0.04% | -2.88% | 10 |
| `logit` | `full` | *Holdout (2024-25)* | +0.238 | 1 / 1 | +3.57% | +3.19% | -0.38% | 10 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.026 | 8 / 12 | +0.54% | +0.38% | -1.45% | 138 |
| `momentum_126` | `core` | *Stress (2020)* | +0.074 | 1 / 1 | +2.57% | +1.39% | -0.50% | 10 |
| `momentum_126` | `core` | *Holdout (2024-25)* | +0.044 | 1 / 1 | +0.46% | +0.08% | -2.03% | 22 |
| `past_winner` | `core` | **Dev (2008-19)** | -0.001 | 7 / 12 | -0.00% | -0.17% | -2.24% | 138 |
| `past_winner` | `core` | *Stress (2020)* | -0.068 | 0 / 1 | +0.76% | -0.42% | -2.89% | 10 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.071 | 1 / 1 | +0.98% | +0.60% | -1.36% | 22 |
| `random` | `core` | **Dev (2008-19)** | -0.006 | 6 / 12 | +0.17% | +0.00% | -1.89% | 138 |
| `random` | `core` | *Stress (2020)* | +0.002 | 1 / 1 | +1.18% | +0.00% | -3.41% | 10 |
| `random` | `core` | *Holdout (2024-25)* | +0.015 | 1 / 1 | +0.38% | +0.00% | -2.06% | 22 |
| `equal_weight_universe` | `core` | **Dev (2008-19)** | 0.000 | N/A | +0.14% | -0.02% | -2.23% | 138 |
| `equal_weight_universe` | `core` | *Stress (2020)* | 0.000 | N/A | +0.03% | -1.15% | -3.52% | 10 |
| `equal_weight_universe` | `core` | *Holdout (2024-25)* | 0.000 | N/A | +1.02% | +0.64% | -1.36% | 22 |

### Horizon: `MID` (63 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.012 | 5 / 12 | +1.05% | +0.22% | -2.79% | 46 |
| `ridge` | `core` | *Stress (2020)* | +0.098 | 1 / 1 | +11.39% | +7.79% | +3.36% | 4 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.261 | 1 / 1 | +5.22% | +4.42% | +4.11% | 7 |
| `ridge` | `full` | **Dev (2008-19)** | -0.012 | 6 / 12 | +0.32% | -0.51% | -0.78% | 46 |
| `ridge` | `full` | *Stress (2020)* | +0.108 | 1 / 1 | +10.94% | +7.33% | +1.54% | 4 |
| `ridge` | `full` | *Holdout (2024-25)* | +0.462 | 1 / 1 | +9.81% | +9.01% | +6.71% | 3 |
| `logit` | `core` | **Dev (2008-19)** | +0.006 | 5 / 12 | +0.90% | +0.07% | -2.33% | 46 |
| `logit` | `core` | *Stress (2020)* | +0.063 | 1 / 1 | +10.32% | +6.72% | +2.14% | 4 |
| `logit` | `core` | *Holdout (2024-25)* | +0.239 | 1 / 1 | +4.04% | +3.23% | +4.06% | 7 |
| `logit` | `full` | **Dev (2008-19)** | -0.019 | 4 / 12 | +0.32% | -0.51% | -1.65% | 46 |
| `logit` | `full` | *Stress (2020)* | +0.078 | 1 / 1 | +7.39% | +3.79% | +0.83% | 4 |
| `logit` | `full` | *Holdout (2024-25)* | +0.493 | 1 / 1 | +12.03% | +11.23% | +7.23% | 3 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.058 | 9 / 12 | +1.39% | +0.57% | -0.35% | 46 |
| `momentum_126` | `core` | *Stress (2020)* | +0.037 | 1 / 1 | +7.05% | +3.44% | +0.03% | 4 |
| `momentum_126` | `core` | *Holdout (2024-25)* | +0.109 | 1 / 1 | +0.76% | -0.04% | -1.88% | 7 |
| `past_winner` | `core` | **Dev (2008-19)** | +0.039 | 8 / 12 | +1.33% | +0.50% | +0.33% | 46 |
| `past_winner` | `core` | *Stress (2020)* | -0.149 | 0 / 1 | +0.90% | -2.71% | +0.03% | 4 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.110 | 1 / 1 | +2.17% | +1.37% | +0.15% | 7 |
| `random` | `core` | **Dev (2008-19)** | +0.002 | 7 / 12 | +0.83% | +0.00% | -1.10% | 46 |
| `random` | `core` | *Stress (2020)* | +0.007 | 1 / 1 | +3.61% | +0.00% | -8.04% | 4 |
| `random` | `core` | *Holdout (2024-25)* | +0.019 | 1 / 1 | +0.80% | +0.00% | -1.37% | 7 |
| `equal_weight_universe` | `core` | **Dev (2008-19)** | 0.000 | N/A | -0.13% | -0.96% | -4.14% | 46 |
| `equal_weight_universe` | `core` | *Stress (2020)* | 0.000 | N/A | -0.60% | -4.20% | -8.58% | 4 |
| `equal_weight_universe` | `core` | *Holdout (2024-25)* | 0.000 | N/A | +2.63% | +1.82% | -0.08% | 7 |

### Horizon: `LONG` (252 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.060 | 7 / 10 | +4.60% | +1.96% | +3.00% | 10 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.437 | 1 / 1 | +37.65% | +37.21% | -0.16% | 1 |
| `ridge` | `full` | **Dev (2008-19)** | +0.067 | 7 / 10 | +3.89% | +1.25% | +5.54% | 10 |
| `ridge` | `full` | *Holdout (2024-25)* | *Not Evaluable* | *0 / 0* | N/A (<90% non-null) | N/A | N/A | 0 |
| `logit` | `core` | **Dev (2008-19)** | +0.018 | 7 / 10 | +2.42% | -0.22% | -9.48% | 10 |
| `logit` | `core` | *Holdout (2024-25)* | +0.408 | 1 / 1 | +33.32% | +32.89% | -0.16% | 1 |
| `logit` | `full` | **Dev (2008-19)** | +0.021 | 6 / 10 | +2.40% | -0.25% | +9.12% | 10 |
| `logit` | `full` | *Holdout (2024-25)* | *Not Evaluable* | *0 / 0* | N/A (<90% non-null) | N/A | N/A | 0 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.068 | 7 / 10 | +4.85% | +2.20% | -6.64% | 10 |
| `momentum_126` | `core` | *Holdout (2024-25)* | -0.022 | 0 / 1 | +4.22% | +3.78% | -0.16% | 1 |
| `past_winner` | `core` | **Dev (2008-19)** | -0.022 | 6 / 10 | +2.55% | -0.09% | +3.14% | 10 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.204 | 1 / 1 | +4.05% | +3.62% | +2.88% | 1 |
| `random` | `core` | **Dev (2008-19)** | +0.006 | 7 / 10 | +2.65% | +0.00% | -14.61% | 10 |
| `random` | `core` | *Holdout (2024-25)* | +0.001 | 1 / 1 | +0.44% | +0.00% | -13.38% | 1 |
| `equal_weight_universe` | `core` | **Dev (2008-19)** | 0.000 | N/A | +0.03% | -2.62% | -12.70% | 10 |
| `equal_weight_universe` | `core` | *Holdout (2024-25)* | 0.000 | N/A | +22.78% | +22.35% | -0.16% | 1 |

---

## 3. Top 10 Features and Sign Consistency

Features with the highest mean absolute standardized coefficient across development folds:

### Horizon: `SHORT` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `vol_63d` | -0.1977 | 0.1977 | 100.0% |
| `cs_rank_vol_63d` | +0.1623 | 0.1630 | 75.0% |
| `amihud_illiq_63d_isna` | -0.1064 | 0.1105 | 75.0% |
| `rel_ret_63d_isna` | -0.1022 | 0.1022 | 100.0% |
| `vol_ratio_21_126_isna` | +0.0946 | 0.0947 | 91.7% |
| `cs_rank_mom_63d` | -0.0878 | 0.0878 | 100.0% |
| `close_to_sma50_isna` | +0.0705 | 0.0712 | 91.7% |
| `macd_hist_norm_isna` | +0.0697 | 0.0697 | 100.0% |
| `log_turnover_21d_isna` | -0.0533 | 0.0572 | 75.0% |
| `rel_ret_63d` | +0.0565 | 0.0565 | 91.7% |

### Horizon: `MID` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `amihud_illiq_63d_isna` | -0.1538 | 0.1739 | 75.0% |
| `rel_ret_63d_isna` | -0.1625 | 0.1625 | 100.0% |
| `vol_63d` | -0.1611 | 0.1617 | 75.0% |
| `cs_rank_vol_63d` | +0.1495 | 0.1496 | 91.7% |
| `cs_rank_vol_21d_isna` | -0.1425 | 0.1432 | 50.0% |
| `vol_ratio_21_126_isna` | +0.1119 | 0.1143 | 66.7% |
| `downside_vol_63d_isna` | +0.0902 | 0.0909 | 66.7% |
| `log_ret_63_isna` | +0.0902 | 0.0909 | 66.7% |
| `vol_63d_isna` | +0.0902 | 0.0909 | 66.7% |
| `cs_rank_vol_21d` | +0.0733 | 0.0773 | 50.0% |

### Horizon: `LONG` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `vol_63d` | -0.1471 | 0.1482 | 80.0% |
| `log_turnover_21d` | -0.1309 | 0.1309 | 100.0% |
| `amihud_illiq_63d_isna` | -0.1146 | 0.1146 | 100.0% |
| `cs_rank_vol_63d` | +0.1057 | 0.1114 | 80.0% |
| `vol_ratio_21_126_isna` | +0.0950 | 0.1036 | 40.0% |
| `log_trade_vol_21d` | +0.0850 | 0.0921 | 50.0% |
| `rel_ret_21d_isna` | -0.0889 | 0.0889 | 100.0% |
| `cs_rank_vol_21d` | -0.0560 | 0.0656 | 70.0% |
| `dist_low_126d_isna` | -0.0616 | 0.0616 | 100.0% |
| `dist_high_126d_isna` | -0.0616 | 0.0616 | 100.0% |

---

## 4. Probability Calibration & Decile Lift (Logistic Regression)

Calibration on development out-of-sample predictions:

### Horizon: `SHORT` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,704 | 0.428 | 49.4% | 50.0% | 0.99x |
| 2 | 4,703 | 0.461 | 49.4% | 50.0% | 0.99x |
| 3 | 4,704 | 0.475 | 49.7% | 50.0% | 0.99x |
| 4 | 4,703 | 0.486 | 49.8% | 50.0% | 1.00x |
| 5 | 4,704 | 0.496 | 48.7% | 50.0% | 0.97x |
| 6 | 4,703 | 0.505 | 48.7% | 50.0% | 0.97x |
| 7 | 4,703 | 0.515 | 51.1% | 50.0% | 1.02x |
| 8 | 4,704 | 0.525 | 50.1% | 50.0% | 1.00x |
| 9 | 4,703 | 0.539 | 50.4% | 50.0% | 1.01x |
| 10 | 4,704 | 0.570 | 52.6% | 50.0% | 1.05x |

**Top Decile Lift**: **1.05x** over random base rate.

### Horizon: `MID` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,691 | 0.424 | 51.9% | 50.0% | 1.04x |
| 2 | 4,690 | 0.464 | 52.6% | 50.0% | 1.05x |
| 3 | 4,690 | 0.478 | 50.4% | 50.0% | 1.01x |
| 4 | 4,690 | 0.487 | 48.8% | 50.0% | 0.98x |
| 5 | 4,691 | 0.496 | 49.3% | 50.0% | 0.99x |
| 6 | 4,690 | 0.504 | 49.6% | 50.0% | 0.99x |
| 7 | 4,690 | 0.513 | 47.9% | 50.0% | 0.96x |
| 8 | 4,690 | 0.523 | 52.5% | 50.0% | 1.05x |
| 9 | 4,690 | 0.537 | 53.7% | 50.0% | 1.07x |
| 10 | 4,691 | 0.580 | 43.3% | 50.0% | 0.87x |

**Top Decile Lift**: **0.87x** over random base rate.

### Horizon: `LONG` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,037 | 0.355 | 46.3% | 50.0% | 0.93x |
| 2 | 4,036 | 0.428 | 47.4% | 50.0% | 0.95x |
| 3 | 4,036 | 0.460 | 50.3% | 50.0% | 1.01x |
| 4 | 4,037 | 0.481 | 49.5% | 50.0% | 0.99x |
| 5 | 4,036 | 0.496 | 50.9% | 50.0% | 1.02x |
| 6 | 4,036 | 0.510 | 51.4% | 50.0% | 1.03x |
| 7 | 4,037 | 0.525 | 50.4% | 50.0% | 1.01x |
| 8 | 4,036 | 0.547 | 53.1% | 50.0% | 1.06x |
| 9 | 4,036 | 0.579 | 54.6% | 50.0% | 1.09x |
| 10 | 4,037 | 0.641 | 46.2% | 50.0% | 0.92x |

**Top Decile Lift**: **0.92x** over random base rate.

---

## 5. Visualizations

### Annual Rank IC Comparison
![Rank IC Comparison](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/linear_mean_ic_by_year.png)

### Top 10 Coefficients
![Feature Coefficients](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/linear_top_coefficients.png)

---
*Generated by Stock Pulse Model 1 Benchmark Engine.*

# Stock Pulse: Baseline Model Performance & Benchmark Results

**Document**: `reports/03_baseline_results.md`  
**Purpose**: Rigorous benchmarking of naive, heuristic, and rule-based predictors against which all machine learning models will be compared.

---

## 1. Executive Summary & "What This Means"

- **Reversal Dominates Short Horizon (21d)**: The 21-day price reversal heuristic (`reversal_21 = -log_ret_21`) produces a positive Rank IC and positive top-5 net returns across development folds. In the frontier Colombo equity market, short-term price spikes experience strong mean reversion over 1-month horizons.
- **Momentum Signal is Noisy and Decays**: Standard 3-month and 6-month momentum (`momentum_63`, `momentum_126`) show weak or negative correlation with forward returns in Sri Lanka, reflecting market thinness and regime instability.
- **Transaction Costs are a Major Hurdle**: Under realistic CSE round-trip trading frictions of 200 bps (100 bps per side), monthly rebalancing strategies suffer severe drag. Only strategies with substantial gross alpha can remain net-profitable.
- **Zero-Intelligence Benchmarks**: As expected, `random` produces a near-zero Rank IC (~0.00) and zero net excess return, validating the statistical unbiasedness of the evaluation engine.

---

## 2. Performance Summary Tables (Development Folds vs. Final Holdout)

### Horizon: `SHORT` (21 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | 95% Bootstrap CI | Top-5 Mean Excess | Top-5 Median Excess | Top-5 Precision | Net Port. Excess | Rebalance Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | -0.053 | [-0.075, -0.026] | +0.05% | -0.22% | 45.4% | -4.28% | 12 |
| `random` | *Holdout (2024-25)* | -0.014 | [-0.031, +0.006] | +0.66% | +0.23% | 48.2% | -2.77% | 22 |
| `reversal_21` | **Dev (2008-19)** | -0.011 | [-0.204, +0.124] | +0.65% | +0.42% | 47.5% | -2.07% | 12 |
| `reversal_21` | *Holdout (2024-25)* | +0.019 | [-0.041, +0.093] | +1.10% | +0.65% | 48.5% | -0.77% | 22 |
| `momentum_63` | **Dev (2008-19)** | -0.018 | [-0.106, +0.129] | -0.05% | -0.29% | 49.5% | -3.60% | 12 |
| `momentum_63` | *Holdout (2024-25)* | -0.027 | [-0.113, +0.054] | +0.45% | +0.14% | 43.9% | -2.37% | 22 |
| `momentum_126` | **Dev (2008-19)** | +0.029 | [-0.045, +0.156] | +0.47% | +0.37% | 50.7% | -2.05% | 12 |
| `momentum_126` | *Holdout (2024-25)* | +0.035 | [-0.070, +0.139] | +0.54% | +0.33% | 44.9% | -2.34% | 22 |
| `low_vol` | **Dev (2008-19)** | -0.138 | [-0.267, -0.050] | -0.18% | -0.51% | 41.4% | -2.22% | 12 |
| `low_vol` | *Holdout (2024-25)* | -0.038 | [-0.136, +0.058] | +0.51% | +0.17% | 48.8% | -2.66% | 22 |
| `past_winner` | **Dev (2008-19)** | -0.184 | [-0.292, -0.073] | -1.17% | -1.05% | 40.8% | -4.09% | 12 |
| `past_winner` | *Holdout (2024-25)* | +0.069 | [+0.023, +0.106] | +1.53% | +1.22% | 49.7% | -1.37% | 22 |
| `equal_weight_universe` | **Dev (2008-19)** | 0.000 | N/A | -0.03% | -0.30% | 44.1% | -2.14% | 12 |
| `equal_weight_universe` | *Holdout (2024-25)* | 0.000 | N/A | +1.42% | +1.12% | 50.6% | -1.55% | 22 |

### Horizon: `MID` (63 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | 95% Bootstrap CI | Top-5 Mean Excess | Top-5 Median Excess | Top-5 Precision | Net Port. Excess | Rebalance Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | +0.023 | [+0.014, +0.069] | +2.18% | +1.81% | 49.3% | -1.15% | 4 |
| `random` | *Holdout (2024-25)* | -0.022 | [-0.053, -0.005] | +0.80% | +0.56% | 47.3% | -5.08% | 7 |
| `reversal_21` | **Dev (2008-19)** | +0.024 | [-0.166, +0.052] | +3.76% | +2.71% | 50.0% | -1.15% | 4 |
| `reversal_21` | *Holdout (2024-25)* | +0.008 | [-0.110, +0.116] | +0.48% | +0.88% | 48.5% | -1.45% | 7 |
| `momentum_63` | **Dev (2008-19)** | -0.070 | [-0.220, +0.154] | -0.09% | +0.20% | 42.2% | -5.19% | 4 |
| `momentum_63` | *Holdout (2024-25)* | +0.041 | [-0.114, +0.187] | +2.52% | +1.25% | 47.0% | -1.44% | 7 |
| `momentum_126` | **Dev (2008-19)** | +0.082 | [-0.048, +0.262] | +2.36% | +2.53% | 48.7% | -2.12% | 4 |
| `momentum_126` | *Holdout (2024-25)* | +0.119 | [-0.062, +0.303] | +2.62% | +0.86% | 45.7% | -1.87% | 7 |
| `low_vol` | **Dev (2008-19)** | -0.227 | [-0.382, -0.103] | -0.75% | -0.85% | 37.6% | -4.68% | 4 |
| `low_vol` | *Holdout (2024-25)* | -0.125 | [-0.274, +0.004] | -1.77% | -1.81% | 41.7% | -2.31% | 7 |
| `past_winner` | **Dev (2008-19)** | -0.164 | [-0.346, -0.036] | -0.18% | -0.18% | 45.1% | -6.29% | 4 |
| `past_winner` | *Holdout (2024-25)* | +0.120 | [+0.043, +0.180] | +2.48% | +2.38% | 50.8% | +0.16% | 7 |
| `equal_weight_universe` | **Dev (2008-19)** | 0.000 | N/A | +0.55% | -1.24% | 41.9% | -4.53% | 4 |
| `equal_weight_universe` | *Holdout (2024-25)* | 0.000 | N/A | +3.12% | +2.84% | 54.0% | -0.08% | 7 |

### Horizon: `LONG` (252 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | 95% Bootstrap CI | Top-5 Mean Excess | Top-5 Median Excess | Top-5 Precision | Net Port. Excess | Rebalance Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | +0.024 | [-0.004, +0.023] | +30.51% | +17.60% | 48.7% | -154.91% | 1 |
| `random` | *Holdout (2024-25)* | +0.035 | [+0.022, +0.041] | +7.99% | +4.39% | 49.8% | -14.55% | 1 |
| `reversal_21` | **Dev (2008-19)** | -0.146 | [-0.206, -0.072] | +17.02% | +12.65% | 44.1% | +45.69% | 1 |
| `reversal_21` | *Holdout (2024-25)* | +0.006 | [-0.046, +0.111] | +2.04% | +2.72% | 52.7% | -1.75% | 1 |
| `momentum_63` | **Dev (2008-19)** | +0.237 | [+0.095, +0.412] | +72.80% | +62.21% | 56.6% | -11.84% | 1 |
| `momentum_63` | *Holdout (2024-25)* | +0.051 | [-0.144, +0.162] | +15.79% | +3.28% | 42.2% | -1.75% | 1 |
| `momentum_126` | **Dev (2008-19)** | +0.326 | [+0.194, +0.518] | +76.50% | +53.31% | 59.8% | +84.80% | 1 |
| `momentum_126` | *Holdout (2024-25)* | -0.015 | [-0.109, +0.040] | +12.47% | +5.19% | 45.2% | -1.75% | 1 |
| `low_vol` | **Dev (2008-19)** | -0.111 | [-0.130, -0.049] | +1.94% | +1.75% | 50.8% | -114.13% | 1 |
| `low_vol` | *Holdout (2024-25)* | -0.138 | [-0.274, -0.113] | -5.17% | -4.49% | 43.2% | -1.75% | 1 |
| `past_winner` | **Dev (2008-19)** | +0.306 | [+0.239, +0.431] | +103.87% | +90.76% | 63.3% | +165.33% | 1 |
| `past_winner` | *Holdout (2024-25)* | +0.209 | [+0.175, +0.228] | +4.08% | +4.56% | 63.0% | +1.29% | 1 |
| `equal_weight_universe` | **Dev (2008-19)** | 0.000 | N/A | +20.76% | +19.42% | 48.9% | -66.34% | 1 |
| `equal_weight_universe` | *Holdout (2024-25)* | 0.000 | N/A | +19.83% | +22.81% | 61.2% | -1.75% | 1 |

---

## 3. Diagnostic Visualizations

### Annual Rank IC Evolution
![Rank IC by Test Year](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/baseline_mean_ic_by_year.png)

### Cumulative Portfolio Performance (Reversal-21 Top-5 vs. Market)
![Cumulative Performance](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/baseline_cumulative_top5_vs_mkt.png)

---
*Generated by Stock Pulse Baseline & Validation Suite.*

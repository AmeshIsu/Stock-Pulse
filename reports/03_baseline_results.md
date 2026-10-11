# Stock Pulse: Baseline Model Performance & Benchmark Results

**Document**: `reports/03_baseline_results.md`  
**Purpose**: Rigorous benchmarking of naive, heuristic, and rule-based predictors against which all machine learning models will be compared.

> [!WARNING]
> **Survivorship Bias Warning**: The 20 stocks in this dataset represent today's S&P SL20 constituents. Testing long-horizon momentum and historical winner heuristics backwards across 2008–2019 carries survivorship bias, so long-horizon past performance estimates are likely inflated.

---

## 1. Executive Summary & "What This Means"

- **Zero-Intelligence Benchmark Validated**: With independent date-level RNG streams and series shuffling, the `random` baseline produces a mean Rank IC within $\pm 0.01$ of zero across development folds and a tie-adjusted top-5 precision of ~50% (between 48% and 52%), confirming that the evaluation framework is statistically unbiased.
- **What the Fold-Level Numbers Show**:
  - `reversal_21`: Shows modest mean Rank IC on short horizon (~ -0.01 to +0.02) with fold win rate of ~50%. It does not show a statistically robust positive edge across annual folds.
  - `momentum_63` & `momentum_126`: Exhibit mixed performance across horizons. Long-horizon momentum has positive Rank IC on dev folds, but this is heavily subject to survivorship bias.
  - `low_vol`: Produces consistently negative Rank IC across horizons. **Low_vol having negative IC means higher-volatility stocks did better, which may be a market-beta effect we will test later.**
- **Transaction Costs are a Dominant Hurdle**: Under the **ASSUMPTION** of 200 bps round-trip friction (100 bps per side), monthly rebalancing strategies suffer substantial drag. Only models generating large gross spreads can remain net-profitable.

---

## 2. Performance Summary Tables (Development Folds vs. Final Holdout)

### Horizon: `SHORT` (21 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | 95% Bootstrap CI | Top-5 Median Excess | Excess vs Random | Top-5 Precision | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | +0.003 | No | 0.251 | [-0.007, +0.012] | +0.33% | +0.00% | 50.4% | -2.05% | +1.22% (comp: +266.0%) | 138 |
| `random` | *Holdout (2024-25)* | +0.014 | No | 0.159 | [-0.006, +0.038] | +0.64% | +0.00% | 50.0% | -0.37% | +5.08% (comp: +110.4%) | 16 |
| `reversal_21` | **Dev (2008-19)** | -0.004 | No | 0.785 | [-0.035, +0.030] | +0.30% | -0.03% | 49.8% | -1.83% | +1.22% (comp: +266.0%) | 138 |
| `reversal_21` | *Holdout (2024-25)* | +0.013 | No | 0.183 | [-0.056, +0.096] | +0.35% | -0.29% | 47.1% | -1.07% | +5.08% (comp: +110.4%) | 16 |
| `momentum_63` | **Dev (2008-19)** | +0.019 | Yes | 0.000 | [-0.015, +0.053] | +0.28% | -0.04% | 50.4% | -1.70% | +1.22% (comp: +266.0%) | 138 |
| `momentum_63` | *Holdout (2024-25)* | -0.025 | No | 0.972 | [-0.125, +0.069] | +0.39% | -0.25% | 45.6% | -2.82% | +5.08% (comp: +110.4%) | 16 |
| `momentum_126` | **Dev (2008-19)** | +0.036 | Yes | 0.000 | [+0.003, +0.072] | +0.60% | +0.28% | 52.3% | -1.60% | +1.22% (comp: +266.0%) | 138 |
| `momentum_126` | *Holdout (2024-25)* | +0.038 | Yes | 0.001 | [-0.058, +0.143] | +1.12% | +0.48% | 48.8% | -1.79% | +5.08% (comp: +110.4%) | 16 |
| `low_vol` | **Dev (2008-19)** | -0.025 | Yes | 1.000 | [-0.062, +0.011] | -0.01% | -0.33% | 49.4% | -2.46% | +1.22% (comp: +266.0%) | 138 |
| `low_vol` | *Holdout (2024-25)* | -0.058 | Yes | 1.000 | [-0.167, +0.065] | +0.22% | -0.42% | 48.5% | -2.74% | +5.08% (comp: +110.4%) | 16 |
| `past_winner` | **Dev (2008-19)** | +0.010 | Yes | 0.025 | [-0.025, +0.045] | +0.27% | -0.06% | 49.4% | -1.66% | +1.22% (comp: +266.0%) | 138 |
| `past_winner` | *Holdout (2024-25)* | +0.053 | Yes | 0.000 | [-0.034, +0.113] | +0.18% | -0.47% | 46.7% | -2.85% | +5.08% (comp: +110.4%) | 16 |
| `equal_weight_market (benchmark)` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | - | +1.22% (comp: +266.0%) | 138 |
| `equal_weight_market (benchmark)` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | - | +5.08% (comp: +110.4%) | 16 |

### Horizon: `MID` (63 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | 95% Bootstrap CI | Top-5 Median Excess | Excess vs Random | Top-5 Precision | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | +0.002 | No | 0.306 | [-0.006, +0.013] | +0.91% | +0.00% | 50.6% | -2.52% | +4.03% (comp: +305.6%) | 46 |
| `random` | *Holdout (2024-25)* | +0.007 | No | 0.311 | N/A (<5 blocks) | +0.75% | +0.00% | 50.3% | +2.84% | +18.97% (comp: +125.4%) | 5 |
| `reversal_21` | **Dev (2008-19)** | -0.035 | Yes | 1.000 | [-0.077, +0.005] | +0.56% | -0.36% | 49.3% | -1.96% | +4.03% (comp: +305.6%) | 46 |
| `reversal_21` | *Holdout (2024-25)* | -0.038 | Yes | 0.996 | N/A (<5 blocks) | -0.92% | -1.67% | 46.4% | -2.07% | +18.97% (comp: +125.4%) | 5 |
| `momentum_63` | **Dev (2008-19)** | +0.049 | Yes | 0.000 | [-0.005, +0.115] | +1.29% | +0.37% | 50.9% | -1.45% | +4.03% (comp: +305.6%) | 46 |
| `momentum_63` | *Holdout (2024-25)* | +0.084 | Yes | 0.000 | N/A (<5 blocks) | +3.31% | +2.56% | 51.7% | -1.68% | +18.97% (comp: +125.4%) | 5 |
| `momentum_126` | **Dev (2008-19)** | +0.066 | Yes | 0.000 | [+0.009, +0.127] | +1.60% | +0.69% | 51.7% | -0.75% | +4.03% (comp: +305.6%) | 46 |
| `momentum_126` | *Holdout (2024-25)* | +0.114 | Yes | 0.000 | N/A (<5 blocks) | +3.71% | +2.97% | 51.1% | -0.46% | +18.97% (comp: +125.4%) | 5 |
| `low_vol` | **Dev (2008-19)** | -0.037 | Yes | 1.000 | [-0.096, +0.029] | +0.01% | -0.91% | 49.2% | -3.39% | +4.03% (comp: +305.6%) | 46 |
| `low_vol` | *Holdout (2024-25)* | -0.136 | Yes | 1.000 | N/A (<5 blocks) | -2.44% | -3.19% | 40.2% | -3.10% | +18.97% (comp: +125.4%) | 5 |
| `past_winner` | **Dev (2008-19)** | +0.023 | Yes | 0.000 | [-0.043, +0.092] | +0.93% | +0.02% | 50.4% | -0.41% | +4.03% (comp: +305.6%) | 46 |
| `past_winner` | *Holdout (2024-25)* | +0.098 | Yes | 0.000 | N/A (<5 blocks) | -1.01% | -1.76% | 42.7% | -2.11% | +18.97% (comp: +125.4%) | 5 |
| `equal_weight_market (benchmark)` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | - | +4.03% (comp: +305.6%) | 46 |
| `equal_weight_market (benchmark)` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | - | +18.97% (comp: +125.4%) | 5 |

### Horizon: `LONG` (252 Trading Days)

| Baseline Model | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | 95% Bootstrap CI | Top-5 Median Excess | Excess vs Random | Top-5 Precision | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `random` | **Dev (2008-19)** | -0.001 | No | 0.593 | [-0.009, +0.008] | +2.46% | +0.00% | 49.9% | -8.66% | +19.07% (comp: +164.1%) | 10 |
| `random` | *Holdout (2024-25)* | +0.004 | No | 0.436 | N/A (<5 blocks) | +6.55% | +0.00% | 47.8% | -11.33% | +87.20% (comp: +87.2%) | 1 |
| `reversal_21` | **Dev (2008-19)** | -0.054 | Yes | 1.000 | [-0.086, -0.008] | +2.04% | -0.42% | 48.6% | +1.80% | +19.07% (comp: +164.1%) | 10 |
| `reversal_21` | *Holdout (2024-25)* | -0.101 | Yes | 1.000 | N/A (<5 blocks) | -6.71% | -13.26% | 50.5% | -8.77% | +87.20% (comp: +87.2%) | 1 |
| `momentum_63` | **Dev (2008-19)** | +0.076 | Yes | 0.000 | [+0.002, +0.131] | +4.24% | +1.77% | 53.9% | -3.96% | +19.07% (comp: +164.1%) | 10 |
| `momentum_63` | *Holdout (2024-25)* | +0.130 | Yes | 0.000 | N/A (<5 blocks) | +46.90% | +40.35% | 45.6% | -13.79% | +87.20% (comp: +87.2%) | 1 |
| `momentum_126` | **Dev (2008-19)** | +0.068 | Yes | 0.000 | [-0.028, +0.152] | +4.83% | +2.37% | 55.4% | -6.87% | +19.07% (comp: +164.1%) | 10 |
| `momentum_126` | *Holdout (2024-25)* | -0.013 | No | 0.698 | N/A (<5 blocks) | +27.52% | +20.97% | 40.3% | -3.29% | +87.20% (comp: +87.2%) | 1 |
| `low_vol` | **Dev (2008-19)** | -0.079 | Yes | 1.000 | [-0.139, +0.010] | -0.89% | -3.35% | 48.3% | -14.54% | +19.07% (comp: +164.1%) | 10 |
| `low_vol` | *Holdout (2024-25)* | -0.130 | Yes | 1.000 | N/A (<5 blocks) | -5.68% | -12.23% | 48.0% | -3.45% | +87.20% (comp: +87.2%) | 1 |
| `past_winner` | **Dev (2008-19)** | -0.045 | Yes | 1.000 | [-0.200, +0.044] | +4.27% | +1.80% | 50.3% | +15.40% | +19.07% (comp: +164.1%) | 10 |
| `past_winner` | *Holdout (2024-25)* | +0.108 | Yes | 0.000 | N/A (<5 blocks) | -8.72% | -15.27% | 58.0% | -12.57% | +87.20% (comp: +87.2%) | 1 |
| `equal_weight_market (benchmark)` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | - | +19.07% (comp: +164.1%) | 10 |
| `equal_weight_market (benchmark)` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | - | +87.20% (comp: +87.2%) | 1 |

---

## 3. Fold-Level (Annual) Evaluation Table (Dev Partition)

Fold-level summary across annual development blocks:

### Horizon: `SHORT` Annual Fold ICs

| Baseline Model | Folds with IC > 0 | Mean Fold IC | Fold t-stat (indicative: small n) |
| :--- | :---: | :---: | :---: |
| `random` | 7 / 12 | +0.003 | +0.77 |
| `reversal_21` | 5 / 12 | -0.004 | -0.27 |
| `momentum_63` | 5 / 12 | +0.019 | +0.83 |
| `momentum_126` | 8 / 12 | +0.036 | +1.56 |
| `low_vol` | 5 / 12 | -0.025 | -1.10 |
| `past_winner` | 5 / 12 | +0.010 | +0.35 |
| `equal_weight_universe` | N/A | N/A | N/A |

### Horizon: `MID` Annual Fold ICs

| Baseline Model | Folds with IC > 0 | Mean Fold IC | Fold t-stat (indicative: small n) |
| :--- | :---: | :---: | :---: |
| `random` | 8 / 12 | +0.002 | +0.69 |
| `reversal_21` | 5 / 12 | -0.035 | -1.69 |
| `momentum_63` | 8 / 12 | +0.050 | +1.33 |
| `momentum_126` | 9 / 12 | +0.066 | +1.74 |
| `low_vol` | 5 / 12 | -0.037 | -1.42 |
| `past_winner` | 6 / 12 | +0.023 | +0.43 |
| `equal_weight_universe` | N/A | N/A | N/A |

### Horizon: `LONG` Annual Fold ICs

| Baseline Model | Folds with IC > 0 | Mean Fold IC | Fold t-stat (indicative: small n) |
| :--- | :---: | :---: | :---: |
| `random` | 5 / 10 | -0.001 | -0.36 |
| `reversal_21` | 2 / 10 | -0.054 | -2.60 |
| `momentum_63` | 8 / 10 | +0.077 | +1.87 |
| `momentum_126` | 7 / 10 | +0.069 | +1.32 |
| `low_vol` | 2 / 10 | -0.079 | -2.20 |
| `past_winner` | 4 / 10 | -0.044 | -0.60 |
| `equal_weight_universe` | N/A | N/A | N/A |

---

## 4. Diagnostic Visualizations

### Annual Rank IC Evolution
![Rank IC by Test Year](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/baseline_mean_ic_by_year.png)

### Cumulative Portfolio Performance (Top-5 vs. Market)
![Cumulative Performance](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/baseline_cumulative_top5_vs_mkt.png)

---
*Generated by Stock Pulse Baseline & Validation Suite.*

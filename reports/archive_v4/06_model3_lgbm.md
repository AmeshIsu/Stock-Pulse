# Model 3 Research Report: LightGBM Ensembles, Feature Orthogonalization & Multi-Model Ensemble

**Document ID**: `reports/06_model3_lgbm.md`  
**Pipeline Step**: Model 3 (Boosting Ensembles & Synthesis)  
**Evaluation Rules**: Rigorous Part A Corrections Applied (Dual Null Band Envelope, Actual Replaced-Share Friction, Effective Sample Gating $n_{\text{eff}}$).

---

## 1. Executive Summary & Core Usability Verdicts

This report presents the rigorous evaluation of **Model 3 (LightGBM)** across three horizons: **Short (21 trading days)**, **Mid (63 trading days)**, and **Long (252 trading days)**. We benchmark five LightGBM configurations (`lgbm_reg_core`, `lgbm_reg_full`, `lgbm_reg_core_rel`, `lgbm_clf_core`, `lgbm_clf_core_rel`) against the 4-model equal-weight rank ensemble (`ens_rank4`), Random Forest (`rf_reg_core`), Ridge Regression (`ridge_core`), and the best naive baseline (`momentum_126`).

### Usability Verdict & Evidence Level by Horizon

| Horizon | Trading Days | Usability Verdict | App Evidence Level | Null Band Check | Beats Baseline $\ge 2/3$ Folds | Neutralized IC > 0 | Summary Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short** | 21 days | **RED** | **no reliable evidence** | Failed (Inside Null) | Failed | Failed | No ML model or baseline achieves statistically reliable predictability; transaction costs dominate. |
| **Mid** | 63 days | **AMBER** | **weak evidence** | Passed (Baseline Only) | Failed (ML models lose to 126d mom) | Passed | `momentum_126` has positive IC (+0.0658, $p=0.010$); ML models exhibit weak signal inside null band. |
| **Long** | 252 days | **RED** | **no reliable evidence** | Failed (Inside Null) | Failed | Marginal | $n_{\text{eff}} = 10$ on dev (indicative only); holdout has $n_{\text{eff}} = 1 < 5$ (not evaluable). |

> [!IMPORTANT]
> **Key Finding on Feature Orthogonalization (`core_rel`)**:
> Replacing raw liquidity and volatility metrics with their own-history z-scores (`_ownz`) completely eliminated between-stock identity leakage: between-stock variance share $R^2$ dropped from 0.44 (`log_turnover_21d`) to **0.0025** (`log_turnover_21d_ownz`), and from 0.46 (`cs_rank_vol_21d`) to **0.0006** (`vol_21d_ownz`). However, removing stock identity did not improve out-of-sample IC, proving that earlier models were learning persistent cross-sectional stock identity rather than genuine dynamic return predictability.

---

## 2. Horizon Evaluation Tables

### 2.1 Short Horizon (21 Trading Days)
| Model Candidate | Split | Mean IC | Corrected Null Band | Outside Null? (p-val) | $n_{\text{eff}}$ | Folds IC > 0 | Mean Fold IC (t-stat) | Top-5 Med Exc | Net Exc (Full) | Net Exc (Actual) | Replaced Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lgbm_reg_core` | **dev** | -0.0048 | [-0.0374, +0.0368] | No (p=0.609) | 138 | 6/12 | -0.0044 (t=-0.28) | +0.20% | -2.29% | -1.31% | 50.8% |
| `lgbm_reg_core` | **stress** | -0.0812 | [-0.1476, +0.1573] | No (p=0.836) * | 10 | 0/1 | N/A | +0.73% | -2.12% | -1.44% | 62.2% |
| `lgbm_reg_core` | **holdout** | +0.1494 | [-0.1611, +0.1398] | Yes (p=0.020) | 16 | 1/1 | N/A | +2.21% | -0.64% | +0.21% | 54.7% |
| `lgbm_reg_full` | **dev** | -0.0187 | [-0.0374, +0.0368] | No (p=0.832) | 138 | 4/12 | -0.0189 (t=-0.84) | +0.04% | -2.42% | -1.41% | 49.1% |
| `lgbm_reg_full` | **stress** | +0.0012 | [-0.1476, +0.1573] | No (p=0.523) * | 10 | 1/1 | N/A | +1.78% | -0.66% | +0.06% | 60.0% |
| `lgbm_reg_full` | **holdout** | +0.1056 | [-0.1963, +0.1779] | No (p=0.129) * | 10 | 1/1 | N/A | +1.96% | -0.32% | +0.28% | 66.7% |
| `lgbm_reg_core_rel` | **dev** | -0.0132 | [-0.0374, +0.0368] | No (p=0.764) | 138 | 7/12 | -0.0131 (t=-1.06) | +0.03% | -2.20% | -1.25% | 52.4% |
| `lgbm_reg_core_rel` | **stress** | -0.0194 | [-0.1476, +0.1573] | No (p=0.607) * | 10 | 0/1 | N/A | +1.37% | -0.83% | -0.63% | 88.9% |
| `lgbm_reg_core_rel` | **holdout** | +0.0782 | [-0.1963, +0.1779] | No (p=0.207) | 16 | 1/1 | N/A | +0.83% | -1.20% | -0.60% | 68.0% |
| `lgbm_clf_core` | **dev** | +0.0043 | [-0.0374, +0.0368] | No (p=0.401) | 138 | 5/12 | +0.0042 (t=+0.26) | +0.30% | -1.91% | -0.98% | 53.3% |
| `lgbm_clf_core` | **stress** | -0.0339 | [-0.1476, +0.1573] | No (p=0.673) * | 10 | 0/1 | N/A | +1.67% | +0.27% | +0.99% | 60.0% |
| `lgbm_clf_core` | **holdout** | +0.1096 | [-0.1963, +0.1779] | No (p=0.119) | 16 | 1/1 | N/A | +1.15% | -0.01% | +0.79% | 57.3% |
| `lgbm_clf_core_rel` | **dev** | +0.0034 | [-0.0374, +0.0368] | No (p=0.424) | 138 | 6/12 | +0.0033 (t=+0.17) | +0.23% | -2.24% | -1.34% | 54.6% |
| `lgbm_clf_core_rel` | **stress** | -0.1319 | [-0.1476, +0.1573] | No (p=0.952) * | 10 | 0/1 | N/A | +0.22% | -3.32% | -2.52% | 55.6% |
| `lgbm_clf_core_rel` | **holdout** | +0.0576 | [-0.1963, +0.1779] | No (p=0.274) | 16 | 1/1 | N/A | +1.01% | -2.27% | -1.54% | 61.3% |
| `ens_rank4` | **dev** | +0.0315 | [-0.0374, +0.0368] | No (p=0.049) | 138 | 10/12 | +0.0315 (t=+1.72) | +0.33% | -1.95% | -0.90% | 47.2% |
| `ens_rank4` | **stress** | +0.0105 | [-0.1476, +0.1573] | No (p=0.484) * | 10 | 1/1 | N/A | +2.19% | +0.05% | +0.85% | 55.6% |
| `ens_rank4` | **holdout** | +0.1966 | [-0.1611, +0.1398] | Yes (p=0.001) | 16 | 1/1 | N/A | +2.35% | -0.08% | +0.84% | 50.7% |
| `rf_reg_core` | **dev** | +0.0069 | [-0.0374, +0.0368] | No (p=0.350) | 138 | 6/12 | +0.0068 (t=+0.42) | +0.30% | -2.27% | -1.32% | 52.6% |
| `rf_reg_core` | **stress** | +0.0177 | [-0.1476, +0.1573] | No (p=0.453) * | 10 | 1/1 | N/A | +2.14% | -1.25% | -0.73% | 71.1% |
| `rf_reg_core` | **holdout** | +0.2068 | [-0.1611, +0.1398] | Yes (p=0.000) | 16 | 1/1 | N/A | +2.75% | -0.00% | +1.00% | 46.7% |
| `ridge_core` | **dev** | +0.0284 | [-0.0374, +0.0368] | No (p=0.070) | 138 | 8/12 | +0.0284 (t=+1.62) | +0.45% | -1.77% | -0.79% | 50.4% |
| `ridge_core` | **stress** | -0.0271 | [-0.1476, +0.1573] | No (p=0.645) * | 10 | 0/1 | N/A | +1.41% | -1.76% | -0.56% | 33.3% |
| `ridge_core` | **holdout** | +0.1913 | [-0.1611, +0.1398] | Yes (p=0.001) | 16 | 1/1 | N/A | +2.06% | -0.72% | +0.18% | 52.0% |
| `momentum_126` | **dev** | +0.0356 | [-0.0374, +0.0368] | No (p=0.031) | 138 | 8/12 | +0.0359 (t=+1.56) | +0.60% | -1.60% | -0.22% | 30.5% |
| `momentum_126` | **stress** | +0.0925 | [-0.1476, +0.1573] | No (p=0.176) * | 10 | 1/1 | N/A | +3.49% | +1.38% | +2.70% | 26.7% |
| `momentum_126` | **holdout** | +0.0381 | [-0.1611, +0.1398] | No (p=0.304) | 16 | 1/1 | N/A | +1.12% | -1.79% | -0.54% | 33.3% |

### 2.2 Mid Horizon (63 Trading Days)
| Model Candidate | Split | Mean IC | Corrected Null Band | Outside Null? (p-val) | $n_{\text{eff}}$ | Folds IC > 0 | Mean Fold IC (t-stat) | Top-5 Med Exc | Net Exc (Full) | Net Exc (Actual) | Replaced Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lgbm_reg_core` | **dev** | +0.0179 | [-0.0532, +0.0558] | No (p=0.249) | 46 | 6/12 | +0.0181 (t=+0.82) | +0.70% | -2.50% | -1.66% | 56.9% |
| `lgbm_reg_core` | **stress** | +0.0862 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +10.36% | +3.15% | +3.75% | 60.0% |
| `lgbm_reg_core` | **holdout** | +0.1834 | [-0.2772, +0.2602] | No (p=0.102) * | 5 | 1/1 | N/A | +6.26% | -0.77% | -0.37% | 75.0% |
| `lgbm_reg_full` | **dev** | +0.0029 | [-0.0532, +0.0558] | No (p=0.457) | 46 | 5/12 | +0.0029 (t=+0.13) | +0.74% | -2.23% | -1.30% | 52.0% |
| `lgbm_reg_full` | **stress** | +0.0453 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +10.70% | +4.85% | +5.65% | 46.7% |
| `lgbm_reg_full` | **holdout** | +0.2889 | [-0.2772, +0.2602] | not evaluable ($n_{\text{eff}} < 5$) | 3 | 1/1 | N/A | +10.43% | +1.00% | +1.27% | 80.0% |
| `lgbm_reg_core_rel` | **dev** | -0.0230 | [-0.0532, +0.0558] | No (p=0.814) | 46 | 4/12 | -0.0231 (t=-0.97) | +0.46% | -2.42% | -1.65% | 60.9% |
| `lgbm_reg_core_rel` | **stress** | -0.0996 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 0/1 | N/A | +2.76% | -0.35% | -0.05% | 80.0% |
| `lgbm_reg_core_rel` | **holdout** | +0.1121 | [-0.2772, +0.2602] | No (p=0.215) * | 5 | 1/1 | N/A | +1.52% | -2.15% | -1.83% | 80.0% |
| `lgbm_clf_core` | **dev** | +0.0046 | [-0.0532, +0.0558] | No (p=0.435) | 46 | 7/12 | +0.0048 (t=+0.20) | +0.87% | -1.60% | -0.67% | 52.9% |
| `lgbm_clf_core` | **stress** | -0.0291 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 0/1 | N/A | +3.96% | +1.03% | +1.63% | 60.0% |
| `lgbm_clf_core` | **holdout** | +0.1491 | [-0.2772, +0.2602] | No (p=0.151) * | 5 | 1/1 | N/A | +2.76% | +1.68% | +2.08% | 75.0% |
| `lgbm_clf_core_rel` | **dev** | -0.0043 | [-0.0532, +0.0558] | No (p=0.580) | 46 | 6/12 | -0.0044 (t=-0.18) | +0.64% | -2.40% | -1.71% | 64.4% |
| `lgbm_clf_core_rel` | **stress** | -0.0904 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 0/1 | N/A | +2.75% | -0.53% | -0.13% | 73.3% |
| `lgbm_clf_core_rel` | **holdout** | +0.1638 | [-0.2772, +0.2602] | No (p=0.125) * | 5 | 1/1 | N/A | +3.60% | +0.42% | +1.06% | 60.0% |
| `ens_rank4` | **dev** | +0.0429 | [-0.0532, +0.0558] | No (p=0.064) | 46 | 8/12 | +0.0433 (t=+1.41) | +1.49% | -1.58% | -0.62% | 51.1% |
| `ens_rank4` | **stress** | +0.0984 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +10.97% | +1.55% | +2.25% | 53.3% |
| `ens_rank4` | **holdout** | +0.3176 | [-0.2772, +0.2602] | Yes (p=0.002) * | 5 | 1/1 | N/A | +7.00% | +4.78% | +5.74% | 40.0% |
| `rf_reg_core` | **dev** | +0.0250 | [-0.0532, +0.0558] | No (p=0.184) | 46 | 9/12 | +0.0252 (t=+1.16) | +1.15% | -2.83% | -2.01% | 58.2% |
| `rf_reg_core` | **stress** | +0.0919 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +8.32% | -0.24% | +0.26% | 66.7% |
| `rf_reg_core` | **holdout** | +0.2984 | [-0.2772, +0.2602] | Yes (p=0.005) * | 5 | 1/1 | N/A | +7.13% | +5.10% | +5.90% | 50.0% |
| `ridge_core` | **dev** | +0.0144 | [-0.0532, +0.0558] | No (p=0.300) | 46 | 7/12 | +0.0146 (t=+0.52) | +0.84% | -1.54% | -0.69% | 56.9% |
| `ridge_core` | **stress** | +0.0706 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +8.12% | +1.28% | +1.98% | 53.3% |
| `ridge_core` | **holdout** | +0.3304 | [-0.2772, +0.2602] | Yes (p=0.001) * | 5 | 1/1 | N/A | +5.26% | +2.44% | +3.32% | 45.0% |
| `momentum_126` | **dev** | +0.0658 | [-0.0532, +0.0558] | Yes (p=0.010) | 46 | 9/12 | +0.0662 (t=+1.74) | +1.60% | -0.75% | +0.30% | 46.2% |
| `momentum_126` | **stress** | +0.0640 | [-0.2055, +0.1817] | not evaluable ($n_{\text{eff}} < 5$) | 4 | 1/1 | N/A | +11.68% | +2.46% | +3.26% | 46.7% |
| `momentum_126` | **holdout** | +0.1135 | [-0.2772, +0.2602] | No (p=0.212) * | 5 | 1/1 | N/A | +3.71% | -0.46% | +0.10% | 65.0% |

### 2.3 Long Horizon (252 Trading Days)
| Model Candidate | Split | Mean IC | Corrected Null Band | Outside Null? (p-val) | $n_{\text{eff}}$ | Folds IC > 0 | Mean Fold IC (t-stat) | Top-5 Med Exc | Net Exc (Full) | Net Exc (Actual) | Replaced Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lgbm_reg_core` | **dev** | +0.0451 | [-0.1252, +0.1231] | No (p=0.224) * | 10 | 6/10 | +0.0451 (t=+1.64) | +2.26% | -12.65% | -11.97% | 62.2% |
| `lgbm_reg_core` | **holdout** | +0.2929 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +51.95% | +10.51% | +10.51% | 100.0% |
| `lgbm_reg_full` | **dev** | +0.0763 | [-0.1252, +0.1231] | No (p=0.095) * | 10 | 7/10 | +0.0764 (t=+2.68) | +4.16% | +6.51% | +7.35% | 53.3% |
| `lgbm_reg_core_rel` | **dev** | +0.0114 | [-0.1252, +0.1231] | No (p=0.412) * | 10 | 6/10 | +0.0111 (t=+0.49) | +3.40% | -16.73% | -16.09% | 64.4% |
| `lgbm_reg_core_rel` | **holdout** | +0.3766 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +55.65% | +7.45% | +7.45% | 100.0% |
| `lgbm_clf_core` | **dev** | +0.0932 | [-0.1252, +0.1231] | No (p=0.062) * | 10 | 8/10 | +0.0930 (t=+2.86) | +4.56% | +1.61% | +2.53% | 48.9% |
| `lgbm_clf_core` | **holdout** | +0.4267 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +67.72% | -5.72% | -5.72% | 100.0% |
| `lgbm_clf_core_rel` | **dev** | +0.0291 | [-0.1252, +0.1231] | No (p=0.312) * | 10 | 8/10 | +0.0294 (t=+1.15) | +2.12% | -14.31% | -13.91% | 77.8% |
| `lgbm_clf_core_rel` | **holdout** | +0.3448 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +58.99% | +7.73% | +7.73% | 100.0% |
| `ens_rank4` | **dev** | +0.1143 | [-0.1252, +0.1231] | No (p=0.032) * | 10 | 9/10 | +0.1145 (t=+2.79) | +6.76% | +5.32% | +5.88% | 68.9% |
| `ens_rank4` | **holdout** | +0.3939 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +58.84% | +4.12% | +4.12% | 100.0% |
| `rf_reg_core` | **dev** | +0.0595 | [-0.1252, +0.1231] | No (p=0.157) * | 10 | 7/10 | +0.0594 (t=+2.22) | +4.66% | +4.59% | +5.47% | 51.1% |
| `rf_reg_core` | **holdout** | +0.3754 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +62.98% | +10.69% | +10.69% | 100.0% |
| `ridge_core` | **dev** | +0.0694 | [-0.1252, +0.1231] | No (p=0.113) * | 10 | 7/10 | +0.0697 (t=+1.48) | +4.71% | -8.84% | -8.20% | 64.4% |
| `ridge_core` | **holdout** | +0.4780 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 1/1 | N/A | +63.64% | +29.53% | +29.53% | 100.0% |
| `momentum_126` | **dev** | +0.0682 | [-0.1252, +0.1231] | No (p=0.121) * | 10 | 7/10 | +0.0686 (t=+1.32) | +4.83% | -6.87% | -6.43% | 75.6% |
| `momentum_126` | **holdout** | -0.0127 | [-0.4294, +0.4185] | not evaluable ($n_{\text{eff}} < 5$) | 1 | 0/1 | N/A | +27.52% | -3.29% | -3.29% | 100.0% |

---

## 3. Visualizations

### 3.1 Walk-Forward Annual IC Progression (Mid Horizon)
![Annual IC Comparison](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/06_lgbm_annual_ic_comparison.png)
*Figure 1: Mean daily Rank IC by walk-forward test year across models on the 63-day horizon, plotted against the corrected dual null band envelope.*

### 3.2 TreeSHAP Feature Importance
![SHAP Importance](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/06_lgbm_shap_importance.png)
*Figure 2: Mean absolute TreeSHAP contributions by feature family on DEV test folds, highlighting stock-identity proxy flags.*

### 3.3 Raw vs Neutralized Rank IC
![Raw vs Neutralized IC](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/06_lgbm_neutralized_vs_raw_ic.png)
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

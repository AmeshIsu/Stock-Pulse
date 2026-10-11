# Stock Pulse: Validation Framework & Cross-Validation Architecture

**Document**: `reports/03_validation_design.md`  
**Scope**: In-Universe Equity Series (`MainType` N & X selected via `most_rows_among_N_X`)  
**Core Standard**: Expanding-Window Walk-Forward Cross-Validation with Strict Purging and Embargo.

---

## 1. Validation Protocol & Partition Roles

To strictly eliminate lookahead bias and guard against overfitting, data partitions are assigned strict roles:
1. **Development Folds (Test Years 2008–2019)**:
   - Used for model exploration, feature selection, hyperparameter tuning, and threshold calibration.
   - Expanding-window annual test blocks: each year is evaluated strictly out-of-sample after training on all prior history.
2. **Stress-Test Fold (Test Year 2020)**:
   - The COVID-19 pandemic market crisis.
   - Evaluated as an extreme shock stress fold (Short and Mid horizons only). **Never** used for parameter tuning.
3. **Final Holdout Fold (Test Years 2024–2025)**:
   - Evaluated as the final forward holdout block post-crisis.
   - Reported transparently for every candidate model, but **NEVER** used for model selection, feature choice, or threshold tuning.

---

## 2. Purging and Embargo Mechanics

Standard cross-validation creates severe data leakage in financial time series due to forward-looking overlapping return targets.
- **The Purge Rule**:
  For horizon $h$, each row $t$ has target return spanning $[t, t+h]$. A training row is purged if its forward window ends on or after the test block start date:
  `target_end_date_t >= test_start_date`
- **The Embargo Buffer**:
  An embargo of **5 trading days** (1 calendar week) is enforced immediately prior to test block onset to eliminate high-frequency serial correlation across the partition boundary:
  `target_end_date_t >= (test_start_date - embargo)`

---

## 3. Walk-Forward Fold Table

Total Folds Generated: **39 folds** across 3 horizons.

| Horizon | Fold Name | Split Type | Train Start | Train End | Train Rows (Purged) | Test Start | Test End | Test Rows | Avg Stocks / Date | Test Dates |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| short | `dev_2008` | dev | 2002-01-02 | 2007-11-21 | 16,881 | 2008-01-02 | 2008-12-31 | 3,111 | 13.1 | 238 |
| short | `dev_2009` | dev | 2002-01-02 | 2008-11-19 | 20,041 | 2009-01-02 | 2009-12-30 | 3,215 | 13.4 | 240 |
| short | `dev_2010` | dev | 2002-01-02 | 2009-11-20 | 23,208 | 2010-01-04 | 2010-12-31 | 3,777 | 15.9 | 238 |
| short | `dev_2011` | dev | 2002-01-02 | 2010-11-24 | 26,956 | 2011-01-03 | 2011-12-30 | 3,746 | 15.7 | 239 |
| short | `dev_2012` | dev | 2002-01-02 | 2011-11-23 | 30,679 | 2012-01-02 | 2012-12-31 | 4,159 | 17.2 | 242 |
| short | `dev_2013` | dev | 2002-01-02 | 2012-11-20 | 34,815 | 2013-01-02 | 2013-12-31 | 4,291 | 17.7 | 242 |
| short | `dev_2014` | dev | 2002-01-02 | 2013-11-21 | 39,102 | 2014-01-02 | 2014-12-31 | 4,301 | 17.9 | 241 |
| short | `dev_2015` | dev | 2002-01-02 | 2014-11-24 | 43,406 | 2015-01-02 | 2015-12-31 | 4,202 | 17.6 | 239 |
| short | `dev_2016` | dev | 2002-01-02 | 2015-11-20 | 47,603 | 2016-01-04 | 2016-12-30 | 4,206 | 17.5 | 240 |
| short | `dev_2017` | dev | 2002-01-02 | 2016-11-21 | 51,815 | 2017-01-02 | 2017-12-29 | 4,507 | 18.7 | 241 |
| short | `dev_2018` | dev | 2002-01-02 | 2017-11-21 | 56,302 | 2018-01-02 | 2018-12-31 | 4,433 | 18.5 | 240 |
| short | `dev_2019` | dev | 2002-01-02 | 2018-11-21 | 60,732 | 2019-01-02 | 2019-12-31 | 4,527 | 18.8 | 241 |
| short | `stress_2020` | stress | 2002-01-02 | 2019-11-21 | 65,250 | 2020-01-02 | 2020-12-31 | 4,015 | 19.2 | 209 |
| short | `holdout_2024_2025` | holdout | 2002-01-02 | 2022-02-21 | 71,047 | 2024-01-02 | 2025-11-28 | 8,898 | 19.5 | 457 |
| mid | `dev_2008` | dev | 2002-01-02 | 2007-09-19 | 15,996 | 2008-01-02 | 2008-12-31 | 3,025 | 12.7 | 238 |
| mid | `dev_2009` | dev | 2002-01-02 | 2008-09-16 | 19,135 | 2009-01-02 | 2009-12-30 | 3,215 | 13.4 | 240 |
| mid | `dev_2010` | dev | 2002-01-02 | 2009-09-22 | 22,260 | 2010-01-04 | 2010-12-31 | 3,737 | 15.7 | 238 |
| mid | `dev_2011` | dev | 2002-01-02 | 2010-09-20 | 25,924 | 2011-01-03 | 2011-12-30 | 3,744 | 15.7 | 239 |
| mid | `dev_2012` | dev | 2002-01-02 | 2011-09-20 | 29,621 | 2012-01-02 | 2012-12-31 | 4,159 | 17.2 | 242 |
| mid | `dev_2013` | dev | 2002-01-02 | 2012-09-19 | 33,657 | 2013-01-02 | 2013-12-31 | 4,291 | 17.7 | 242 |
| mid | `dev_2014` | dev | 2002-01-02 | 2013-09-20 | 37,944 | 2014-01-02 | 2014-12-31 | 4,301 | 17.9 | 241 |
| mid | `dev_2015` | dev | 2002-01-02 | 2014-09-19 | 42,248 | 2015-01-02 | 2015-12-31 | 4,202 | 17.6 | 239 |
| mid | `dev_2016` | dev | 2002-01-02 | 2015-09-18 | 46,445 | 2016-01-04 | 2016-12-30 | 4,206 | 17.5 | 240 |
| mid | `dev_2017` | dev | 2002-01-02 | 2016-09-21 | 50,657 | 2017-01-02 | 2017-12-29 | 4,507 | 18.7 | 241 |
| mid | `dev_2018` | dev | 2002-01-02 | 2017-09-19 | 55,102 | 2018-01-02 | 2018-12-31 | 4,433 | 18.5 | 240 |
| mid | `dev_2019` | dev | 2002-01-02 | 2018-09-18 | 59,532 | 2019-01-02 | 2019-12-31 | 4,487 | 18.6 | 241 |
| mid | `stress_2020` | stress | 2002-01-02 | 2019-09-20 | 64,050 | 2020-01-02 | 2020-12-22 | 3,822 | 18.7 | 204 |
| mid | `holdout_2024_2025` | holdout | 2002-01-02 | 2020-12-22 | 69,187 | 2024-01-02 | 2025-09-26 | 8,058 | 19.4 | 415 |
| long | `dev_2010` | dev | 2002-01-02 | 2008-12-02 | 18,833 | 2010-01-04 | 2010-12-31 | 3,636 | 15.3 | 238 |
| long | `dev_2011` | dev | 2002-01-02 | 2009-12-02 | 22,185 | 2011-01-03 | 2011-12-30 | 3,744 | 15.7 | 239 |
| long | `dev_2012` | dev | 2002-01-02 | 2010-12-06 | 25,809 | 2012-01-02 | 2012-12-31 | 4,159 | 17.2 | 242 |
| long | `dev_2013` | dev | 2002-01-02 | 2011-12-08 | 29,484 | 2013-01-02 | 2013-12-31 | 4,291 | 17.7 | 242 |
| long | `dev_2014` | dev | 2002-01-02 | 2012-12-06 | 33,581 | 2014-01-02 | 2014-12-31 | 4,301 | 17.9 | 241 |
| long | `dev_2015` | dev | 2002-01-02 | 2013-12-05 | 37,869 | 2015-01-02 | 2015-12-31 | 4,202 | 17.6 | 239 |
| long | `dev_2016` | dev | 2002-01-02 | 2014-12-04 | 42,066 | 2016-01-04 | 2016-12-30 | 4,206 | 17.5 | 240 |
| long | `dev_2017` | dev | 2002-01-02 | 2015-12-04 | 46,278 | 2017-01-02 | 2017-12-29 | 4,507 | 18.7 | 241 |
| long | `dev_2018` | dev | 2002-01-02 | 2016-12-05 | 50,551 | 2018-01-02 | 2018-12-31 | 4,397 | 18.3 | 240 |
| long | `dev_2019` | dev | 2002-01-02 | 2017-12-05 | 54,964 | 2019-01-02 | 2019-12-31 | 4,058 | 16.8 | 241 |
| long | `holdout_2024_2025` | holdout | 2002-01-02 | 2020-01-24 | 64,323 | 2024-01-02 | 2024-12-10 | 4,278 | 18.9 | 226 |

---

## 4. Per-Series Coverage & Universe Selection Analysis

Universe rule: `most_rows_among_N_X` selects the dominant share class per company (ties to N).

| Series ID | Company Code | Total Rows | Valid Core Features | Valid Short Targets | Valid Mid Targets | Valid Long Targets | Data Status / Flags |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `AEL.N` | AEL | 2,671 | 2,059 | 2,608 | 2,487 | 2,109 | Normal |
| `CFIN.N` | CFIN | 4,782 | 3,800 | 4,689 | 4,526 | 3,957 | Normal |
| `COMB.N` | COMB | 4,272 | 3,482 | 4,201 | 4,083 | 3,705 | Normal |
| `DFCC.N` | DFCC | 4,976 | 4,094 | 4,900 | 4,779 | 4,399 | Normal |
| `DIAL.N` | DIAL | 4,267 | 3,656 | 4,204 | 4,083 | 3,705 | Normal |
| `HAYL.N` | HAYL | 4,819 | 3,961 | 4,746 | 4,625 | 4,245 | Normal |
| `HHL.N` | HHL | 4,481 | 3,811 | 4,416 | 4,295 | 3,917 | Normal |
| `HNB.N` | HNB | 2,796 | 1,702 | 2,638 | 2,366 | 1,596 | FLAG: Limited trading rows (interleaved share class history) |
| `JKH.N` | JKH | 4,958 | 3,727 | 4,877 | 4,756 | 4,375 | Normal |
| `LFIN.N` | LFIN | 4,231 | 3,225 | 4,106 | 3,933 | 3,368 | Normal |
| `LIOC.N` | LIOC | 4,399 | 3,788 | 4,336 | 4,215 | 3,837 | Normal |
| `LOLC.N` | LOLC | 4,697 | 3,913 | 4,628 | 4,539 | 4,159 | Normal |
| `MELS.N` | MELS | 1,501 | 1,191 | 1,438 | 1,317 | 939 | Normal |
| `NDB.N` | NDB | 5,025 | 4,119 | 4,945 | 4,824 | 4,443 | Normal |
| `NTB.N` | NTB | 4,827 | 3,796 | 4,725 | 4,562 | 4,026 | Normal |
| `PLC.N` | PLC | 2,754 | 2,142 | 2,691 | 2,570 | 2,192 | Normal |
| `SAMP.N` | SAMP | 4,963 | 3,857 | 4,890 | 4,769 | 4,389 | Normal |
| `SEYB.X` | SEYB | 3,901 | 3,316 | 3,836 | 3,730 | 3,352 | Normal |
| `SINS.N` | SINS | 4,466 | 3,713 | 4,402 | 4,281 | 3,902 | Normal |
| `SUN.N` | SUN | 3,037 | 2,081 | 2,766 | 2,505 | 1,986 | Normal |

### Specific Coverage Findings:
- **`SEYB.X` (Seylan Bank Non-Voting)**: Uses `SEYB.X` (**3,901 rows**, 3,316 valid core features) instead of `SEYB.N` (1,166 rows), resolving the data sparsity problem.
- **`HNB.N` (Hatton National Bank Voting)**: Has **2,796 voting rows** (selected over `HNB.X` 2,231 rows).
- **Threshold**: All 20 selected series satisfy `min_valid_feature_rows_for_scoring >= 750`.

---

## 5. Evaluation Metrics & Financial Assumptions

1. **Rank IC (Information Coefficient)**: Daily cross-sectional Spearman rank correlation between predicted score and realized market-excess return (`fwd_excess`).
2. **Moving-Block Bootstrap (95% CI)**: Block length equals horizon $h$ trading days with **2,000 resamples**. Standard t-statistics are invalid due to overlapping forward return windows. If `effective_blocks = test_days / h < 5`, CI is marked `N/A (too few independent blocks)`.
3. **Top-K Metrics ($K = 5$)**: Median excess return, 1%/99% winsorized mean excess return, tie-adjusted precision, and top-K minus bottom-K spread.
4. **Transaction Cost Assumption**:
   > [!IMPORTANT]
   > **Trading Friction Assumption**: Round-trip transaction cost of **200 bps (2.0%)** per rebalance is applied to simulated portfolios. This is an **ASSUMPTION** reflecting Sri Lankan brokerage commissions, clearing levies, taxes, and liquidity slippage.

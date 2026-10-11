# Stock Pulse: Initial Data Audit Report (CSE S&P SL20)

**Dataset Scope**: Colombo Stock Exchange (CSE) S&P SL20 Companies (2002 – 2025)  
**Total Records Analyzed**: 86,950 rows across 20 constituents.  
**Source Sheet**: `Full_Dataset_2001_2025`  

---

## 1. Executive Summary & Critical Findings
1. **Severe Temporal Gap & Regime Discontinuity**:
   - **2021 & 2022 Sparse Trading**: Only ~1,145 rows in 2021 and ~1,128 rows in 2022 (~1 quarter of trading per year) compared to normal annual volume (~4,500 rows).
   - **2023 Complete Absence**: The calendar year 2023 is entirely missing from the dataset. Across all 20 companies, there is a uniform **642-day continuous void** spanning from **2022-03-31 to 2024-01-02**.
   - **Market Halts & Segmentation**: Trading records must be strictly partitioned into segments (`gap_days > 30` calendar days) to prevent rolling technical windows and forward return targets from bridging massive macroeconomic voids.
2. **Extreme Daily Return Anomalies (2,175 moves > 50%)**:
   - There are **4,482 days with absolute return > 20%** and **2,175 days exceeding 50%**.
   - Analysis reveals that in the raw dataset, ordinary voting shares (`MainType = 'N'`) and non-voting shares (`MainType = 'X'`) were interleaved under the same `CompanyCode` on alternating days. Because non-voting shares trade at ~40–50% discount to voting shares, alternating rows produce false swings of +100% or -50%.
3. **Unadjusted Stock Splits Detected**:
   - Genuine corporate subdivision actions exist in the raw prices (e.g., JKH 10:1 split on 2024-11-06, LOLC 10:1 split on 2010-11-22, SUN 4:1 split on 2025-02-19). Backward-adjustment is required to eliminate artificial 80–90% price drops.
4. **Data Integrity Checks**:
   - Zero or negative prices: **0**.
   - High < Low errors: **0**.
   - Close outside [Low, High]: **550 rows** (indicating intra-day recording errors).
   - Zero trade volume: **0 rows**.

---

## 2. Company Coverage & Date Ranges
| Company Code | Official Name | Trading Days | Start Date | End Date | Share Types |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AEL | Access Engineering PLC | 2,671 | 2012-03-27 | 2025-12-31 | N |
| CFIN | Central Finance Company PLC | 4,783 | 2002-01-02 | 2025-12-31 | N/R |
| COMB | Commercial Bank of Ceylon PLC | 5,107 | 2002-01-02 | 2025-12-31 | X/N/P/R |
| DFCC | DFCC Bank PLC | 4,985 | 2002-01-02 | 2025-12-31 | N/B/R |
| DIAL | Dialog Axiata PLC | 4,269 | 2005-07-28 | 2025-12-31 | N/R |
| HAYL | Hayleys PLC | 4,820 | 2002-01-02 | 2025-12-31 | N/R |
| HHL | Hemas Holdings PLC | 4,487 | 2003-10-15 | 2025-12-31 | N/R |
| HNB | Hatton National Bank PLC | 5,111 | 2002-01-02 | 2025-12-31 | N/X/D/R |
| JKH | John Keells Holdings PLC | 5,106 | 2002-01-02 | 2025-12-31 | N/R/W |
| LFIN | LB Finance PLC | 4,540 | 2002-01-03 | 2025-12-31 | N/W/B/R |
| LIOC | Lanka IOC Ltd | 4,399 | 2004-12-22 | 2025-12-31 | N |
| LOLC | LOLC Holdings PLC | 4,698 | 2002-01-02 | 2025-12-31 | N/B |
| MELS | Melstacorp PLC | 1,501 | 2016-12-30 | 2025-12-31 | N |
| NDB | National Development Bank PLC | 5,032 | 2002-01-02 | 2025-12-31 | N/B/R |
| NTB | Nations Trust Bank PLC | 5,086 | 2002-01-02 | 2025-12-31 | N/R/W/X |
| PLC | Peoples Leasing & Finance PLC | 2,754 | 2011-11-24 | 2025-12-31 | N |
| SAMP | Sampath Bank PLC | 4,988 | 2002-01-02 | 2025-12-31 | N/D/R |
| SEYB | Seylan Bank PLC | 5,110 | 2002-01-02 | 2025-12-31 | N/D/X/R |
| SINS | Singer Sri Lanka PLC | 4,466 | 2002-01-02 | 2025-12-31 | N |
| SUN | Sunshine Holdings PLC | 3,037 | 2002-05-03 | 2025-12-31 | N |

---

## 3. Annual Trading Days & Heatmap
![Trading Days per Company per Year](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/rows_per_company_year_heatmap.png)

### Annual Total Row Distribution
| Year | Total Records | Observations |
| :--- | :--- | :--- |
| 2002 | 2,794 | Earlier history; partial constituent listings |
| 2003 | 2,952 | Earlier history; partial constituent listings |
| 2004 | 3,153 | Normal trading coverage |
| 2005 | 3,615 | Normal trading coverage |
| 2006 | 3,657 | Normal trading coverage |
| 2007 | 3,519 | Normal trading coverage |
| 2008 | 3,541 | Normal trading coverage |
| 2009 | 3,607 | Normal trading coverage |
| 2010 | 3,950 | Normal trading coverage |
| 2011 | 4,030 | Normal trading coverage |
| 2012 | 4,412 | Normal trading coverage |
| 2013 | 4,490 | Normal trading coverage |
| 2014 | 4,492 | Normal trading coverage |
| 2015 | 4,424 | Normal trading coverage |
| 2016 | 4,409 | Normal trading coverage |
| 2017 | 4,681 | Normal trading coverage |
| 2018 | 4,598 | Normal trading coverage |
| 2019 | 4,664 | Normal trading coverage |
| 2020 | 4,145 | Normal trading coverage |
| 2021 | 1,145 | **Sparse coverage** (~57 trading days per company; Q2-Q4 missing) |
| 2022 | 1,128 | **Sparse coverage** (~57 trading days per company; Q2-Q4 missing) |
| 2024 | 4,794 | Normal trading coverage |
| 2025 | 4,750 | Normal trading coverage |
| **2023** | **0** | **Completely absent (642-day gap across all 20 companies)** |

---

## 4. Data Integrity & Missing Values Audit

### Column Null Counts
| Column | Missing Count | Data Status |
| :--- | :--- | :--- |
| `Date` | 0 | Clean |
| `CompanyCode` | 0 | Clean |
| `Close` | 0 | Clean (100% present) |
| `Open` | 52,661 | High nulls (reconstruct or rely primarily on Close/High/Low) |
| `High` | 13 | Minor nulls (13 rows) |
| `Low` | 47 | Minor nulls (47 rows) |
| `TradeVolume` | 0 | Clean |
| `ShareVolume` | 12 | Minor nulls (12 rows) |
| `Turnover` | 4,421 | 4,421 nulls |

### Core Integrity Checks
- **Duplicates on `(Date, CompanyCode)`**: `0` (No duplicate daily records found).
- **High < Low violation**: `0` cases.
- **Close outside `[Low, High]`**: `550` cases (bounded to High/Low during cleaning).
- **Negative or Zero prices**: `0` cases.

---

## 5. Extreme Moves Distribution & Unadjusted Splits
![Distribution of Raw Returns](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/extreme_moves_distribution.png)

- **Total moves with |Return| > 20%**: `4,482`
- **Total moves with |Return| > 50%**: `2,175`

### Top Extreme Price Moves (Sample)
| Company | Date | Prev Close | Close | Raw Return | Price Ratio | Volume Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SAMP | 2019-06-07 | 0.03 | 45.70 | +137137.2% | 0.00 | -99% |
| NDB | 2018-10-04 | 0.10 | 103.90 | +103800.0% | 0.00 | -99% |
| HNB | 2011-07-25 | 0.20 | 203.50 | +101650.0% | 0.00 | -99% |
| NTB | 2018-01-30 | 0.10 | 79.80 | +79700.0% | 0.00 | -100% |
| NTB | 2018-01-26 | 0.10 | 79.00 | +78900.0% | 0.00 | -95% |
| DFCC | 2019-04-18 | 0.10 | 71.20 | +71100.0% | 0.00 | -99% |
| HNB | 2017-07-26 | 1.10 | 190.00 | +17172.7% | 0.01 | -89% |
| SEYB | 2006-05-24 | 0.10 | 12.75 | +12650.0% | 0.01 | -95% |
| SEYB | 2011-06-02 | 0.60 | 73.40 | +12133.3% | 0.01 | -97% |
| SAMP | 2005-01-17 | 0.27 | 20.75 | +7680.3% | 0.01 | -100% |
| HNB | 2017-07-19 | 3.00 | 229.90 | +7563.3% | 0.01 | +4182% |
| JKH | 2016-09-23 | 2.10 | 150.10 | +7047.6% | 0.01 | +165% |
| SAMP | 2005-01-06 | 0.33 | 20.58 | +6075.6% | 0.02 | -95% |
| HHL | 2015-05-07 | 1.50 | 77.60 | +5073.3% | 0.02 | -81% |
| JKH | 2016-09-20 | 2.90 | 148.90 | +5034.5% | 0.02 | +21% |

### Candidate Unadjusted Stock Splits (Tolerance = 5%)
Total moves matching canonical split factors (`2, 3, 4, 5, 10` or inverse): **546 events**.

---

## 6. Forward-Looking Window Usability Report
Forward windows are strictly constrained within each continuous segment (`gap_days <= 30`). No rolling window or target is permitted to cross a segment boundary.

| Year | Total Rows | Short (21d) Valid | Short % | Mid (63d) Valid | Mid % | Long (252d) Valid | Long % | Status / Usability Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2002 | 2,794 | 2,778 | 99.4% | 2,778 | 99.4% | 2,778 | 99.4% | Fully Usable for Short & Mid |
| 2003 | 2,952 | 2,941 | 99.6% | 2,941 | 99.6% | 2,941 | 99.6% | Fully Usable for Short & Mid |
| 2004 | 3,153 | 3,132 | 99.3% | 3,086 | 97.9% | 3,086 | 97.9% | Fully Usable for Short & Mid |
| 2005 | 3,615 | 3,573 | 98.8% | 3,528 | 97.6% | 3,528 | 97.6% | Fully Usable for Short & Mid |
| 2006 | 3,657 | 3,628 | 99.2% | 3,628 | 99.2% | 3,628 | 99.2% | Fully Usable for Short & Mid |
| 2007 | 3,519 | 3,497 | 99.4% | 3,492 | 99.2% | 3,417 | 97.1% | Fully Usable for Short & Mid |
| 2008 | 3,541 | 3,485 | 98.4% | 3,441 | 97.2% | 3,327 | 94.0% | Fully Usable for Short & Mid |
| 2009 | 3,607 | 3,596 | 99.7% | 3,596 | 99.7% | 3,596 | 99.7% | Fully Usable for Short & Mid |
| 2010 | 3,950 | 3,950 | 100.0% | 3,950 | 100.0% | 3,950 | 100.0% | Fully Usable for Short & Mid |
| 2011 | 4,030 | 4,030 | 100.0% | 4,030 | 100.0% | 4,030 | 100.0% | Fully Usable for Short & Mid |
| 2012 | 4,412 | 4,412 | 100.0% | 4,412 | 100.0% | 4,412 | 100.0% | Fully Usable for Short & Mid |
| 2013 | 4,490 | 4,490 | 100.0% | 4,490 | 100.0% | 4,490 | 100.0% | Fully Usable for Short & Mid |
| 2014 | 4,492 | 4,492 | 100.0% | 4,492 | 100.0% | 4,492 | 100.0% | Fully Usable for Short & Mid |
| 2015 | 4,424 | 4,424 | 100.0% | 4,424 | 100.0% | 4,424 | 100.0% | Fully Usable for Short & Mid |
| 2016 | 4,409 | 4,409 | 100.0% | 4,409 | 100.0% | 4,409 | 100.0% | Fully Usable for Short & Mid |
| 2017 | 4,681 | 4,681 | 100.0% | 4,681 | 100.0% | 4,681 | 100.0% | Fully Usable for Short & Mid |
| 2018 | 4,598 | 4,598 | 100.0% | 4,598 | 100.0% | 4,562 | 99.2% | Fully Usable for Short & Mid |
| 2019 | 4,664 | 4,664 | 100.0% | 4,346 | 93.2% | 602 | 12.9% | **Severely Degraded for Long** (only 12.9% valid) |
| 2020 | 4,145 | 3,725 | 89.9% | 3,088 | 74.5% | 0 | 0.0% | **UNUSABLE for Long** (0% valid due to 2021 gap) |
| 2021 | 1,145 | 725 | 63.3% | 0 | 0.0% | 0 | 0.0% | **UNUSABLE for Mid/Long**; Short partially viable (63%) |
| 2022 | 1,128 | 708 | 62.8% | 0 | 0.0% | 0 | 0.0% | **UNUSABLE for Mid/Long**; Short partially viable (63%) |
| 2024 | 4,794 | 4,794 | 100.0% | 4,794 | 100.0% | 4,504 | 94.0% | Fully Usable for Short & Mid |
| 2025 | 4,750 | 4,330 | 91.2% | 3,490 | 73.5% | 0 | 0.0% | **UNUSABLE for Long** (terminal dataset boundary; pending forward days) |

### Usability Takeaways for Modeling:
1. **Short Horizon (21 trading days)**: Highly usable across 2002–2020 and 2024–2025 (>90–100%). Weak in 2021 (~63%) and 2022 (~63%).
2. **Mid Horizon (63 trading days)**: Fully usable across 2002–2020 and 2024 (~93–100%). **Entirely unusable in 2021 and 2022 (0% valid forward windows)**.
3. **Long Horizon (252 trading days)**: Usable across 2002–2018 and 2024. **Completely unusable in 2019 (12.9%), 2020 (0%), 2021 (0%), 2022 (0%), and 2025 (0%)**.

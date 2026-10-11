# Stock Pulse: Share Class Separation & Revised Gap Architecture Report

**Document**: `reports/02_share_class_fix.md`  
**Purpose**: Document the elimination of share-class alternation artifacts, corporate stock split adjustments, market-wide closure bridging, and revised usability matrix.

---

## 1. Share Class Breakdown Per Company (`series_id = CompanyCode.MainType`)

In the Colombo Stock Exchange (CSE), tickers encompass multiple security types under the same issuer:
- `N`: Ordinary Voting Shares (Primary institutional ranking universe)
- `X`: Non-Voting Shares
- `R`: Rights Issues
- `W`: Warrants
- `D`: Debentures
- `P`: Preference Shares
- `B`: Corporate Bonds

All 20 companies possess an active `N` voting series. Series with `MainType != 'N'` are preserved in `prices_clean.parquet` with `in_ranking_universe = False` for transparency, but are strictly excluded from ranking and feature pipelines.

| Company Code | MainType | Row Count | Start Date | End Date | Status in Universe |
| :--- | :---: | :---: | :---: | :---: | :--- |
| AEL | N | 2,671 | 2012-03-27 | 2025-12-31 | **In Universe** |
| CFIN | N | 4,782 | 2002-01-02 | 2025-12-31 | **In Universe** |
| CFIN | R | 1 | 2004-11-02 | 2004-11-02 | Preserved (Non-Universe) |
| COMB | N | 4,272 | 2002-01-03 | 2025-12-31 | **In Universe** |
| COMB | P | 23 | 2002-03-01 | 2007-10-05 | Preserved (Non-Universe) |
| COMB | R | 22 | 2003-10-14 | 2024-08-06 | Preserved (Non-Universe) |
| COMB | X | 790 | 2002-01-02 | 2025-12-08 | Preserved (Non-Universe) |
| DFCC | B | 1 | 2006-07-17 | 2006-07-17 | Preserved (Non-Universe) |
| DFCC | N | 4,976 | 2002-01-02 | 2025-12-31 | **In Universe** |
| DFCC | R | 8 | 2007-05-23 | 2019-04-17 | Preserved (Non-Universe) |
| DIAL | N | 4,267 | 2005-07-28 | 2025-12-31 | **In Universe** |
| DIAL | R | 2 | 2007-06-08 | 2007-06-12 | Preserved (Non-Universe) |
| HAYL | N | 4,819 | 2002-01-02 | 2025-12-31 | **In Universe** |
| HAYL | R | 1 | 2002-10-08 | 2002-10-08 | Preserved (Non-Universe) |
| HHL | N | 4,481 | 2003-10-15 | 2025-12-31 | **In Universe** |
| HHL | R | 6 | 2015-04-27 | 2015-05-06 | Preserved (Non-Universe) |
| HNB | D | 73 | 2002-01-21 | 2005-06-16 | Preserved (Non-Universe) |
| HNB | N | 2,796 | 2002-01-02 | 2025-12-22 | **In Universe** |
| HNB | R | 11 | 2011-07-15 | 2017-07-25 | Preserved (Non-Universe) |
| HNB | X | 2,231 | 2002-01-08 | 2025-12-31 | Preserved (Non-Universe) |
| JKH | N | 4,958 | 2002-01-02 | 2025-12-31 | **In Universe** |
| JKH | R | 16 | 2003-11-28 | 2024-10-03 | Preserved (Non-Universe) |
| JKH | W | 132 | 2013-11-14 | 2016-09-22 | Preserved (Non-Universe) |
| LFIN | B | 1 | 2006-11-09 | 2006-11-09 | Preserved (Non-Universe) |
| LFIN | N | 4,231 | 2002-01-03 | 2025-12-31 | **In Universe** |
| LFIN | R | 5 | 2007-12-17 | 2008-08-15 | Preserved (Non-Universe) |
| LFIN | W | 303 | 2004-04-28 | 2006-08-15 | Preserved (Non-Universe) |
| LIOC | N | 4,399 | 2004-12-22 | 2025-12-31 | **In Universe** |
| LOLC | B | 1 | 2003-07-21 | 2003-07-21 | Preserved (Non-Universe) |
| LOLC | N | 4,697 | 2002-01-02 | 2025-12-31 | **In Universe** |
| MELS | N | 1,501 | 2016-12-30 | 2025-12-31 | **In Universe** |
| NDB | B | 1 | 2006-05-18 | 2006-05-18 | Preserved (Non-Universe) |
| NDB | N | 5,025 | 2002-01-02 | 2025-12-31 | **In Universe** |
| NDB | R | 6 | 2018-09-26 | 2018-10-03 | Preserved (Non-Universe) |
| NTB | N | 4,827 | 2002-01-02 | 2025-12-31 | **In Universe** |
| NTB | R | 9 | 2008-01-30 | 2018-01-29 | Preserved (Non-Universe) |
| NTB | W | 242 | 2008-03-05 | 2011-02-21 | Preserved (Non-Universe) |
| NTB | X | 8 | 2018-10-01 | 2019-11-29 | Preserved (Non-Universe) |
| PLC | N | 2,754 | 2011-11-24 | 2025-12-31 | **In Universe** |
| SAMP | D | 8 | 2002-04-22 | 2005-06-23 | Preserved (Non-Universe) |
| SAMP | N | 4,963 | 2002-01-02 | 2025-12-31 | **In Universe** |
| SAMP | R | 17 | 2005-01-05 | 2019-06-06 | Preserved (Non-Universe) |
| SEYB | D | 26 | 2002-01-09 | 2004-12-13 | Preserved (Non-Universe) |
| SEYB | N | 1,166 | 2002-01-02 | 2025-12-24 | **In Universe** |
| SEYB | R | 17 | 2006-05-15 | 2019-11-28 | Preserved (Non-Universe) |
| SEYB | X | 3,901 | 2003-09-22 | 2025-12-31 | Preserved (Non-Universe) |
| SINS | N | 4,466 | 2002-01-02 | 2025-12-31 | **In Universe** |
| SUN | N | 3,037 | 2002-05-03 | 2025-12-31 | **In Universe** |

---

## 2. Comparison Against Previous Run (Old Run vs. New Run)

| Diagnostic Metric | Previous Run (Mixed Classes) | Revised Run (`series_id`) | Impact / Diagnosis |
| :--- | :---: | :---: | :--- |
| **Split Adjustments Logged** | **31** | **12** (8 Voting `N`) | **19 spurious splits eliminated** (artifacts of voting/non-voting price differences) |
| **Suspect Moves Flagged (>50%)** | **2,152** | **58** (18 in Voting `N`) | **98.8% reduction in spurious anomalies** |
| **Close Outside [Low, High]** | 550 unhandled | 550 clipped & flagged | `High/Low` bounded to include `Close`; `close_out_of_range` flag added |

### Surviving Confirmed Stock Splits (Voting `N` Series):
- **HHL.N** (2025-05-07): Forward Split factor `5.0` (Observed: `4.9694`, Raw Close: `24.5`, Prior rows adjusted: `4,321`)
- **SUN.N** (2025-02-19): Forward Split factor `4.0` (Observed: `3.9911`, Raw Close: `22.5`, Prior rows adjusted: `2,829`)
- **JKH.N** (2024-11-06): Forward Split factor `10.0` (Observed: `10.0236`, Raw Close: `21.2`, Prior rows adjusted: `4,682`)
- **SINS.N** (2021-03-10): Forward Split factor `3.0` (Observed: `2.9226`, Raw Close: `16.8`, Prior rows adjusted: `3,916`)
- **LFIN.N** (2011-03-07): Forward Split factor `2.0` (Observed: `2.0068`, Raw Close: `18.3875`, Prior rows adjusted: `1,422`)
- **LOLC.N** (2010-11-22): Forward Split factor `10.0` (Observed: `10.0309`, Raw Close: `129.6`, Prior rows adjusted: `1,778`)
- **DFCC.N** (2010-11-02): Forward Split factor `2.0` (Observed: `2.0165`, Raw Close: `229.8`, Prior rows adjusted: `1,982`)
- **COMB.N** (2005-05-31): Forward Split factor `2.0` (Observed: `1.9643`, Raw Close: `66.5`, Prior rows adjusted: `589`)

---

## 3. Revised Market Gap & Closure Bridging Rules

### Bridging Rule Definition:
- A calendar gap is classified as a **market-wide closure** if:
  1. `gap_days <= 60` calendar days (`bridge_max_gap_days`).
  2. The gap is synchronously shared by at least **80%** (`shared_gap_fraction`) of universe constituents active around the gap dates.
- **Bridged Gaps**: Do not increment `segment_id`. Price continuity is maintained because trading orderly resumed across the whole exchange.
- **Hard Breaks**: Any gap `> 60` days (`hard_break_gap_days`) or any idiosyncratic gap `> 30` days (`gap_days_threshold`) triggers a new `segment_id`.

### Market-Wide Closures Bridged:
- **2020 COVID-19 Nationwide Halt**: `2020-03-20` to `2020-05-11` (**52 calendar days**).
  - Shared by **17 of 20 active voting series (85.0%)**.
  - **Decision: BRIDGED**. Forward windows and rolling windows remain continuous.

### Hard Breaks Preserved:
- **2021 Data Void**: `2021-03-31` to `2022-01-03` (**278 calendar days**). Exceeds 60-day limit.
- **2022–2024 Data Void**: `2022-03-31` to `2024-01-02` (**642 calendar days**). Exceeds 60-day limit. Macro crisis regime break.
- **Idiosyncratic Ticker Suspensions**: Isolated single-company suspensions (e.g. SEYB.N, SUN.N, LFIN.N long gaps) with shared fraction < 80%.

---

## 4. Revised Forward Window Usability Matrix (Voting Universe)

| Year | Total Rows | Short (21d) Valid | Short % | Mid (63d) Valid | Mid % | Long (252d) Valid | Long % | Usability Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 2002 | 2,422 | 2,406 | 99.3% | 2,406 | 99.3% | 2,400 | 99.1% | Fully Usable for Short & Mid |
| 2003 | 2,626 | 2,615 | 99.6% | 2,615 | 99.6% | 2,459 | 93.6% | Fully Usable for Short & Mid |
| 2004 | 2,853 | 2,818 | 98.8% | 2,730 | 95.7% | 2,643 | 92.6% | Fully Usable for Short & Mid |
| 2005 | 3,216 | 3,140 | 97.6% | 3,048 | 94.8% | 2,919 | 90.8% | Fully Usable for Short & Mid |
| 2006 | 3,325 | 3,248 | 97.7% | 3,201 | 96.3% | 3,173 | 95.4% | Fully Usable for Short & Mid |
| 2007 | 3,209 | 3,182 | 99.2% | 3,177 | 99.0% | 3,052 | 95.1% | Fully Usable for Short & Mid |
| 2008 | 3,172 | 3,111 | 98.1% | 3,025 | 95.4% | 2,907 | 91.6% | Fully Usable for Short & Mid |
| 2009 | 3,251 | 3,215 | 98.9% | 3,215 | 98.9% | 3,196 | 98.3% | Fully Usable for Short & Mid |
| 2010 | 3,777 | 3,777 | 100.0% | 3,737 | 98.9% | 3,636 | 96.3% | Fully Usable for Short & Mid |
| 2011 | 3,767 | 3,746 | 99.4% | 3,744 | 99.4% | 3,744 | 99.4% | Fully Usable for Short & Mid |
| 2012 | 4,159 | 4,159 | 100.0% | 4,159 | 100.0% | 4,159 | 100.0% | Fully Usable for Short & Mid |
| 2013 | 4,291 | 4,291 | 100.0% | 4,291 | 100.0% | 4,291 | 100.0% | Fully Usable for Short & Mid |
| 2014 | 4,301 | 4,301 | 100.0% | 4,301 | 100.0% | 4,301 | 100.0% | Fully Usable for Short & Mid |
| 2015 | 4,202 | 4,202 | 100.0% | 4,202 | 100.0% | 4,202 | 100.0% | Fully Usable for Short & Mid |
| 2016 | 4,206 | 4,206 | 100.0% | 4,206 | 100.0% | 4,206 | 100.0% | Fully Usable for Short & Mid |
| 2017 | 4,507 | 4,507 | 100.0% | 4,507 | 100.0% | 4,507 | 100.0% | Fully Usable for Short & Mid |
| 2018 | 4,433 | 4,433 | 100.0% | 4,433 | 100.0% | 4,397 | 99.2% | Fully Usable for Short & Mid |
| 2019 | 4,527 | 4,527 | 100.0% | 4,487 | 99.1% | 4,058 | 89.6% | Fully Usable for Short, Mid, & Long (89.6%) |
| 2020 | 4,057 | 4,015 | 99.0% | 3,822 | 94.2% | 211 | 5.2% | Fully Usable for Short & Mid (93%); Long partially limited |
| 2021 | 1,111 | 691 | 62.2% | 0 | 0.0% | 0 | 0.0% | **UNUSABLE for Mid/Long**; Short partially viable (61-62%) |
| 2022 | 1,093 | 673 | 61.6% | 0 | 0.0% | 0 | 0.0% | **UNUSABLE for Mid/Long**; Short partially viable (61-62%) |
| 2024 | 4,680 | 4,680 | 100.0% | 4,680 | 100.0% | 4,278 | 91.4% | Fully Usable for Short & Mid |
| 2025 | 4,638 | 4,218 | 90.9% | 3,378 | 72.8% | 0 | 0.0% | **UNUSABLE for Long** (terminal dataset boundary) |

---
*Generated automatically by Stock Pulse pipeline.*

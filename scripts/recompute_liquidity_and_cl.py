"""scripts/recompute_liquidity_and_cl.py

Computes market calendar, trade_day_share_63m, horizon spans, new liquidity threshold (20th percentile),
and analyzes Consistent Leaders before vs after filtering on Dev OOS.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

base_dir = Path(__file__).resolve().parents[1]

# 1. Load data
features_path = base_dir / "data/processed/features_targets.parquet"
mask_path = base_dir / "data/processed/rankable_mask.parquet"

df_feat = pd.read_parquet(features_path)
mask = pd.read_parquet(mask_path)["rankable"].values
df_feat["rankable"] = mask

u = df_feat[df_feat["in_ranking_universe"]].copy()
u["Date_dt"] = pd.to_datetime(u["Date"])

# 2. Build market calendar: dates with >= 10 ranking series
date_counts = u.groupby("Date")["series_id"].nunique()
mkt_dates_str = sorted(date_counts[date_counts >= 10].index)
mkt_dates_dt = pd.to_datetime(mkt_dates_str)
mkt_df = pd.DataFrame({"Date": mkt_dates_dt})

# 3. Market median spans for 21, 63, 252 market calendar days
mkt_spans = {}
for n in [21, 63, 252]:
    span = (mkt_df["Date"].shift(-n + 1) - mkt_df["Date"]).dt.days.dropna()
    mkt_spans[n] = float(span.median())

# 4. Stock spans & reports/08_horizon_span_by_stock.csv
span_rows = []
for sid, s_df in u.sort_values("Date").groupby("series_id"):
    s_df = s_df.copy()
    rec = {"series_id": sid}
    for n in [21, 63, 252]:
        span = (s_df["Date_dt"].shift(-n + 1) - s_df["Date_dt"]).dt.days.dropna()
        med_span = float(span.median()) if not span.empty else np.nan
        ratio = med_span / mkt_spans[n] if pd.notnull(med_span) else np.nan
        rec[f"median_span_{n}"] = med_span
        rec[f"ratio_{n}"] = ratio
    rec["stretched_horizon"] = bool(rec["ratio_21"] > 1.5)
    span_rows.append(rec)

df_spans = pd.DataFrame(span_rows)
csv_spans_path = base_dir / "reports/08_horizon_span_by_stock.csv"
df_spans.to_csv(csv_spans_path, index=False)
print(f"Saved {csv_spans_path}")
print("\n=== HORIZON SPAN BY STOCK ===")
print(df_spans.to_string(index=False))

stretched_stocks = df_spans[df_spans["stretched_horizon"]]["series_id"].tolist()
print(f"\nStretched horizon series (ratio > 1.5): {stretched_stocks}")

# 5. Compute trade_day_share_63m per series and date
tds_records = []
for sid, s_df in u.groupby("series_id"):
    for seg, seg_df in s_df.groupby("segment_id"):
        seg_dates = set(seg_df["Date"])
        min_d = seg_df["Date"].min()
        max_d = seg_df["Date"].max()
        m_dates_in_seg = [d for d in mkt_dates_str if min_d <= d <= max_d]
        if not m_dates_in_seg:
            continue
        traded = pd.Series([1 if d in seg_dates else 0 for d in m_dates_in_seg], index=m_dates_in_seg)
        roll = traded.rolling(63).mean()
        for d in seg_df["Date"]:
            tds_records.append({"series_id": sid, "Date": d, "trade_day_share_63m": roll.get(d, np.nan)})

df_tds = pd.DataFrame(tds_records)
u = pd.merge(u, df_tds, on=["series_id", "Date"], how="left")

# 6. Compute 20th percentile liquidity threshold on dev rankable rows
dev_mask = (u["Date"] >= "2008-01-01") & (u["Date"] <= "2019-12-31") & u["rankable"]
dev_tds = u.loc[dev_mask, "trade_day_share_63m"].dropna()
p20 = float(dev_tds.quantile(0.20))
p20_round = round(p20, 4)
print(f"\nNew Liquidity Warning Threshold (20th percentile on Dev Rankable rows): {p20:.6f} -> {p20_round} ({p20_round*100:.2f}%)")

# 7. Update thresholds.json
thresh_path = base_dir / "thresholds.json"
with open(thresh_path, "r") as f:
    thresh = json.load(f)

thresh["liquidity_warning_cutoff"] = {
    "id": "liquidity_warning_cutoff",
    "value": p20_round,
    "unit": "trade_day_share_63m",
    "method": "20th percentile of trade_day_share_63m (share of last 63 market calendar days with traded observations within segment) pooled across dev rankable rows. Warning fires when stock is at or below this cutoff.",
    "data_used": "Dev period clean rankable rows (2008-2019)",
    "derivation_date": "2026-10-11",
    "status": "empirical"
}

thresh["thin_trading_flag"] = {
    "id": "thin_trading_flag",
    "flagged_series": stretched_stocks,
    "threshold_ratio": 1.5,
    "description": "Series whose median calendar days spanned by 21 consecutive trading rows exceeds 1.5x the market median. All 'N trading day' windows count the stock's own trading observations, which span longer calendar time for thinly traded stocks.",
    "derivation_date": "2026-10-11",
    "status": "empirical"
}

with open(thresh_path, "w") as f:
    json.dump(thresh, f, indent=2)
print("Updated thresholds.json with new liquidity warning and thin trading flag.")

# 8. Load score_history.parquet and compare Consistent Leaders
app_score_path = base_dir / "reports/archive_v5/app/score_history.parquet"
if not app_score_path.exists():
    app_score_path = base_dir / "data/app/score_history.parquet"

df_score = pd.read_parquet(app_score_path)
df_score = pd.merge(df_score, df_tds, on=["series_id", "Date"], how="left")
df_score["new_liquidity_warning"] = (df_score["trade_day_share_63m"] <= p20_round)

high_count = ((df_score["tier_short"] == "High").astype(int) +
              (df_score["tier_mid"] == "High").astype(int) +
              (df_score["tier_long"] == "High").astype(int))
has_low = ((df_score["tier_short"] == "Low") |
           (df_score["tier_mid"] == "Low") |
           (df_score["tier_long"] == "Low"))

df_score["cl_before"] = (high_count >= 2) & (~has_low) & (~df_score["liquidity_warning"])
df_score["cl_after"] = (high_count >= 2) & (~has_low) & (~df_score["new_liquidity_warning"])

dev_df = df_score[df_score["split_type"] == "dev"].copy()

# Add fwd_excess
for h in ["short", "mid", "long"]:
    dev_df[f"fwd_excess_{h}"] = df_feat.set_index(["Date", "series_id"]).loc[
        dev_df.set_index(["Date", "series_id"]).index, f"fwd_excess_{h}"
    ].values

print("\n=== CONSISTENT LEADERS BEFORE VS AFTER (DEV OOS) ===")
cl_b = dev_df.groupby("series_id")["cl_before"].agg(["sum", "count"])
cl_b["share_before_pct"] = (cl_b["sum"] / cl_b["count"]) * 100.0

cl_a = dev_df.groupby("series_id")["cl_after"].agg(["sum", "count"])
cl_a["share_after_pct"] = (cl_a["sum"] / cl_a["count"]) * 100.0

comp = pd.DataFrame({
    "Days_Before": cl_b["sum"],
    "Share_Before_%": cl_b["share_before_pct"],
    "Days_After": cl_a["sum"],
    "Share_After_%": cl_a["share_after_pct"],
    "Change_Days": cl_a["sum"] - cl_b["sum"]
}).sort_values("Days_Before", ascending=False)
print(comp.to_string())

print("\n--- TARGET STOCKS TRACKING ---")
for ticker in ["SUN.N", "HNB.N", "MELS.N", "SEYB.X"]:
    if ticker in comp.index:
        r = comp.loc[ticker]
        print(f"{ticker}: Before = {int(r['Days_Before'])} days ({r['Share_Before_%']:.1f}%), After = {int(r['Days_After'])} days ({r['Share_After_%']:.1f}%), Change = {int(r['Change_Days']):+d} days")

print("\n--- CONSISTENT LEADERS HIT RATE COMPARISON (DEV OOS) ---")
for h in ["short", "mid", "long"]:
    hr_b_lead = float((dev_df[dev_df["cl_before"]][f"fwd_excess_{h}"] > 0).mean() * 100.0)
    hr_b_oth = float((dev_df[~dev_df["cl_before"]][f"fwd_excess_{h}"] > 0).mean() * 100.0)
    
    hr_a_lead = float((dev_df[dev_df["cl_after"]][f"fwd_excess_{h}"] > 0).mean() * 100.0)
    hr_a_oth = float((dev_df[~dev_df["cl_after"]][f"fwd_excess_{h}"] > 0).mean() * 100.0)
    
    print(f"Horizon {h.upper()}:")
    print(f"  Before Filter: Leaders {hr_b_lead:.1f}% vs Others {hr_b_oth:.1f}% (Spread = {hr_b_lead - hr_b_oth:+.1f}%)")
    print(f"  After Filter:  Leaders {hr_a_lead:.1f}% vs Others {hr_a_oth:.1f}% (Spread = {hr_a_lead - hr_a_oth:+.1f}%)")

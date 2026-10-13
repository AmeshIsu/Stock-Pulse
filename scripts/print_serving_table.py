"""scripts/print_serving_table.py

Prints institutional serving table for all 20 universe stocks as of data/serving/CURRENT.
Reads ONLY from data/serving/{as_of}/ (via data/serving/CURRENT pointer).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
serving_dir = base_dir / "data" / "serving"

def print_serving_table():
    current_ptr = serving_dir / "CURRENT"
    if not current_ptr.exists():
        print("ERROR: data/serving/CURRENT not found.")
        sys.exit(1)
        
    as_of = current_ptr.read_text(encoding="utf-8").strip()
    target_dir = serving_dir / as_of
    
    # Read active defaults to know chosen model per horizon
    active_def_path = serving_dir / "active_model_defaults.json"
    if active_def_path.exists():
        with open(active_def_path, "r", encoding="utf-8") as f:
            active_defaults = json.load(f)
        active_models = active_defaults.get("active_models", {})
    else:
        active_models = {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"}
        
    active_short = active_models.get("short", "ens_rank4")
    active_mid = active_models.get("mid", "momentum_126")
    active_long = active_models.get("long", "ens_rank4")

    # Load served scores parquet
    scores_path = target_dir / "scores_served.parquet"
    if not scores_path.exists():
        # Fallback: composite dynamically from active model parquets
        df_short = pd.read_parquet(target_dir / f"scores_{active_short}.parquet").set_index("series_id")
        df_mid = pd.read_parquet(target_dir / f"scores_{active_mid}.parquet").set_index("series_id")
        df_long = pd.read_parquet(target_dir / f"scores_{active_long}.parquet").set_index("series_id")
        
        # Load evidence for confidence labels
        with open(base_dir / "data/app/evidence.json", "r", encoding="utf-8") as ef:
            root_ev = json.load(ef)
        
        comp_rows = []
        for sid in df_short.index:
            sr = df_short.loc[sid]
            mr = df_mid.loc[sid]
            lr = df_long.loc[sid]
            r_dict = sr.to_dict()
            r_dict["score_mid"] = mr["score_mid"]
            r_dict["tier_mid"] = mr["tier_mid"]
            r_dict["confidence_short"] = root_ev.get("short", {}).get("confidence_label", "Very low confidence")
            r_dict["confidence_mid"] = root_ev.get("mid", {}).get("confidence_label", "Low confidence")
            r_dict["confidence_long"] = root_ev.get("long", {}).get("confidence_label", "Very low confidence")
            r_dict["overall_upside_score"] = round(sr["score_short"] * 0.25 + mr["score_mid"] * 0.50 + lr["score_long"] * 0.25, 2)
            comp_rows.append(r_dict)
        df = pd.DataFrame(comp_rows)
    else:
        df = pd.read_parquet(scores_path)
    
    # Load thresholds for cutoff display
    with open(base_dir / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    liq_cutoff = thresholds["liquidity_warning_cutoff"]["value"]
    
    print(f"\n==========================================================================================================")
    print(f" STOCK PULSE SERVING TABLE — AS OF: {as_of} [ADMIN ONLY: Active Models: Short={active_short} | Mid={active_mid} | Long={active_long}]")
    print(f"==========================================================================================================")
    print(f"Reading exclusively from: {target_dir}\n")
    
    rows = []
    for _, r in df.sort_values("series_id").iterrows():
        sid = r["series_id"]
        status = r.get("status", "ranked")
        last_traded = r.get("last_traded_date", as_of)
        
        s_short = f"{r['score_short']:.1f} ({r['tier_short']})" if pd.notnull(r.get("score_short")) else "N/A"
        s_mid = f"{r['score_mid']:.1f} ({r['tier_mid']})" if pd.notnull(r.get("score_mid")) else "N/A"
        s_long = f"{r['score_long']:.1f} ({r['tier_long']})" if pd.notnull(r.get("score_long")) else "N/A"
        
        overall = f"{r['overall_upside_score']:.1f}" if pd.notnull(r.get("overall_upside_score")) else "N/A"
        cl = "YES" if r.get("consistent_leader") else "NO"
        
        # Risk & Vol
        vol_val = r.get("vol_63d_ann", r.get("vol_63d", 0.0) * 100.0)
        risk = f"{r.get('risk_category', 'N/A')} ({vol_val:.1f}%)" if pd.notnull(r.get("vol_63d")) else str(r.get("risk_category", "N/A"))
        
        # Liquidity
        tds = r.get("trade_day_share_63m", 1.0)
        warn_flag = "WARN" if r.get("liquidity_warning") else "OK"
        liq = f"{warn_flag} ({tds:.2f} <= {liq_cutoff})" if r.get("liquidity_warning") else f"{warn_flag} ({tds:.2f})"
        
        # Confidence
        c_short = r.get("confidence_short", "")
        c_mid = r.get("confidence_mid", "")
        c_long = r.get("confidence_long", "")
        conf = f"S:{c_short} | M:{c_mid} | L:{c_long}"
        
        # Drivers
        reasons_dict = r.get("reasons", {})
        pos_list = reasons_dict.get("top_positive", [])
        neg_list = reasons_dict.get("top_negative", [])
        
        if isinstance(pos_list, (list, np.ndarray)):
            pos_str = "; ".join(pos_list)
        else:
            pos_str = str(pos_list)
            
        if isinstance(neg_list, (list, np.ndarray)):
            neg_str = "; ".join(neg_list)
        else:
            neg_str = str(neg_list)
            
        drivers = f"[+] {pos_str} | [-] {neg_str}"
        
        rows.append({
            "Stock": sid,
            "Status": status,
            "Last Traded": last_traded,
            "Short (21d)": s_short,
            "Mid (63d)": s_mid,
            "Long (252d)": s_long,
            "Overall": overall,
            "Leader": cl,
            "Risk": risk,
            "Liquidity": liq,
            "Confidence": conf,
            "Key Drivers": drivers
        })
        
    df_out = pd.DataFrame(rows)
    # Print formatted table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)
    print(df_out.to_string(index=False))
    print(f"\nTotal series displayed: {len(df_out)} constituents.")

if __name__ == "__main__":
    print_serving_table()

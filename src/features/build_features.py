"""Feature engineering and target calculation engine for Stock Pulse.

Universe: Voting-share series only (MainType == 'N').
Strict Temporal Rules:
- Backward-looking features only (up to and including date t).
- Rolling windows and lagged calculations never cross hard segment boundaries.
- Targets look forward strictly within the same segment.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.config_loader import load_config

def compute_technical_features(df_u: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute momentum, risk, trend, and volume features per (series_id, segment_id)."""
    dfs = []
    
    # Process each segment independently to guarantee zero cross-boundary leakage
    for (sid, seg), grp in df_u.groupby(["series_id", "segment_id"], sort=False):
        g = grp.sort_values("Date").copy()
        n_rows = len(g)
        
        close = g["Close"].values
        high = g["High"].values
        low = g["Low"].values
        trade_vol = g["TradeVolume"].values
        share_vol = g["ShareVolume"].values
        turnover = g["Turnover"].values
        daily_ret = g["daily_return"].values
        log_ret = g["log_return"].values
        
        # 1. Momentum Features (log returns over trading rows)
        for w in [5, 10, 21, 63, 126, 252]:
            ret_col = f"log_ret_{w}"
            vals = np.full(n_rows, np.nan, dtype=np.float64)
            if n_rows > w:
                vals[w:] = np.log(close[w:] / close[:-w])
            g[ret_col] = vals
            
        # 12-1 Momentum (252-day return excluding latest 21 days)
        mom_12_1 = np.full(n_rows, np.nan, dtype=np.float64)
        if n_rows > 252:
            mom_12_1[252:] = np.log(close[231:-21] / close[:-252])
        g["mom_12_1"] = mom_12_1
        
        # 2. Risk Features
        s_log_ret = pd.Series(log_ret, index=g.index)
        g["vol_21d"] = s_log_ret.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
        g["vol_63d"] = s_log_ret.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
        
        # Downside Volatility (63d)
        neg_log_ret = np.minimum(log_ret, 0.0)
        s_neg = pd.Series(neg_log_ret, index=g.index)
        downside_var = (s_neg ** 2).rolling(63, min_periods=63).sum() / (63 - 1)
        g["downside_vol_63d"] = np.sqrt(downside_var * 252)
        
        # Parkinson Volatility (21d): sqrt( 252 / (21 * 4 * ln(2)) * sum( ln(H/L)^2 ) )
        hl_ratio = np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6))
        s_hl2 = pd.Series(hl_ratio ** 2, index=g.index)
        parkinson_sum = s_hl2.rolling(21, min_periods=21).sum()
        g["parkinson_vol_21d"] = np.sqrt((252.0 / (21.0 * 4.0 * np.log(2.0))) * parkinson_sum)
        
        # Normalized ATR14
        prev_close_arr = np.roll(close, 1)
        prev_close_arr[0] = np.nan
        tr1 = high - low
        tr2 = np.abs(high - prev_close_arr)
        tr3 = np.abs(low - prev_close_arr)
        tr = np.nanmax(np.column_stack([tr1, tr2, tr3]), axis=1)
        s_tr = pd.Series(tr, index=g.index)
        atr14 = s_tr.rolling(14, min_periods=14).mean()
        g["atr14_norm"] = atr14 / close
        
        # Drawdowns & Price Ranges
        s_close = pd.Series(close, index=g.index)
        s_high = pd.Series(high, index=g.index)
        s_low = pd.Series(low, index=g.index)
        
        peak_63 = s_close.rolling(63, min_periods=63).max()
        g["max_drawdown_63d"] = (s_close - peak_63) / peak_63
        
        peak_252 = s_close.rolling(252, min_periods=252).max()
        g["max_drawdown_252d"] = (s_close - peak_252) / peak_252
        
        max_h_126 = s_high.rolling(126, min_periods=126).max()
        min_l_126 = s_low.rolling(126, min_periods=126).min()
        g["dist_high_126d"] = (s_close - max_h_126) / max_h_126
        g["dist_low_126d"] = (s_close - min_l_126) / min_l_126
        
        max_h_252 = s_high.rolling(252, min_periods=252).max()
        min_l_252 = s_low.rolling(252, min_periods=252).min()
        g["dist_high_252d"] = (s_close - max_h_252) / max_h_252
        g["dist_low_252d"] = (s_close - min_l_252) / min_l_252
        
        # 3. Trend Features
        sma20 = s_close.rolling(20, min_periods=20).mean()
        sma50 = s_close.rolling(50, min_periods=50).mean()
        sma200 = s_close.rolling(200, min_periods=200).mean()
        
        g["close_to_sma20"] = s_close / sma20 - 1.0
        g["close_to_sma50"] = s_close / sma50 - 1.0
        g["close_to_sma200"] = s_close / sma200 - 1.0
        g["sma50_to_sma200"] = sma50 / sma200 - 1.0
        
        # RSI 14
        delta = s_close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        g["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        
        # MACD Histogram normalized by Close
        ema12 = s_close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = s_close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        g["macd_hist_norm"] = (macd_line - signal_line) / s_close
        
        # Bollinger %B (20d, 2 std)
        std20 = s_close.rolling(20, min_periods=20).std(ddof=1)
        upper_band = sma20 + 2.0 * std20
        lower_band = sma20 - 2.0 * std20
        g["bollinger_pct_b_20"] = (s_close - lower_band) / (upper_band - lower_band + 1e-8)
        
        # 4. Liquidity & Volume Features
        s_trade_vol = pd.Series(trade_vol, index=g.index)
        s_share_vol = pd.Series(share_vol, index=g.index)
        s_turnover = pd.Series(turnover, index=g.index)
        
        g["log_trade_vol_21d"] = np.log(1.0 + s_trade_vol.rolling(21, min_periods=21).mean())
        mean_vol_21 = s_share_vol.rolling(21, min_periods=21).mean()
        mean_vol_126 = s_share_vol.rolling(126, min_periods=126).mean()
        g["vol_ratio_21_126"] = mean_vol_21 / (mean_vol_126 + 1e-6)
        g["log_turnover_21d"] = np.log(1.0 + s_turnover.rolling(21, min_periods=21).mean())
        
        # Amihud Illiquidity (63d): mean(|ret| / (Turnover / 1M))
        scaled_turnover = np.maximum(s_turnover / 1e6, 1e-6)
        illiq = np.abs(daily_ret) / scaled_turnover
        s_illiq = pd.Series(illiq, index=g.index)
        g["amihud_illiq_63d"] = s_illiq.rolling(63, min_periods=63).mean()
        
        # Share of zero-trade days in last 63 rows
        zero_trade = ((s_trade_vol == 0) | (s_share_vol == 0)).astype(float)
        g["zero_trade_share_63d"] = zero_trade.rolling(63, min_periods=63).mean()
        
        # 5. Own-history z-scores (B1)
        # (value - trailing 126-row mean) / trailing 126-row std (min 63 rows, causal, within segment)
        for col_name in ["log_turnover_21d", "log_trade_vol_21d", "amihud_illiq_63d", "vol_21d", "vol_63d"]:
            s_val = g[col_name]
            m126 = s_val.rolling(126, min_periods=63).mean()
            std126 = s_val.rolling(126, min_periods=63).std(ddof=1)
            g[f"{col_name}_ownz"] = np.where(std126 > 1e-9, (s_val - m126) / std126, np.nan)
            
        dfs.append(g)
        
    return pd.concat(dfs, ignore_index=True)


def compute_market_relative_features(df_u: pd.DataFrame, min_constituents: int = 8) -> pd.DataFrame:
    """Construct equal-weighted market index and compute market-relative features."""
    df_u = df_u.sort_values(["series_id", "Date"]).reset_index(drop=True)
    
    # 1. Equal-weight index return on days with >= min_constituents
    date_grp = df_u.groupby("Date")["daily_return"].agg(["count", "mean"])
    date_grp.loc[date_grp["count"] < min_constituents, "mean"] = np.nan
    mkt_returns = date_grp[["mean"]].rename(columns={"mean": "mkt_ret"}).reset_index()
    
    # Merge market return
    df_u = df_u.merge(mkt_returns, on="Date", how="left")
    
    # Compute market rolling 63d regime features
    mkt_df = mkt_returns.sort_values("Date").copy()
    mkt_ret_s = pd.Series(mkt_df["mkt_ret"].values, index=mkt_df.index)
    mkt_df["index_ret_63d"] = mkt_ret_s.rolling(63, min_periods=63).sum()
    mkt_df["index_vol_63d"] = mkt_ret_s.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
    
    df_u = df_u.merge(mkt_df[["Date", "index_ret_63d", "index_vol_63d"]], on="Date", how="left")
    
    # 2. Relative returns & Rolling 252d Beta per (series_id, segment_id)
    dfs = []
    for (sid, seg), grp in df_u.groupby(["series_id", "segment_id"], sort=False):
        g = grp.sort_values("Date").copy()
        
        s_mkt = g["mkt_ret"]
        s_ret = g["daily_return"]
        
        # Relative returns: stock log return minus sum of index returns
        mkt_sum_21 = s_mkt.rolling(21, min_periods=21).sum()
        mkt_sum_63 = s_mkt.rolling(63, min_periods=63).sum()
        
        g["rel_ret_21d"] = g["log_ret_21"] - mkt_sum_21
        g["rel_ret_63d"] = g["log_ret_63"] - mkt_sum_63
        
        # Rolling Beta (252d, min_periods=126)
        x = s_ret
        y = s_mkt
        sum_x = x.rolling(252, min_periods=126).sum()
        sum_y = y.rolling(252, min_periods=126).sum()
        sum_xy = (x * y).rolling(252, min_periods=126).sum()
        sum_yy = (y * y).rolling(252, min_periods=126).sum()
        n = x.rolling(252, min_periods=126).count()
        
        cov_xy = (sum_xy - sum_x * sum_y / n) / (n - 1)
        var_y = (sum_yy - sum_y * sum_y / n) / (n - 1)
        g["beta_252d"] = np.where((var_y > 1e-8) & (n >= 126), cov_xy / var_y, np.nan)
        
        dfs.append(g)
        
    df_res = pd.concat(dfs, ignore_index=True)
    
    # 3. Vectorized cross-sectional percentile ranks per Date
    date_cnt_mom = df_res.groupby("Date")["log_ret_63"].transform("count")
    df_res["cs_rank_mom_63d"] = np.where(
        date_cnt_mom >= min_constituents,
        df_res.groupby("Date")["log_ret_63"].rank(pct=True),
        np.nan
    )
    
    date_cnt_vol = df_res.groupby("Date")["vol_63d"].transform("count")
    df_res["cs_rank_vol_63d"] = np.where(
        date_cnt_vol >= min_constituents,
        df_res.groupby("Date")["vol_63d"].rank(pct=True),
        np.nan
    )
    
    date_cnt_tvol = df_res.groupby("Date")["log_trade_vol_21d"].transform("count")
    df_res["cs_rank_vol_21d"] = np.where(
        date_cnt_tvol >= min_constituents,
        df_res.groupby("Date")["log_trade_vol_21d"].rank(pct=True),
        np.nan
    )
    
    return df_res.drop(columns=["mkt_ret"])


def compute_targets(df_u: pd.DataFrame, horizons: dict[str, int], min_constituents: int = 8) -> pd.DataFrame:
    """Compute forward returns, cross-sectional excess returns, rank percentiles, and binary labels."""
    df_u = df_u.sort_values(["series_id", "Date"]).reset_index(drop=True)
    
    # Calculate raw forward returns per (series_id, segment_id)
    dfs = []
    for (sid, seg), grp in df_u.groupby(["series_id", "segment_id"], sort=False):
        g = grp.sort_values("Date").copy()
        close = g["Close"].values
        n = len(close)
        
        for name, h in horizons.items():
            fwd_col = f"fwd_ret_{name}"
            vals = np.full(n, np.nan, dtype=np.float64)
            if n > h:
                vals[:-h] = (close[h:] / close[:-h]) - 1.0
            g[fwd_col] = vals
            
        dfs.append(g)
        
    df_res = pd.concat(dfs, ignore_index=True)
    
    # Vectorized cross-sectional excess returns, rank percentiles, and outperform flags per Date
    for name in horizons.keys():
        fwd_col = f"fwd_ret_{name}"
        excess_col = f"fwd_excess_{name}"
        rank_col = f"fwd_rankpct_{name}"
        outperf_col = f"outperform_{name}"
        
        date_counts = df_res.groupby("Date")[fwd_col].transform("count")
        date_medians = df_res.groupby("Date")[fwd_col].transform("median")
        
        df_res[excess_col] = np.where(
            date_counts >= min_constituents,
            df_res[fwd_col] - date_medians,
            np.nan
        )
        avg_rank = df_res.groupby("Date")[fwd_col].rank(method="average")
        rank_pct = (avg_rank - 0.5) / date_counts
        df_res[rank_col] = np.where(date_counts >= min_constituents, rank_pct, np.nan)
        
        # Outperform: 1 if > 0.5, 0 if < 0.5, NaN if exactly 0.5 (dropped from classifier training)
        outperform = np.where(rank_pct > 0.5, 1.0, np.where(rank_pct < 0.5, 0.0, np.nan))
        df_res[outperf_col] = np.where(date_counts >= min_constituents, outperform, np.nan)
        
    return df_res


def build_features_and_targets(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete feature engineering and target calculation pipeline."""
    if config is None:
        config = load_config()
        
    base_dir = Path(__file__).resolve().parents[2]
    paths = config["paths"]
    horizons = config["horizons"]
    min_constituents = config.get("features", {}).get("min_universe_constituents", 8)
    
    clean_path = base_dir / paths["clean_prices"]
    out_features_path = base_dir / paths["features_targets"]
    out_avail_path = base_dir / paths["feature_availability_csv"]
    out_latest_path = base_dir / paths["latest_features"]
    dict_md_path = base_dir / paths["feature_dictionary_md"]
    
    print(f"Loading cleaned prices from {clean_path}...")
    df = pd.read_parquet(clean_path)
    
    # Restrict to ranking universe
    df_u = df[df["in_ranking_universe"]].copy()
    print(f"Filtered ranking universe: {len(df_u):,} rows across {df_u['series_id'].nunique()} series.")
    
    print("Computing technical features (momentum, risk, trend, liquidity)...")
    df_feats = compute_technical_features(df_u, config)
    
    print("Computing market index and market-relative features...")
    df_all_feats = compute_market_relative_features(df_feats, min_constituents=min_constituents)
    
    print("Computing forward-looking targets...")
    df_final = compute_targets(df_all_feats, horizons=horizons, min_constituents=min_constituents)
    
    # Check minimum valid feature rows threshold (750)
    min_feature_rows = config.get("min_valid_feature_rows_for_scoring", 750)
    core_features = config["feature_sets"]["core"]
    series_valid = df_final.groupby("series_id")[core_features].apply(lambda g: g.dropna().shape[0])
    under_threshold = series_valid[series_valid < min_feature_rows].index.tolist()
    if under_threshold:
        print(f"WARNING: The following series have < {min_feature_rows} valid feature rows and are excluded from ranking: {under_threshold}")
        df_final.loc[df_final["series_id"].isin(under_threshold), "in_ranking_universe"] = False
    else:
        print(f"All {df_final['series_id'].nunique()} ranking series have >= {min_feature_rows} valid feature rows.")
    
    # Sort deterministically
    df_final = df_final.sort_values(["series_id", "Date"]).reset_index(drop=True)
    
    # Save features and targets
    df_final.to_parquet(out_features_path, engine="pyarrow", index=False)
    print(f"Saved features and targets to {out_features_path} ({len(df_final):,} rows, {len(df_final.columns)} columns).")
    
    # Save latest features for live inference
    latest_rows = df_final.sort_values("Date").groupby("series_id", as_index=False).last()
    latest_rows.to_parquet(out_latest_path, engine="pyarrow", index=False)
    print(f"Saved latest-date feature snapshot for live scoring to {out_latest_path} ({len(latest_rows)} series).")
    
    # Feature availability matrix by year
    feature_cols = [
        col for col in df_final.columns
        if col not in [
            "Date", "CompanyCode", "MainType", "SubType", "ShortName", "CompanyName_Official",
            "Era", "Year", "Month", "Quarter", "DayOfWeek", "series_id", "segment_id",
            "in_ranking_universe", "close_out_of_range", "suspect_move", "gap_days",
            "Close", "High", "Low", "Close_raw", "Open_raw", "High_raw", "Low_raw",
            "TradeVolume", "ShareVolume", "Turnover", "daily_return", "log_return"
        ] and not col.startswith("fwd_") and not col.startswith("outperform_")
    ]
    
    df_final["Year"] = df_final["Date"].dt.year
    avail_df = df_final.groupby("Year")[feature_cols].apply(lambda g: g.notna().mean()).round(4)
    avail_df.to_csv(out_avail_path)
    print(f"Saved feature availability table to {out_avail_path}.")
    
    # Write feature dictionary
    write_feature_dictionary(feature_cols, horizons, dict_md_path)
    
    return df_final, avail_df, latest_rows


def write_feature_dictionary(feature_cols: list[str], horizons: dict, dict_path: Path):
    """Write comprehensive feature dictionary markdown document."""
    descriptions = {
        "log_ret_5": ("Momentum", "ln(Close_t / Close_{t-5})", "Short-term 1-week momentum reflecting immediate price velocity."),
        "log_ret_10": ("Momentum", "ln(Close_t / Close_{t-10})", "2-week price momentum capturing bi-weekly trend momentum."),
        "log_ret_21": ("Momentum", "ln(Close_t / Close_{t-21})", "Monthly 1-month momentum; institutional tactical horizon benchmark."),
        "log_ret_63": ("Momentum", "ln(Close_t / Close_{t-63})", "Quarterly 3-month momentum; aligns with corporate earnings cycle."),
        "log_ret_126": ("Momentum", "ln(Close_t / Close_{t-126})", "Semi-annual 6-month intermediate momentum."),
        "log_ret_252": ("Momentum", "ln(Close_t / Close_{t-252})", "Annual 1-year total price momentum; classic long-term trend factor."),
        "mom_12_1": ("Momentum", "ln(Close_{t-21} / Close_{t-252})", "Classic Jegadeesh-Titman 12-1 momentum excluding short-term reversal month."),
        "vol_21d": ("Risk", "std(log_ret, 21) * sqrt(252)", "Annualized 1-month realized volatility measuring tactical risk."),
        "vol_63d": ("Risk", "std(log_ret, 63) * sqrt(252)", "Annualized 3-month quarterly realized volatility measuring medium-term risk."),
        "downside_vol_63d": ("Risk", "sqrt(252 * sum(min(log_ret, 0)^2) / 62)", "Downside semi-volatility penalizing negative tail variance only."),
        "parkinson_vol_21d": ("Risk", "sqrt(252 / (21 * 4 * ln 2) * sum(ln(High/Low)^2))", "Extreme-value volatility estimator utilizing intraday High and Low ranges."),
        "atr14_norm": ("Risk", "ATR_14 / Close_t", "14-day Average True Range normalized by price for cross-sectional volatility comparability."),
        "max_drawdown_63d": ("Risk", "(Close_t - max(Close_{t-63..t})) / max(Close_{t-63..t})", "Peak-to-trough decline over preceding quarter; assesses drawdown severity."),
        "max_drawdown_252d": ("Risk", "(Close_t - max(Close_{t-252..t})) / max(Close_{t-252..t})", "Maximum peak-to-trough drawdown over 1-year window."),
        "dist_high_126d": ("Risk/Range", "(Close_t - max(High_{t-126..t})) / max(High_{t-126..t})", "Proximity to 6-month price ceiling."),
        "dist_low_126d": ("Risk/Range", "(Close_t - min(Low_{t-126..t})) / min(Low_{t-126..t})", "Proximity to 6-month price floor."),
        "dist_high_252d": ("Risk/Range", "(Close_t - max(High_{t-252..t})) / max(High_{t-252..t})", "Proximity to 52-week annual high; George-Hwang anchor indicator."),
        "dist_low_252d": ("Risk/Range", "(Close_t - min(Low_{t-252..t})) / min(Low_{t-252..t})", "Proximity to 52-week annual low; bounce and distress indicator."),
        "close_to_sma20": ("Trend", "Close_t / SMA_20(Close) - 1", "Percentage divergence from 20-day tactical moving average."),
        "close_to_sma50": ("Trend", "Close_t / SMA_50(Close) - 1", "Percentage divergence from 50-day intermediate moving average."),
        "close_to_sma200": ("Trend", "Close_t / SMA_200(Close) - 1", "Percentage divergence from 200-day institutional regime trend line."),
        "sma50_to_sma200": ("Trend", "SMA_50 / SMA_200 - 1", "Golden Cross / Death Cross moving average divergence spread."),
        "rsi_14": ("Trend/Oscillator", "100 - (100 / (1 + avg_gain / avg_loss))", "14-day Relative Strength Index measuring overbought/oversold momentum."),
        "macd_hist_norm": ("Trend", "(MACD - Signal) / Close_t", "Moving Average Convergence Divergence normalized oscillator histogram."),
        "bollinger_pct_b_20": ("Trend/Volatility", "(Close - LowerBand) / (UpperBand - LowerBand)", "Bollinger Band %B position within 20-day 2-sigma envelope."),
        "log_trade_vol_21d": ("Liquidity", "ln(1 + SMA_21(TradeVolume))", "Log-transformed 1-month average daily trade transaction count."),
        "vol_ratio_21_126": ("Liquidity", "SMA_21(ShareVolume) / SMA_126(ShareVolume)", "Volume surge ratio identifying institutional accumulation or distribution."),
        "log_turnover_21d": ("Liquidity", "ln(1 + SMA_21(Turnover))", "Log-transformed 1-month average daily rupee trading turnover."),
        "amihud_illiq_63d": ("Liquidity", "mean( |daily_ret| / (Turnover / 1M) )", "Amihud price impact ratio measuring illiquidity cost per rupee traded."),
        "zero_trade_share_63d": ("Liquidity", "mean( I(TradeVolume == 0 or ShareVolume == 0) )", "Fraction of zero-activity trading sessions in the trailing quarter."),
        "rel_ret_21d": ("Market-Relative", "log_ret_21 - mkt_cum_ret_21", "1-month alpha / excess return relative to Colombo equal-weighted index."),
        "rel_ret_63d": ("Market-Relative", "log_ret_63 - mkt_cum_ret_63", "3-month alpha / excess return relative to Colombo equal-weighted index."),
        "beta_252d": ("Market-Relative", "Cov(ret_i, ret_mkt) / Var(ret_mkt)", "252-day trailing systematic market beta to Colombo S&P SL20 index."),
        "index_ret_63d": ("Macro Regime", "sum(mkt_ret, 63)", "Trailing 63-day Colombo market index return representing market regime."),
        "index_vol_63d": ("Macro Regime", "std(mkt_ret, 63) * sqrt(252)", "Trailing 63-day Colombo market volatility representing market risk regime."),
        "cs_rank_mom_63d": ("Cross-Sectional", "rank(log_ret_63, pct=True)", "Cross-sectional percentile ranking of 3-month momentum across universe on day t."),
        "cs_rank_vol_63d": ("Cross-Sectional", "rank(vol_63d, pct=True)", "Cross-sectional percentile ranking of volatility across universe on day t."),
        "cs_rank_vol_21d": ("Cross-Sectional", "rank(log_trade_vol_21d, pct=True)", "Cross-sectional percentile ranking of trading activity across universe on day t.")
    }
    
    md_content = """# Stock Pulse: Feature Engineering & Target Data Dictionary

**Document**: `reports/02_feature_dictionary.md`  
**Scope**: In-Universe Voting Equity Series (`MainType == 'N'`)  
**Strict Temporal Invariant**: All features use data strictly up to and including trading date $t$. Windows never bridge hard segment boundaries.

---

## 1. Feature Specifications

| Feature Name | Category | Plain-English Formula | Financial / Empirical Rationale |
| :--- | :--- | :--- | :--- |
"""
    for col in feature_cols:
        cat, form, rat = descriptions.get(col, ("Engineered", col, "Technical predictor."))
        md_content += f"| `{col}` | {cat} | `{form}` | {rat} |\n"
        
    md_content += """
---

## 2. Target Specifications

All targets look forward strictly within the same segment. Offsets: $h \in \\{21, 63, 252\\}$ trading days.

| Target Variable | Horizon | Formula | Purpose |
| :--- | :---: | :--- | :--- |
| `fwd_ret_short` | 21 days | $Close[t+21] / Close[t] - 1$ | 1-month forward raw return |
| `fwd_excess_short` | 21 days | $fwd\\_ret - \\text{median}_{date}(fwd\\_ret)$ | 1-month forward return relative to cross-sectional median |
| `fwd_rankpct_short` | 21 days | $\\text{rank}_{date}(fwd\\_ret, \\text{pct}=True)$ | 1-month cross-sectional ranking percentile $[0, 1]$ |
| `outperform_short` | 21 days | $\\mathbb{I}(fwd\\_excess > 0)$ | Binary upside outperformance label |
| `fwd_ret_mid` | 63 days | $Close[t+63] / Close[t] - 1$ | 3-month forward raw return |
| `fwd_excess_mid` | 63 days | $fwd\\_ret - \\text{median}_{date}(fwd\\_ret)$ | 3-month forward return relative to cross-sectional median |
| `fwd_rankpct_mid` | 63 days | $\\text{rank}_{date}(fwd\\_ret, \\text{pct}=True)$ | 3-month cross-sectional ranking percentile $[0, 1]$ |
| `outperform_mid` | 63 days | $\\mathbb{I}(fwd\\_excess > 0)$ | Binary upside outperformance label |
| `fwd_ret_long` | 252 days | $Close[t+252] / Close[t] - 1$ | 1-year forward raw return |
| `fwd_excess_long` | 252 days | $fwd\\_ret - \\text{median}_{date}(fwd\\_ret)$ | 1-year forward return relative to cross-sectional median |
| `fwd_rankpct_long` | 252 days | $\\text{rank}_{date}(fwd\\_ret, \\text{pct}=True)$ | 1-year cross-sectional ranking percentile $[0, 1]$ |
| `outperform_long` | 252 days | $\\mathbb{I}(fwd\\_excess > 0)$ | Binary upside outperformance label |

---
*Generated by Stock Pulse Feature Engine.*
"""
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Feature dictionary written to: {dict_path}")

if __name__ == "__main__":
    build_features_and_targets()

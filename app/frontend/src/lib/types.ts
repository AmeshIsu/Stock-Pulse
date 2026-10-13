// Data types for Stock Pulse Frontend

export type Horizon = 'short' | 'mid' | 'long';
export type Tier = 'High' | 'Neutral' | 'Low';
export type RiskLevel = 'Normal' | 'Elevated';
export type ConfidenceLevel = 'High confidence' | 'Moderate confidence' | 'Low confidence' | 'Very low confidence';

export interface User {
  id: string;
  email: string;
  role: 'user' | 'admin';
  is_active: boolean;
  created_at: string;
  last_login_at?: string;
}

export interface PublicInfo {
  app_name: string;
  data_as_of: string;
  disclaimer_text: string;
}

export interface HorizonScore {
  score: number;
  tier: Tier;
  rank: number;
}

export interface StockRankingRow {
  rank: number;
  code: string;
  name: string;
  display_ticker?: string;
  overall_score: number;
  overall_tier: Tier;
  horizons: {
    short: HorizonScore;
    mid: HorizonScore;
    long: HorizonScore;
  };
  risk: RiskLevel;
  liquidity_flag: boolean;
  liquidity_reason?: string;
  last_price: number;
  daily_change_pct: number;
  low_52w?: number;
  high_52w?: number;
  is_stale?: boolean;
  last_traded_date?: string;
}

export interface RankingsResponse {
  as_of_date: string;
  weights: {
    short: number;
    mid: number;
    long: number;
  };
  confidence_labels: {
    short: ConfidenceLevel;
    mid: ConfidenceLevel;
    long: ConfidenceLevel;
  };
  confidence_explanations: {
    short: string;
    mid: string;
    long: string;
  };
  rankings: StockRankingRow[];
}

export interface ConsistentLeaderItem {
  code: string;
  name: string;
  top_tier_horizons: string[];
  last_price: number;
  daily_change_pct: number;
}

export interface MoverItem {
  code: string;
  name: string;
  last_price: number;
  daily_change_pct: number;
  is_indicative: boolean;
  last_traded_date: string;
}

export interface MarketSummaryResponse {
  as_of_date: string;
  high_tier_count: number;
  risk_mix: {
    normal: number;
    elevated: number;
  };
  volatility_note: {
    elevated: boolean;
    headline: string;
    message: string;
    current_vol_pct: number;
    threshold_pct: number;
  };
  consistent_leaders: ConsistentLeaderItem[];
  top_movers: {
    gainers: MoverItem[];
    losers: MoverItem[];
  };
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface HorizonDriver {
  name: string;
  direction: 'supporting' | 'holding_back';
  description: string;
}

export interface HorizonOutlook {
  score: number;
  tier: Tier;
  confidence: ConfidenceLevel;
  confidence_explanation: string;
  drivers: {
    supporting: string[];
    holding_back: string[];
  };
  thin_trading_note?: string;
}

export interface StockDetailResponse {
  code: string;
  name: string;
  display_ticker: string;
  as_of_date: string;
  last_price: number;
  daily_change_pct: number;
  is_stale: boolean;
  last_traded_date: string;
  low_52w: number;
  high_52w: number;
  outlooks: {
    short: HorizonOutlook;
    mid: HorizonOutlook;
    long: HorizonOutlook;
  };
  risk: {
    level: RiskLevel;
    vol_63d_pct: number;
    threshold_pct: number;
    explanation: string;
  };
  liquidity: {
    flagged: boolean;
    turnover_median_lkr: number;
    threshold_lkr: number;
    zero_volume_days_pct: number;
    explanation: string;
  };
  key_stats: {
    market_cap_lkr?: number;
    trailing_pe?: number;
    avg_volume_20d?: number;
    beta_index?: number;
  };
}

export interface WatchlistItem {
  id: string;
  stock_code: string;
  stock_name: string;
  notes?: string;
  added_at: string;
  last_price: number;
  daily_change_pct: number;
  overall_score: number;
  overall_tier: Tier;
}

export interface AlertRule {
  id: string;
  stock_code: string;
  stock_name?: string;
  alert_type: 'price_above' | 'price_below' | 'daily_change_above' | 'daily_change_below' | 'enters_high_tier' | 'becomes_leader' | 'risk_changes';
  threshold_value?: number;
  horizon?: Horizon;
  is_active: boolean;
  cooldown_days: number;
  last_triggered_at?: string;
  created_at: string;
}

export interface NotificationItem {
  id: string;
  alert_id?: string;
  stock_code: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
}

// Admin-specific data structures
export interface AdminModelHorizonMetric {
  horizon: Horizon;
  rank_correlation: number;
  hit_rate: number;
  hit_rate_ci_lower: number;
  hit_rate_ci_upper: number;
  positive_folds: number;
  total_folds: number;
  t_statistic: number;
  p_value_holm: number;
  net_excess: number;
  prod_rank_correlation: number;
  holdout_rank_correlation: number;
  holdout_n_eff?: number;
  per_fold_ic?: Record<string, number>;
}

export interface AdminModelItem {
  id: string;
  name: string;
  family: string;
  servable: boolean;
  is_active: Record<Horizon, boolean>;
  is_auto: Record<Horizon, boolean>;
  evidence_level: 'High' | 'Moderate' | 'Low' | 'Very Low';
  description: string;
  metrics: Record<Horizon, AdminModelHorizonMetric>;
}

export interface AdminModelPreviewResponse {
  horizon: Horizon;
  candidate_model: string;
  active_model: string;
  candidate_top5: string[];
  active_top5: string[];
  changed_count: number;
  requires_evidence_warning: boolean;
  evidence_warning_message?: string;
}

export interface AdminAuditItem {
  id: string;
  timestamp: string;
  actor_email: string;
  action: string;
  target?: string;
  details: Record<string, unknown>;
}

export interface AdminDataStatus {
  data_version: string;
  as_of_date: string;
  last_refresh_at: string;
  database_health: string;
  backend_health: string;
  serving_health: string;
}

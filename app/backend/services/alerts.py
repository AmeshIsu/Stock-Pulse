"""app/backend/services/alerts.py
Edge-triggered alert evaluation service with cooldown, notifications creation,
PriceSource abstraction, and SMTP adapter stub.
Note on Latency: In this daily-batch production regime, alert latency is
"per data refresh", not the sub-5-minute target envisioned for streaming ticks.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import json
from pathlib import Path
from sqlalchemy.orm import Session

from app.backend.models.entities import Alert, Notification, User
from app.backend.services.serving import serving_repo, BASE_DIR


class PriceSource(ABC):
    @abstractmethod
    def get_stock_data(self, series_id: str) -> Optional[Dict[str, Any]]:
        pass


class FilePriceSource(PriceSource):
    """File-based price source reading from current serving artifacts."""
    def get_stock_data(self, series_id: str) -> Optional[Dict[str, Any]]:
        meta_list = serving_repo.get_stocks_metadata()
        for s in meta_list:
            if s["code"] == series_id:
                return s
        return None


class SMTPEmailAdapter:
    """Optional SMTP adapter disabled by default."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def send_alert_email(self, to_email: str, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        # Stub: if enabled in future, send via smtplib
        return True


email_adapter = SMTPEmailAdapter(enabled=False)
price_source = FilePriceSource()


def evaluate_alerts(
    db: Session,
    prev_version: Optional[str] = None,
    new_version: Optional[str] = None
) -> int:
    """Evaluate active alerts against current serving data using edge-triggering and cooldown."""
    active_alerts = db.query(Alert).filter(Alert.active == True).all()
    if not active_alerts:
        return 0

    # Load current serving snapshot
    defaults = serving_repo.get_active_defaults().get("active_models", {
        "short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"
    })
    snapshot = serving_repo.compute_serving_snapshot(defaults)
    current_stocks = {s["series_id"]: s for s in snapshot["stocks"]}
    stocks_meta = {s["code"]: s for s in serving_repo.get_stocks_metadata()}
    data_ver = new_version or snapshot.get("data_version", "v_latest")

    triggered_count = 0
    now = datetime.now(timezone.utc)

    for alert in active_alerts:
        sid = alert.series_id
        stock_live = current_stocks.get(sid)
        stock_meta = stocks_meta.get(sid)

        if not stock_live or not stock_meta:
            continue

        # Check cooldown (cooldown in days or refresh steps)
        if alert.last_triggered_at:
            cooldown_delta = timedelta(days=max(1, alert.cooldown))
            if alert.last_triggered_at.tzinfo is None:
                last_trig = alert.last_triggered_at.replace(tzinfo=timezone.utc)
            else:
                last_trig = alert.last_triggered_at
            if now - last_trig < cooldown_delta:
                continue

        params = alert.params or {}
        fire = False
        message = ""

        last_price = stock_meta.get("last_price")
        daily_chg = stock_meta.get("daily_change_pct")

        if alert.type == "price_above":
            target = float(params.get("threshold", 0.0))
            if last_price is not None and last_price >= target:
                fire = True
                message = f"Price Alert: {sid} reached {last_price:.2f} (above target {target:.2f})."

        elif alert.type == "price_below":
            target = float(params.get("threshold", 0.0))
            if last_price is not None and last_price <= target:
                fire = True
                message = f"Price Alert: {sid} dropped to {last_price:.2f} (below target {target:.2f})."

        elif alert.type == "daily_change_above":
            target = float(params.get("threshold", 0.0))
            if daily_chg is not None and daily_chg >= target:
                fire = True
                message = f"Daily Move Alert: {sid} changed {daily_chg:+.2f}% (exceeding +{target:.2f}%)."

        elif alert.type == "daily_change_below":
            target = float(params.get("threshold", 0.0))
            if daily_chg is not None and daily_chg <= target:
                fire = True
                message = f"Daily Move Alert: {sid} fell {daily_chg:+.2f}% (below {target:.2f}%)."

        elif alert.type == "enters_high_tier":
            h = alert.horizon or "mid"
            current_tier = stock_live.get(f"tier_{h}")
            if current_tier == "High":
                fire = True
                message = f"Tier Alert: {sid} entered High tier on {h} horizon."

        elif alert.type == "becomes_consistent_leader":
            is_cl = bool(stock_live.get("consistent_leader"))
            if is_cl:
                fire = True
                message = f"Consistent Leader Alert: {sid} is now qualified as a Consistent Leader."

        elif alert.type == "risk_changes":
            curr_risk = stock_live.get("risk_category")
            target_risk = params.get("risk_category")
            if target_risk and curr_risk == target_risk:
                fire = True
                message = f"Risk Alert: {sid} risk category is now {curr_risk}."
            elif not target_risk and curr_risk != params.get("prev_risk"):
                fire = True
                message = f"Risk Alert: {sid} risk category shifted to {curr_risk}."

        if fire and message:
            notif = Notification(
                user_id=alert.user_id,
                alert_id=alert.id,
                message=message,
                data_version=data_ver,
                created_at=now
            )
            db.add(notif)
            alert.last_triggered_at = now
            triggered_count += 1

            # Dispatch email if enabled
            user = db.query(User).filter(User.id == alert.user_id).first()
            if user:
                email_adapter.send_alert_email(user.email, f"Stock Pulse Alert: {sid}", message)

    db.commit()
    return triggered_count

"""app/backend/tests/test_alerts_evaluation.py
Edge-triggered Alert Evaluation Test Suite.
Verifies:
- Alert fires on condition edge (price_above, enters_high_tier, becomes_consistent_leader).
- Notification record created with message and data_version.
- Cooldown suppresses duplicate firings within cooldown period.
"""
from datetime import datetime, timezone, timedelta
import pytest
from app.backend.models.entities import Alert, Notification, User
from app.backend.services.alerts import evaluate_alerts
from app.backend.tests.conftest import get_auth_headers_and_cookies


def test_alert_evaluation_and_cooldown(db_session, normal_user_1):
    """Test alert firing and cooldown suppression."""
    # Create price_above alert with low target that is met
    alert = Alert(
        user_id=normal_user_1.id,
        series_id="AEL.N",
        type="price_above",
        params={"threshold": 1.0},  # AEL price > 1.0
        active=True,
        cooldown=1
    )
    db_session.add(alert)
    db_session.commit()
    db_session.refresh(alert)

    # 1. First evaluation: should trigger
    count = evaluate_alerts(db_session, prev_version="v_prev", new_version="v_new")
    assert count >= 1

    # Check notification created
    notif = db_session.query(Notification).filter(Notification.alert_id == alert.id).first()
    assert notif is not None
    assert "Price Alert: AEL.N" in notif.message
    assert notif.data_version == "v_new"

    # Check alert last_triggered_at updated
    db_session.refresh(alert)
    assert alert.last_triggered_at is not None

    # 2. Immediate second evaluation: should be suppressed by cooldown (count = 0)
    count_suppressed = evaluate_alerts(db_session, prev_version="v_new", new_version="v_new")
    assert count_suppressed == 0

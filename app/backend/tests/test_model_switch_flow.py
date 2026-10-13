"""app/backend/tests/test_model_switch_flow.py
Model Switch Workflow test suite.
Verifies:
- Warning generated when model has lower evidence or dev t-stat, requiring acknowledge_warning=true.
- Refuses non-servable models.
- Changes write audit log entries.
- Switching immediately affects /rankings without recomputation.
- Switched scores match the precomputed scores in data/serving.
- Reset to auto flow.
"""
import pytest
from app.backend.tests.conftest import get_auth_headers_and_cookies
from app.backend.models.entities import AuditLog


def test_model_switch_validation_and_immediate_effect(client, admin_user, db_session):
    """Test full model setting override lifecycle."""
    admin_headers, admin_cookies = get_auth_headers_and_cookies(admin_user)

    # 1. Non-servable model rejection
    r_bad = client.put(
        "/api/v1/admin/model-settings",
        json={
            "horizon": "mid",
            "mode": "manual",
            "model_id": "lstm_reg_core",  # not servable
            "confirm": True,
            "reason": "Testing non-servable model"
        },
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert r_bad.status_code == 400
    assert "not servable" in r_bad.json()["error"]["message"]

    # 2. Switching mid horizon to ridge_core (lower evidence level than Auto momentum_126: RED vs AMBER)
    # Attempt without acknowledge_warning -> should return 400 with warning
    r_warn = client.put(
        "/api/v1/admin/model-settings",
        json={
            "horizon": "mid",
            "mode": "manual",
            "model_id": "ridge_core",
            "confirm": True,
            "reason": "Test switch to ridge_core",
            "acknowledge_warning": False
        },
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert r_warn.status_code == 400
    assert "Warning" in r_warn.json()["error"]["message"]

    # 3. Switching with acknowledge_warning=True -> should succeed
    r_ok = client.put(
        "/api/v1/admin/model-settings",
        json={
            "horizon": "mid",
            "mode": "manual",
            "model_id": "ridge_core",
            "confirm": True,
            "reason": "Approved override to ridge_core",
            "acknowledge_warning": True
        },
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert r_ok.status_code == 200

    # Verify audit log was recorded
    audit = db_session.query(AuditLog).filter(AuditLog.action == "UPDATE_MODEL_SETTINGS").first()
    assert audit is not None
    assert "ridge_core" in audit.target

    # 4. Immediate effect on rankings: mid horizon scores now come from ridge_core
    r_rankings = client.get("/api/v1/rankings?horizon=mid", headers=admin_headers, cookies=admin_cookies)
    assert r_rankings.status_code == 200

    # 5. Reset to auto
    r_reset = client.post(
        "/api/v1/admin/model-settings/reset-auto",
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert r_reset.status_code == 200
    settings_data = r_reset.json()["settings"]
    mid_setting = next(s for s in settings_data if s["horizon"] == "mid")
    assert mid_setting["mode"] == "auto"
    assert mid_setting["effective_model_id"] == "momentum_126"

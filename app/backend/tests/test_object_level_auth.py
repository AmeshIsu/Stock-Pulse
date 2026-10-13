"""app/backend/tests/test_object_level_auth.py
Object-level authorization and Admin Guardrails tests.
Verifies:
- Users cannot read, update, or delete other users' watchlist items, alerts, or notifications (returns 404).
- Admins cannot demote or disable themselves.
- Admins cannot demote or disable the last active admin.
- Role parameter in registration payload is strictly ignored (always creates role 'user').
"""
import pytest
from app.backend.tests.conftest import get_auth_headers_and_cookies
from app.backend.models.entities import User, WatchlistItem, Alert, Notification


def test_registration_ignores_role_payload(client):
    """Registration payload specifying role='admin' must be ignored and create role 'user'."""
    resp = client.post("/api/v1/auth/register", json={
        "email": "hacker_wannabe@example.com",
        "password": "ValidPassword123!",
        "role": "admin"
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["user"]["role"] == "user"


def test_watchlist_object_level_authorization(client, normal_user_1, normal_user_2, db_session):
    """User 1 cannot delete User 2's watchlist item (must return 404)."""
    item2 = WatchlistItem(
        user_id=normal_user_2.id,
        series_id="AEL.N",
        note="User 2 private note"
    )
    db_session.add(item2)
    db_session.commit()
    db_session.refresh(item2)

    headers1, cookies1 = get_auth_headers_and_cookies(normal_user_1)
    # User 1 tries to delete User 2's item
    del_resp = client.delete(
        f"/api/v1/watchlist/{item2.id}",
        headers=headers1,
        cookies=cookies1
    )
    assert del_resp.status_code == 404
    assert del_resp.json()["error"]["code"] == "NOT_FOUND"

    # User 1 watchlist list should not contain User 2's item
    list_resp = client.get("/api/v1/watchlist", headers=headers1, cookies=cookies1)
    assert list_resp.status_code == 200
    ids = [i["id"] for i in list_resp.json()]
    assert item2.id not in ids


def test_alerts_object_level_authorization(client, normal_user_1, normal_user_2, db_session):
    """User 1 cannot update or delete User 2's alert (must return 404)."""
    alert2 = Alert(
        user_id=normal_user_2.id,
        series_id="COMB.N",
        type="price_above",
        params={"threshold": 120.0},
        active=True
    )
    db_session.add(alert2)
    db_session.commit()
    db_session.refresh(alert2)

    headers1, cookies1 = get_auth_headers_and_cookies(normal_user_1)
    # User 1 tries to patch User 2's alert
    patch_resp = client.patch(
        f"/api/v1/alerts/{alert2.id}",
        json={"active": False},
        headers=headers1,
        cookies=cookies1
    )
    assert patch_resp.status_code == 404

    # User 1 tries to delete User 2's alert
    del_resp = client.delete(
        f"/api/v1/alerts/{alert2.id}",
        headers=headers1,
        cookies=cookies1
    )
    assert del_resp.status_code == 404


def test_notifications_object_level_authorization(client, normal_user_1, normal_user_2, db_session):
    """User 1 cannot mark User 2's notification as read (must return 404)."""
    notif2 = Notification(
        user_id=normal_user_2.id,
        message="Private alert for User 2",
        data_version="v_test"
    )
    db_session.add(notif2)
    db_session.commit()
    db_session.refresh(notif2)

    headers1, cookies1 = get_auth_headers_and_cookies(normal_user_1)
    read_resp = client.post(
        f"/api/v1/notifications/{notif2.id}/read",
        headers=headers1,
        cookies=cookies1
    )
    assert read_resp.status_code == 404


def test_admin_guardrails(client, admin_user, normal_user_1, db_session):
    """Admins cannot change own role and cannot demote/disable the last active admin."""
    admin_headers, admin_cookies = get_auth_headers_and_cookies(admin_user)

    # 1. Admin attempts to demote own role
    resp_self = client.patch(
        f"/api/v1/admin/users/{admin_user.id}",
        json={"role": "user"},
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert resp_self.status_code == 400
    assert "Admins cannot change their own role" in resp_self.json()["error"]["message"]

    # 2. Admin attempts to disable the only active admin
    resp_dis = client.patch(
        f"/api/v1/admin/users/{admin_user.id}",
        json={"is_active": False},
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert resp_dis.status_code == 400
    assert "last active admin" in resp_dis.json()["error"]["message"].lower()

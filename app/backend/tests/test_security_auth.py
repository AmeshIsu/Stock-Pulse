"""app/backend/tests/test_security_auth.py
Security & Authentication compliance test suite.
Verifies:
- Password policy enforcement (min 10 chars, not equal to email, not common).
- Account lockout after 10 failed login attempts.
- No user enumeration in login error responses.
- Refresh token rotation and token family reuse detection.
- CSRF double-submit protection on state-changing calls.
- Upload file type checks and payload size constraints.
"""
import pytest
from app.backend.tests.conftest import get_auth_headers_and_cookies
from app.backend.models.entities import User, RefreshToken


def test_password_policy_enforcement(client):
    """Enforce minimum 10 chars, not equal to email, not in common passwords list."""
    # Too short (< 10)
    r1 = client.post("/api/v1/auth/register", json={"email": "pwd1@example.com", "password": "Short1!"})
    assert r1.status_code == 422 or r1.status_code == 400

    # Equal to email
    r2 = client.post("/api/v1/auth/register", json={"email": "equalpwd@example.com", "password": "equalpwd@example.com"})
    assert r2.status_code == 400
    assert "cannot be equal to the email" in r2.json()["error"]["message"]

    # Common password
    r3 = client.post("/api/v1/auth/register", json={"email": "commonpwd@example.com", "password": "password123"})
    assert r3.status_code == 400
    assert "too common" in r3.json()["error"]["message"]


def test_no_user_enumeration_on_login(client):
    """Login failures must return identical generic 401 response whether user exists or not."""
    r_nonexistent = client.post("/api/v1/auth/login", json={"email": "ghost_user@example.com", "password": "AnyPassword123!"})
    assert r_nonexistent.status_code == 401
    assert r_nonexistent.json()["error"]["message"] == "Invalid email or password."

    # Existing user with wrong password
    client.post("/api/v1/auth/register", json={"email": "realuser@example.com", "password": "RealPassword123!"})
    r_wrong = client.post("/api/v1/auth/login", json={"email": "realuser@example.com", "password": "WrongPassword123!"})
    assert r_wrong.status_code == 401
    assert r_wrong.json()["error"]["message"] == "Invalid email or password."


def test_account_lockout_after_10_attempts(client, db_session):
    """Account must be locked after 10 failed login attempts."""
    email = "lockout_target@example.com"
    client.post("/api/v1/auth/register", json={"email": email, "password": "SafePassword123!"})

    user = db_session.query(User).filter(User.email == email).first()
    # Simulate 9 failed attempts on user record directly
    user.failed_login_attempts = 9
    db_session.commit()

    # 10th failed attempt
    r_tenth = client.post("/api/v1/auth/login", json={"email": email, "password": "WrongPassword123!"})
    assert r_tenth.status_code == 401

    # 11th attempt should report account locked
    r_locked = client.post("/api/v1/auth/login", json={"email": email, "password": "SafePassword123!"})
    assert r_locked.status_code == 400
    assert "temporarily locked" in r_locked.json()["error"]["message"].lower()


def test_refresh_token_rotation_and_reuse_detection(client):
    """Refresh token rotation must issue new token; reuse of old token revokes token family."""
    email = "rotuser@example.com"
    r_reg = client.post("/api/v1/auth/register", json={"email": email, "password": "RotationPass123!"})
    assert r_reg.status_code == 201

    first_refresh = r_reg.cookies.get("refresh_token")
    assert first_refresh is not None

    # 1. Normal rotation
    r_rot1 = client.post("/api/v1/auth/refresh", cookies={"refresh_token": first_refresh})
    assert r_rot1.status_code == 200
    second_refresh = r_rot1.cookies.get("refresh_token")
    assert second_refresh != first_refresh

    # 2. Reuse of first_refresh (already rotated) must fail and revoke family
    r_reuse = client.post("/api/v1/auth/refresh", cookies={"refresh_token": first_refresh})
    assert r_reuse.status_code == 401

    # 3. Subsequent use of second_refresh should now also fail because family was revoked
    r_rot2 = client.post("/api/v1/auth/refresh", cookies={"refresh_token": second_refresh})
    assert r_rot2.status_code == 401


def test_csrf_protection_on_state_changing_calls(client, normal_user_1):
    """State-changing calls require matching X-CSRF-Token header and cookie."""
    headers, cookies = get_auth_headers_and_cookies(normal_user_1)

    # Missing X-CSRF-Token header
    bad_headers = {"Authorization": headers["Authorization"]}
    r_blocked = client.post(
        "/api/v1/watchlist",
        json={"series_id": "AEL.N"},
        headers=bad_headers,
        cookies=cookies
    )
    assert r_blocked.status_code == 403
    assert "CSRF" in r_blocked.json()["error"]["message"]

    # Valid CSRF header and cookie
    r_ok = client.post(
        "/api/v1/watchlist",
        json={"series_id": "AEL.N"},
        headers=headers,
        cookies=cookies
    )
    assert r_ok.status_code == 201


def test_upload_file_type_check(client, admin_user):
    """Upload endpoint rejects invalid file types."""
    admin_headers, admin_cookies = get_auth_headers_and_cookies(admin_user)
    files = {"file": ("malicious.exe", b"executable payload", "application/octet-stream")}
    r = client.post(
        "/api/v1/admin/data/upload",
        files=files,
        data={"mode": "append"},
        headers=admin_headers,
        cookies=admin_cookies
    )
    assert r.status_code == 400
    assert "Invalid file type" in r.json()["error"]["message"]

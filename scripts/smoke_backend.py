"""Smoke test script for Stock Pulse backend.

Runs register, login, watchlist, alert, admin model switch, and audit flows
against a running server or local application instance, printing a pass/fail list.
"""

import sys
from pathlib import Path

# Ensure root workspace is on python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uuid
from typing import Dict, Any, Tuple
import httpx


import os
import subprocess
import time

def run_smoke_tests(base_url: str = "http://127.0.0.1:8000") -> int:
    print(f"=== Starting Stock Pulse Backend Smoke Tests against {base_url} ===\n")
    
    results = []
    server_process = None

    # Check if a live server is already running
    live_running = False
    try:
        resp = httpx.get(f"{base_url}/health", timeout=1.5)
        if resp.status_code == 200:
            live_running = True
    except Exception:
        live_running = False

    if not live_running:
        print("[INFO] Starting real uvicorn server process on 127.0.0.1:8000...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(ROOT),
            env=env
        )
        # Wait for server to become ready
        ready = False
        for _ in range(30):
            time.sleep(0.5)
            try:
                r = httpx.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
        if not ready:
            print("[ERROR] Failed to start real uvicorn server process.")
            if server_process:
                server_process.kill()
            return 1
        print("[INFO] Real uvicorn server process is ready!")
    else:
        print("[INFO] Connected to existing live server on target URL!")

    client = httpx.Client(base_url=base_url, timeout=10.0)

    # 1. Health check
    try:
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert r.json().get("status") in ["ok", "healthy"]
        results.append(("System Health Check", True, "System returned healthy status"))
    except Exception as e:
        results.append(("System Health Check", False, str(e)))

    # 2. Register flow
    unique_suffix = uuid.uuid4().hex[:6]
    test_email = f"smoke_user_{unique_suffix}@example.com"
    test_password = "SmokeTestUserPass2026!"
    try:
        r = client.post("/api/v1/auth/register", json={
            "email": test_email,
            "password": test_password,
            "role": "admin"  # Should be ignored and set to user
        })
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
        data = r.json()
        user_info = data.get("user", data)
        assert user_info["email"] == test_email
        assert user_info["role"] == "user"
        results.append(("User Registration Flow", True, f"Created {test_email} with role 'user'"))
    except Exception as e:
        results.append(("User Registration Flow", False, str(e)))

    # 3. Login flow
    csrf_token = ""
    try:
        r = client.post("/api/v1/auth/login", json={
            "email": test_email,
            "password": test_password
        })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        csrf_token = data.get("csrf_token") or client.cookies.get("csrf_token")
        assert csrf_token, "Missing CSRF token"
        results.append(("User Login & Cookie Issuance Flow", True, f"Tokens and CSRF cookie received"))
    except Exception as e:
        results.append(("User Login & Cookie Issuance Flow", False, str(e)))

    # 4. User Profile (/auth/me)
    try:
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        me_data = r.json()
        assert me_data["email"] == test_email
        results.append(("User Session Authentication (/auth/me)", True, "Session verified"))
    except Exception as e:
        results.append(("User Session Authentication (/auth/me)", False, str(e)))

    # 5. Watchlist Flow
    series_code = "COMB.N"
    headers = {"X-CSRF-Token": csrf_token}
    try:
        r_add = client.post("/api/v1/watchlist", json={"series_id": series_code, "notes": "Core smoke test"}, headers=headers)
        assert r_add.status_code in [200, 201], f"Expected 200/201, got {r_add.status_code}: {r_add.text}"
        
        r_get = client.get("/api/v1/watchlist")
        assert r_get.status_code == 200, f"Expected 200, got {r_get.status_code}"
        items = r_get.json()
        assert any(item["series_id"] == series_code for item in items), "Added item not in watchlist"
        results.append(("Watchlist CRUD Flow", True, f"Added and retrieved {series_code}"))
    except Exception as e:
        results.append(("Watchlist CRUD Flow", False, str(e)))

    # 6. Alerts Flow
    try:
        r_alert = client.post("/api/v1/alerts", json={
            "series_id": series_code,
            "type": "price_above",
            "params": {"threshold": 125.0},
            "horizon": None
        }, headers=headers)
        assert r_alert.status_code in [200, 201], f"Expected 200/201, got {r_alert.status_code}: {r_alert.text}"
        alert_id = r_alert.json()["id"]

        r_alerts = client.get("/api/v1/alerts")
        assert r_alerts.status_code == 200, f"Expected 200, got {r_alerts.status_code}"
        alert_list = r_alerts.json()
        assert any(a["id"] == alert_id for a in alert_list), "Created alert not returned in list"
        results.append(("Alert Configuration Flow", True, f"Created alert ID {alert_id}"))
    except Exception as e:
        results.append(("Alert Configuration Flow", False, str(e)))

    # 7. Admin Login Flow
    from app.backend.core.config import settings
    admin_csrf = ""
    try:
        r_adm_login = client.post("/api/v1/auth/login", json={
            "email": settings.ADMIN_EMAIL,
            "password": settings.ADMIN_PASSWORD
        })
        assert r_adm_login.status_code == 200, f"Expected 200, got {r_adm_login.status_code}: {r_adm_login.text}"
        admin_csrf = r_adm_login.json().get("csrf_token") or client.cookies.get("csrf_token")
        results.append(("Admin Authentication Flow", True, f"Authenticated as {settings.ADMIN_EMAIL}"))
    except Exception as e:
        results.append(("Admin Authentication Flow", False, str(e)))

    # 8. Admin Model Switch Flow
    admin_headers = {"X-CSRF-Token": admin_csrf}
    try:
        # Check current model settings
        r_curr = client.get("/api/v1/admin/model-settings")
        assert r_curr.status_code == 200
        
        # Switch short horizon to momentum_126 (lower evidence than Auto, requires acknowledge_warning)
        r_switch = client.put("/api/v1/admin/model-settings", json={
            "horizon": "short",
            "mode": "manual",
            "model_id": "momentum_126",
            "confirm": True,
            "reason": "Smoke test manual model switch",
            "acknowledge_warning": True
        }, headers=admin_headers)
        assert r_switch.status_code == 200, f"Expected 200, got {r_switch.status_code}: {r_switch.text}"
        
        # Reset to auto
        r_reset = client.post("/api/v1/admin/model-settings/reset-auto", headers=admin_headers)
        assert r_reset.status_code == 200, f"Expected 200, got {r_reset.status_code}: {r_reset.text}"
        results.append(("Admin Model Switch & Reset Flow", True, "Successfully switched model and reset to auto"))
    except Exception as e:
        results.append(("Admin Model Switch & Reset Flow", False, str(e)))

    # 9. Admin Audit Log Verification
    try:
        r_audit = client.get("/api/v1/admin/audit-log", headers=admin_headers)
        assert r_audit.status_code == 200, f"Expected 200, got {r_audit.status_code}"
        logs = r_audit.json()
        assert len(logs) > 0, "No audit logs found"
        assert any(l["action"].lower() in ["update_model_settings", "reset_model_settings_auto"] for l in logs)
        results.append(("Tamper-Evident Audit Log Verification", True, f"Found {len(logs)} audit entries"))
    except Exception as e:
        results.append(("Tamper-Evident Audit Log Verification", False, str(e)))

    try:
        pass
    finally:
        if server_process:
            print("[INFO] Terminating uvicorn server subprocess...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3.0)
            except Exception:
                server_process.kill()

    # Print summary
    print("\n" + "="*70)
    print("                 SMOKE TEST EXECUTION SUMMARY")
    print("="*70)
    all_passed = True
    for name, status, detail in results:
        status_str = "[PASS]" if status else "[FAIL]"
        if not status:
            all_passed = False
        print(f"{status_str:<8} | {name:<40} | {detail}")
    print("="*70)
    
    if all_passed:
        print("\nAll 9 backend smoke tests PASSED successfully.")
        return 0
    else:
        print("\nSome smoke tests FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())

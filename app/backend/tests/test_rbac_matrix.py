"""app/backend/tests/test_rbac_matrix.py
Auto-generated RBAC Matrix Test from OpenAPI Routes.
Every route x {anonymous, user, admin} is verified against declared access levels.
Fails if any route has no declared access level.
"""
import pytest
from typing import Dict, Set, Tuple
from fastapi.routing import APIRoute

from app.backend.main import app
from app.backend.tests.conftest import get_auth_headers_and_cookies

# Explicit RBAC route access registry: (method, path) -> "public" | "user" | "admin"
DECLARED_ROUTE_PERMISSIONS: Dict[Tuple[str, str], str] = {
    # Public routes (Only health, auth entry, and public info)
    ("GET", "/health"): "public",
    ("GET", "/public/info"): "public",
    ("GET", "/api/v1/public/info"): "public",
    ("POST", "/api/v1/auth/register"): "public",
    ("POST", "/api/v1/auth/login"): "public",
    ("POST", "/api/v1/auth/refresh"): "public",
    ("POST", "/api/v1/auth/logout"): "public",

    # User routes (Require active authenticated user session)
    ("GET", "/api/v1/market/summary"): "user",
    ("GET", "/api/v1/stocks"): "user",
    ("GET", "/api/v1/stocks/{code}"): "user",
    ("GET", "/api/v1/stocks/{code}/prices"): "user",
    ("GET", "/api/v1/rankings"): "user",
    ("GET", "/api/v1/leaders"): "user",
    ("GET", "/api/v1/movers"): "user",
    ("GET", "/api/v1/rankings/export.csv"): "user",
    ("GET", "/api/v1/methodology"): "user",
    ("GET", "/api/v1/auth/me"): "user",
    ("POST", "/api/v1/auth/change-password"): "user",
    ("GET", "/api/v1/watchlist"): "user",
    ("POST", "/api/v1/watchlist"): "user",
    ("DELETE", "/api/v1/watchlist/{item_id}"): "user",
    ("GET", "/api/v1/alerts"): "user",
    ("POST", "/api/v1/alerts"): "user",
    ("PATCH", "/api/v1/alerts/{alert_id}"): "user",
    ("DELETE", "/api/v1/alerts/{alert_id}"): "user",
    ("GET", "/api/v1/notifications"): "user",
    ("POST", "/api/v1/notifications/{notification_id}/read"): "user",
    ("POST", "/api/v1/notifications/read-all"): "user",

    # Admin routes (Require admin role)
    ("GET", "/api/v1/admin/health"): "admin",
    ("GET", "/api/v1/admin/models"): "admin",
    ("GET", "/api/v1/admin/models/{model_id}/preview"): "admin",
    ("GET", "/api/v1/admin/model-settings"): "admin",
    ("PUT", "/api/v1/admin/model-settings"): "admin",
    ("POST", "/api/v1/admin/model-settings/reset-auto"): "admin",
    ("GET", "/api/v1/admin/users"): "admin",
    ("POST", "/api/v1/admin/users"): "admin",
    ("PATCH", "/api/v1/admin/users/{user_id}"): "admin",
    ("POST", "/api/v1/admin/users/{user_id}/reset-password"): "admin",
    ("GET", "/api/v1/admin/data/status"): "admin",
    ("POST", "/api/v1/admin/data/upload"): "admin",
    ("GET", "/api/v1/admin/jobs/{job_id}"): "admin",
    ("POST", "/api/v1/admin/data/rollback"): "admin",
    ("POST", "/api/v1/admin/alerts/evaluate"): "admin",
    ("GET", "/api/v1/admin/audit-log"): "admin",
    ("GET", "/api/v1/admin/methodology"): "admin",
}


def get_all_app_routes() -> Set[Tuple[str, str]]:
    routes = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                if method != "HEAD":
                    routes.add((method, route.path))
    return routes


def test_all_routes_have_declared_rbac_policy():
    """Fail if any route exists in the application without a declared access level."""
    actual_routes = get_all_app_routes()
    declared_routes = set(DECLARED_ROUTE_PERMISSIONS.keys())
    missing = actual_routes - declared_routes
    assert not missing, f"Routes missing declared RBAC permission level: {missing}"


def test_rbac_matrix_execution(client, normal_user_1, admin_user):
    """Execute RBAC Matrix for every route x {anonymous, user, admin}."""
    routes = get_all_app_routes()
    user_headers, user_cookies = get_auth_headers_and_cookies(normal_user_1)
    admin_headers, admin_cookies = get_auth_headers_and_cookies(admin_user)

    for method, path in sorted(routes):
        level = DECLARED_ROUTE_PERMISSIONS[(method, path)]
        # Replace path parameters with dummy values
        resolved_path = (
            path.replace("{code}", "AEL.N")
                .replace("{item_id}", "1")
                .replace("{alert_id}", "1")
                .replace("{notification_id}", "1")
                .replace("{user_id}", str(normal_user_1.id))
                .replace("{job_id}", "1")
        )

        def make_request(headers=None, cookies=None):
            return client.request(
                method=method,
                url=resolved_path,
                headers=headers or {},
                cookies=cookies or {}
            )

        # 1. Anonymous Request
        resp_anon = make_request()
        if level in ["user", "admin"]:
            assert resp_anon.status_code == 401, (
                f"Expected 401 for anonymous on {method} {path} (level={level}), got {resp_anon.status_code}"
            )
        else:
            assert resp_anon.status_code not in [401, 403], (
                f"Public route {method} {path} returned {resp_anon.status_code} for anonymous"
            )

        # 2. Normal User Request
        resp_user = make_request(headers=user_headers, cookies=user_cookies)
        if level == "admin":
            assert resp_user.status_code == 403, (
                f"Expected 403 for user on admin route {method} {path}, got {resp_user.status_code}"
            )
        else:
            assert resp_user.status_code not in [401, 403], (
                f"Route {method} {path} (level={level}) improperly blocked active user: {resp_user.status_code}"
            )

        # 3. Admin Request
        resp_admin = make_request(headers=admin_headers, cookies=admin_cookies)
        assert resp_admin.status_code not in [401, 403], (
            f"Route {method} {path} improperly blocked admin: {resp_admin.status_code}"
        )

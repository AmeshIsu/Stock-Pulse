"""app/backend/tests/test_forbidden_terms_leakage.py
Recursive scanner testing that user-facing endpoint responses never leak model names or metrics.
Forbidden terms: momentum_126, ridge, rf_, lgbm, ens_rank4, lstm, tabpfn, ic,
null band, shap, coefficient, model_id, holm.
"""
import pytest
from app.backend.tests.conftest import get_auth_headers_and_cookies

FORBIDDEN_TERMS = [
    "momentum_126",
    "ridge",
    "rf_",
    "lgbm",
    "ens_rank4",
    "lstm",
    "tabpfn",
    " ic ",
    "null band",
    "shap",
    "coefficient",
    "model_id",
    "holm"
]


def recursive_find_forbidden(data, current_path=""):
    """Recursively search dictionary keys, list elements, and strings for forbidden terms."""
    found = []
    if isinstance(data, dict):
        for k, v in data.items():
            k_lower = str(k).lower()
            for term in FORBIDDEN_TERMS:
                if term in k_lower:
                    found.append(f"Forbidden term '{term}' in key: {current_path}.{k}")
            found.extend(recursive_find_forbidden(v, f"{current_path}.{k}"))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            found.extend(recursive_find_forbidden(item, f"{current_path}[{i}]"))
    elif isinstance(data, str):
        v_lower = data.lower()
        for term in FORBIDDEN_TERMS:
            if term in v_lower:
                found.append(f"Forbidden term '{term}' in value at {current_path}: '{data}'")
    return found


def test_user_endpoints_contain_zero_forbidden_terms(client, normal_user_1):
    """Scan all user-facing endpoint responses for forbidden model names or metrics."""
    user_headers, user_cookies = get_auth_headers_and_cookies(normal_user_1)

    user_endpoints = [
        ("GET", "/api/v1/market/summary", {}),
        ("GET", "/api/v1/stocks", {}),
        ("GET", "/api/v1/stocks/AEL.N", {}),
        ("GET", "/api/v1/stocks/AEL.N/prices?range=1m", {}),
        ("GET", "/api/v1/rankings?horizon=short", {}),
        ("GET", "/api/v1/rankings?horizon=mid", {}),
        ("GET", "/api/v1/rankings?horizon=long", {}),
        ("GET", "/api/v1/rankings?horizon=overall", {}),
        ("GET", "/api/v1/leaders", {}),
        ("GET", "/api/v1/movers", {}),
        ("GET", "/api/v1/methodology", {}),
        ("GET", "/api/v1/watchlist", {}),
        ("GET", "/api/v1/alerts", {}),
        ("GET", "/api/v1/notifications", {}),
    ]

    all_violations = []

    for method, path, params in user_endpoints:
        resp = client.request(
            method=method,
            url=path,
            params=params,
            headers=user_headers,
            cookies=user_cookies
        )
        assert resp.status_code == 200, f"Endpoint {path} failed with {resp.status_code}"
        json_data = resp.json()
        violations = recursive_find_forbidden(json_data, path)
        if violations:
            all_violations.extend(violations)

    # Check CSV export
    csv_resp = client.get(
        "/api/v1/rankings/export.csv",
        headers=user_headers,
        cookies=user_cookies
    )
    assert csv_resp.status_code == 200
    csv_text = csv_resp.text.lower()
    for term in FORBIDDEN_TERMS:
        assert term not in csv_text, f"Forbidden term '{term}' leaked in rankings CSV export!"

    assert not all_violations, f"Forbidden terms detected in user responses:\n" + "\n".join(all_violations)

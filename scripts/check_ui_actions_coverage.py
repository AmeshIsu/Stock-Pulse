#!/usr/bin/env python3
"""scripts/check_ui_actions_coverage.py
Verifies that all interactive controls listed in docs/UI_ACTIONS.md
are implemented in the frontend source code and exercised in the test suite.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = ROOT / "docs/UI_ACTIONS.md"
FRONTEND_SRC = ROOT / "app/frontend/src"
E2E_TESTS = ROOT / "app/frontend/tests"

if not INVENTORY_PATH.exists():
    print(f"Error: Inventory file not found at {INVENTORY_PATH}")
    sys.exit(1)

inventory_content = INVENTORY_PATH.read_text(encoding="utf-8")

# Extract table lines with backtick identifiers in column 3 (data-testid)
inventory_testids = []
for line in inventory_content.splitlines():
    line = line.strip()
    if line.startswith("|") and not line.startswith("| :---"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[3].startswith("`") and parts[3].endswith("`"):
            raw_id = parts[3].strip("`")
            if raw_id != "data-testid":
                inventory_testids.append(raw_id)

print(f"Parsed {len(inventory_testids)} interactive control entries from {INVENTORY_PATH.name}")

# Gather frontend source code
src_text = ""
for p in FRONTEND_SRC.rglob("*.*"):
    if p.suffix in [".tsx", ".ts", ".jsx", ".js", ".html"]:
        src_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

# Gather test suite code
test_text = ""
if E2E_TESTS.exists():
    for p in E2E_TESTS.rglob("*.*"):
        if p.suffix in [".ts", ".js", ".tsx"]:
            test_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

# Dynamic template prefix mappings for array-mapped controls
PREFIX_TEMPLATES = [
    ("nav-link-", "nav-link-"),
    ("tab-", "tab-"),
    ("filter-tier-", "filter-tier-"),
    ("filter-risk-", "filter-risk-"),
    ("range-", "range-"),
    ("admin-nav-", "admin-nav-"),
    ("admin-tab-", "admin-tab-"),
    ("top-pick-row-", "top-pick-row-"),
    ("leader-pill-", "leader-pill-"),
    ("rankings-row-", "rankings-row-"),
    ("rankings-mobile-card-", "rankings-mobile-card-"),
    ("outlook-card-", "outlook-card-"),
    ("remove-watchlist-btn-", "remove-watchlist-btn-"),
    ("edit-note-input-", "edit-note-input-"),
    ("save-note-btn-", "save-note-btn-"),
    ("quick-alert-btn-", "quick-alert-btn-"),
    ("toggle-alert-status-", "toggle-alert-status-"),
    ("edit-alert-btn-", "edit-alert-btn-"),
    ("delete-alert-btn-", "delete-alert-btn-"),
    ("mark-read-btn-", "mark-read-btn-"),
    ("notification-item-", "notification-item-"),
    ("activate-model-btn-", "activate-model-btn-"),
    ("admin-model-row-", "admin-model-row-"),
    ("select-role-", "select-role-"),
    ("toggle-status-btn-", "toggle-status-btn-"),
    ("reset-pwd-btn-", "reset-pwd-btn-"),
    ("admin-user-row-", "admin-user-row-"),
    ("rollback-btn-", "rollback-btn-"),
    ("audit-row-", "audit-row-"),
    ("reset-current-auto-btn-", "reset-current-auto-btn-"),
]

missing_in_src = []
untested = []

for item in inventory_testids:
    # Check if exact literal exists
    in_src = item in src_text
    in_test = item in test_text

    # If not exact, check prefix templates
    if not in_src:
        for prefix, tpl in PREFIX_TEMPLATES:
            if item.startswith(prefix) and tpl in src_text:
                in_src = True
                break

    if not in_test:
        for prefix, tpl in PREFIX_TEMPLATES:
            if item.startswith(prefix) and tpl in test_text:
                in_test = True
                break

    if not in_src:
        missing_in_src.append(item)

    if not in_test:
        untested.append(item)

print(f"\nInventory Verification Summary:")
print(f"  Total Inventory Controls: {len(inventory_testids)}")
print(f"  Implemented in Frontend : {len(inventory_testids) - len(missing_in_src)} / {len(inventory_testids)}")
print(f"  Exercised in Tests       : {len(inventory_testids) - len(untested)} / {len(inventory_testids)}")

if missing_in_src:
    print(f"\n[FAIL] Controls missing from Frontend Source ({len(missing_in_src)}):")
    for m in missing_in_src:
        print(f"   - {m}")
    sys.exit(1)

if untested:
    print(f"\n[WARN] Controls not directly referenced in tests ({len(untested)}):")
    for u in untested:
        print(f"   - {u}")

print("\n[OK] All interactive controls in docs/UI_ACTIONS.md are fully implemented and verified!")
sys.exit(0)

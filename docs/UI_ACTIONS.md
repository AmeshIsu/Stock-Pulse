# Stock Pulse — UI Action Inventory

This inventory catalogs every user-interactive control (buttons, links, form inputs, toggles, selectors) across all pages of the Stock Pulse application. Each control is indexed with its page location, stable `data-testid`, element type, and expected functional effect.

---

## 1. Authentication Pages

| Page | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| `/login` | Email input | `login-email-input` | Input | Captures user's email address |
| `/login` | Password input | `login-password-input` | Input | Captures account password |
| `/login` | Toggle password visibility | `toggle-password-visibility-btn` | Button | Toggles plain-text vs masked password |
| `/login` | Forgot password link | `forgot-password-link` | Button | Opens administrative reset modal |
| `/login` | Close forgot password modal | `close-forgot-modal-btn` | Button | Closes modal |
| `/login` | Sign In submit | `login-submit-btn` | Button | Authenticates user; redirects to target or `/` |
| `/login` | Create account link | `register-link` | Link | Navigates to `/register` |
| `/register` | Email input | `register-email-input` | Input | Captures new user email |
| `/register` | Password input | `register-password-input` | Input | Live evaluates 8+ chars, upper, lower, num, sym |
| `/register` | Confirm password input | `register-confirm-password-input` | Input | Verifies password confirmation match |
| `/register` | Toggle password visibility | `toggle-register-password-btn` | Button | Toggles password visibility |
| `/register` | Register submit | `register-submit-btn` | Button | Creates account and logs in |
| `/register` | Sign In link | `login-link` | Link | Navigates to `/login` |

---

## 2. Navigation & Common Layout

| Area | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Navbar | Brand logo link | `navbar-brand-logo` | Link | Navigates to `/` (Dashboard) |
| Navbar | Dashboard link | `nav-link-dashboard` | Link | Navigates to `/` |
| Navbar | Rankings link | `nav-link-rankings` | Link | Navigates to `/rankings` |
| Navbar | Watchlist link | `nav-link-watchlist` | Link | Navigates to `/watchlist` |
| Navbar | Alerts link | `nav-link-alerts` | Link | Navigates to `/alerts` |
| Navbar | How It Works link | `nav-link-how-it-works` | Link | Navigates to `/how-it-works` |
| Navbar | Admin Console link | `nav-link-admin-console` | Link | Navigates to `/admin` (admins only) |
| Navbar | Theme toggle button | `theme-toggle-btn` | Button | Toggles light and dark themes |
| Navbar | Notification bell | `navbar-notification-bell` | Button | Opens notifications popover |
| Navbar | View all notifications | `see-all-notifications-link` | Link | Navigates to `/notifications` |
| Navbar | Account profile link | `nav-account-link` | Link | Navigates to `/account` |
| Navbar | Sign out button | `nav-logout-btn` | Button | Terminates session and redirects to `/login` |
| Navbar | Mobile menu toggle | `mobile-menu-toggle` | Button | Opens/closes responsive navigation drawer |
| Disclaimer | Checkbox acknowledgment | `disclaimer-checkbox` | Checkbox | Enables modal confirmation button |
| Disclaimer | Acknowledge & Continue | `acknowledge-disclaimer-btn` | Button | Persists disclaimer acknowledgment |

---

## 3. User Dashboard (`/`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Header | View All 20 Stocks button | `view-full-rankings-btn` | Link | Navigates to `/rankings` |
| Top Picks | Overall horizon tab | `tab-overall` | Button | Filters top picks to overall combined scores |
| Top Picks | Short term tab | `tab-short` | Button | Filters top picks to short-horizon scores |
| Top Picks | Mid term tab | `tab-mid` | Button | Filters top picks to mid-horizon scores |
| Top Picks | Long term tab | `tab-long` | Button | Filters top picks to long-horizon scores |
| Top Picks | Methodology link | `how-we-tested-this-link` | Link | Navigates to `/how-it-works` |
| Top Picks | Stock row | `top-pick-row-{code}` | Row | Navigates to `/stocks/{code}` |
| Consistent Leaders | Stock pill | `leader-pill-{code}` | Link | Navigates to `/stocks/{code}` |
| Watchlist Snapshot | Manage Watchlist link | `go-to-watchlist-link` | Link | Navigates to `/watchlist` |

---

## 4. Rankings Page (`/rankings`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Toolbar | Toggle horizon weight sliders | `toggle-weight-sliders-btn` | Button | Toggles weight adjustment sliders |
| Weights | Short horizon slider | `weight-slider-short` | Slider | Live adjusts short horizon weight |
| Weights | Mid horizon slider | `weight-slider-mid` | Slider | Live adjusts mid horizon weight |
| Weights | Long horizon slider | `weight-slider-long` | Slider | Live adjusts long horizon weight |
| Weights | Reset to default | `reset-weights-btn` | Button | Resets weights to 0.33 / 0.33 / 0.34 |
| Weights | Apply weights | `apply-weights-btn` | Button | Recomputes rankings via API |
| Toolbar | Column visibility menu | `column-visibility-btn` | Button | Opens column selection dropdown |
| Toolbar | Export CSV button | `export-csv-btn` | Button | Downloads CSV export of rankings |
| Toolbar | Open Compare drawer | `open-compare-drawer-btn` | Button | Opens multi-stock comparison overlay |
| Compare Drawer | Close compare drawer | `close-compare-drawer-btn` | Button | Closes comparison drawer |
| Filter Bar | Search input | `rankings-search-input` | Input | Filters table by stock code or company name |
| Filter Bar | Filter tier ALL | `filter-tier-all` | Button | Clears tier filter |
| Filter Bar | Filter tier High | `filter-tier-high` | Button | Filters table to High upside stocks |
| Filter Bar | Filter tier Neutral | `filter-tier-neutral` | Button | Filters table to Neutral upside stocks |
| Filter Bar | Filter tier Low | `filter-tier-low` | Button | Filters table to Low upside stocks |
| Filter Bar | Filter risk ALL | `filter-risk-all` | Button | Clears risk filter |
| Filter Bar | Filter risk Normal | `filter-risk-normal` | Button | Shows Normal risk stocks only |
| Filter Bar | Filter risk Elevated | `filter-risk-elevated` | Button | Shows Elevated risk stocks only |
| Filter Bar | Exclude thinly traded | `filter-exclude-liquidity-checkbox` | Checkbox | Toggles exclusion of liquidity-warned stocks |
| Table Headers | Sort by Rank | `sort-rank` | Header | Toggles sort direction by rank |
| Table Headers | Sort by Code | `sort-code` | Header | Toggles sort direction by stock code |
| Table Headers | Sort by Overall | `sort-overall` | Header | Toggles sort direction by overall score |
| Table Headers | Sort by Short | `sort-short` | Header | Toggles sort direction by short score |
| Table Headers | Sort by Mid | `sort-mid` | Header | Toggles sort direction by mid score |
| Table Headers | Sort by Long | `sort-long` | Header | Toggles sort direction by long score |
| Table Headers | Sort by Risk | `sort-risk` | Header | Toggles sort direction by risk category |
| Table Headers | Sort by Price | `sort-price` | Header | Toggles sort direction by price |
| Table Headers | Sort by 1D Change | `sort-change` | Header | Toggles sort direction by daily change |

---

## 5. Stock Detail Page (`/stocks/:code`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Header | Watchlist star button | `toggle-watchlist-btn` | Button | Toggles stock presence in watchlist |
| Header | Create Alert button | `create-alert-btn` | Button | Opens alert creation modal |
| Chart | Candlestick mode | `chart-type-candlestick` | Button | Renders candlestick series |
| Chart | Line mode | `chart-type-line` | Button | Renders line price series |
| Chart | Range 1M | `range-1m` | Button | Zooms chart to 1-month range |
| Chart | Range 3M | `range-3m` | Button | Zooms chart to 3-month range |
| Chart | Range 6M | `range-6m` | Button | Zooms chart to 6-month range |
| Chart | Range 1Y | `range-1y` | Button | Zooms chart to 1-year range |
| Chart | Range 5Y | `range-5y` | Button | Zooms chart to 5-year range |
| Chart | Range Max | `range-max` | Button | Shows full history |
| Chart | Toggle SMA 20 | `toggle-sma20` | Button | Toggles 20-period moving average |
| Chart | Toggle SMA 50 | `toggle-sma50` | Button | Toggles 50-period moving average |
| Chart | Toggle SMA 200 | `toggle-sma200` | Button | Toggles 200-period moving average |
| Chart | Toggle RSI | `toggle-rsi` | Button | Toggles 14-period RSI indicator |
| Chart | Toggle MACD | `toggle-macd` | Button | Toggles MACD oscillator |
| Alert Modal | Alert trigger condition | `alert-type-select` | Select | Sets trigger rule type |
| Alert Modal | Threshold value input | `alert-threshold-input` | Input | Sets numeric threshold |
| Alert Modal | Save alert button | `submit-alert-btn` | Button | Saves alert rule to database |

---

## 6. Watchlist Page (`/watchlist`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Toolbar | Add stock select | `add-stock-select` | Select | Chooses equity to watch |
| Toolbar | Add button | `add-stock-btn` | Button | Adds selected equity to watchlist |
| Table | Remove button | `remove-watchlist-btn-{code}` | Button | Deletes stock from watchlist |
| Table | Note input | `edit-note-input-{code}` | Input | Edits custom investment notes |
| Table | Save note button | `save-note-btn-{code}` | Button | Commits updated note |
| Table | Quick alert button | `quick-alert-btn-{code}` | Button | Navigates to stock alert config |

---

## 7. Alerts & Notifications Pages (`/alerts`, `/notifications`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Alerts | New Alert button | `create-new-alert-btn` | Button | Opens rule creation modal |
| Alerts | Active/Pause toggle | `toggle-alert-status-{id}` | Button | Toggles rule active state |
| Alerts | Edit alert button | `edit-alert-btn-{id}` | Button | Opens modal in edit mode |
| Alerts | Delete alert button | `delete-alert-btn-{id}` | Button | Removes alert rule |
| Alerts Modal | Save rule button | `save-alert-rule-btn` | Button | Submits rule creation/update |
| Notifications | Mark All as Read | `mark-all-read-btn` | Button | Marks all notifications as read |
| Notifications | Mark Read | `mark-read-btn-{id}` | Button | Marks individual notification as read |

---

## 8. Account & Methodology Pages

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Account | Visual Theme toggle | `account-theme-toggle` | Button | Switches light/dark theme |
| Account | Current password input | `current-password-input` | Input | Captures existing password |
| Account | New password input | `new-password-input` | Input | Captures new password |
| Account | Confirm password input | `confirm-new-password-input` | Input | Confirms new password |
| Account | Update password submit | `change-password-submit-btn` | Button | Changes account password |
| Account | Sign out button | `account-signout-btn` | Button | Logs user out |
| How It Works | Print document button | `print-methodology-btn` | Button | Triggers browser print stylesheet |

---

## 9. Admin Console (`/admin`)

| Section | Element Description | `data-testid` | Type | Expected Effect |
| :--- | :--- | :--- | :--- | :--- |
| Navigation | Overview tab | `admin-nav-overview` | Link | Navigates to `/admin` |
| Navigation | Models tab | `admin-nav-model-governance` | Link | Navigates to `/admin/models` |
| Navigation | Users tab | `admin-nav-user-management` | Link | Navigates to `/admin/users` |
| Navigation | Data tab | `admin-nav-data-management` | Link | Navigates to `/admin/data` |
| Navigation | Audit tab | `admin-nav-audit-log` | Link | Navigates to `/admin/audit` |
| Navigation | Back to App | `admin-back-to-app-btn` | Link | Returns to user dashboard `/` |
| Models | Short horizon tab | `admin-tab-short` | Button | Views short horizon model metrics |
| Models | Mid horizon tab | `admin-tab-mid` | Button | Views mid horizon model metrics |
| Models | Long horizon tab | `admin-tab-long` | Button | Views long horizon model metrics |
| Models | Reset All to Auto | `reset-all-auto-btn` | Button | Resets all horizons to Auto model |
| Models | Use Model button | `activate-model-btn-{model_id}` | Button | Opens switch confirmation modal |
| Switch Modal | Warning checkbox | `acknowledge-warning-checkbox` | Checkbox | Acknowledges lower-evidence override |
| Switch Modal | Reason textarea | `switch-reason-textarea` | Textarea | Captures mandatory justification |
| Switch Modal | Confirm switch | `confirm-model-switch-btn` | Button | Commits model switch to database |
| Users | Create user button | `open-create-user-modal-btn` | Button | Opens user creation modal |
| Users | Change role dropdown | `select-role-{id}` | Select | Changes role (user <-> admin) |
| Users | Toggle status button | `toggle-status-btn-{id}` | Button | Enables / disables user account |
| Users | Reset password button | `reset-pwd-btn-{id}` | Button | Generates one-time temporary password |
| Users | Copy temporary password | `copy-temp-password-btn` | Button | Copies temporary password to clipboard |
| Data | Evaluate Alerts Now | `evaluate-alerts-now-btn` | Button | Triggers server alert check |
| Data | Dry-run validation toggle | `dry-run-checkbox` | Checkbox | Toggles dry-run validation mode |
| Data | Submit upload | `submit-upload-btn` | Button | Submits synthetic data payload |
| Data | Rollback button | `rollback-btn-{ver}` | Button | Opens rollback confirmation modal |
| Data | Confirm rollback | `confirm-rollback-btn` | Button | Reverts serving snapshot version |
| Audit | Export CSV button | `export-audit-csv-btn` | Button | Exports audit logs as CSV |

# Stock Pulse API Documentation

This document describes the REST API endpoints, schemas, authentication flows, and error handling for frontend integration with the Stock Pulse backend.

---

## 1. General Architecture & Base URL

- **Base URL**: `/api/v1`
- **Specification**: OpenAPI 3.1.0 document is available at `app/backend/openapi.json`.
- **Environment Modes**:
  - `ENV=dev`: Swagger UI enabled at `/docs`, Redoc enabled at `/redoc`.
  - `ENV=prod`: Interactive documentation disabled.
- **Request Tracking**: Every request generates or propagates an `X-Request-ID` header.
- **Audience Separation**: Public and user-facing endpoints NEVER expose internal model names (e.g. `momentum_126`, `ridge`, `lgbm`, `ens_rank4`) or ML metrics (e.g. `IC`, `SHAP`, `null band`). These are strictly quarantined to `/api/v1/admin/*`.

---

## 2. Authentication & Security Flow

Authentication uses secure HTTP-only cookies combined with a double-submit CSRF protection mechanism.

### Cookie Storage:
- `access_token`: Short-lived JWT (default 15 minutes), `httpOnly=True`, `SameSite=Lax`, `Secure` in production.
- `refresh_token`: Long-lived token (default 7 days), `httpOnly=True`, `SameSite=Lax`, `Secure` in production.
- `csrf_token`: Readable CSRF token cookie, paired with client-submitted `X-CSRF-Token` header.

### Double-Submit CSRF Protection:
For all state-changing HTTP methods (`POST`, `PUT`, `PATCH`, `DELETE`) on protected endpoints:
1. The frontend reads the `csrf_token` cookie value (or uses the value returned in the login/me response).
2. The frontend passes this token in the header:
   ```http
   X-CSRF-Token: <csrf_token_value>
   ```

### Password Policy & Lockout:
- Minimum length: 10 characters.
- Must not equal the user's email address.
- Must not be in common password blacklist.
- Passwords are hashed with Argon2id.
- Account lockout: 10 consecutive failed login attempts locks the account temporarily for 15 minutes.
- Rate limiting: 5 login requests per minute per IP.

### Token Rotation & Family Revocation:
- Refresh tokens are hashed in the database.
- Refreshing an access token rotates the refresh token.
- If an already-rotated refresh token is reused (indicating potential theft), the entire token family is immediately revoked.

---

## 3. Standard Error Format

All error responses across the API follow a uniform JSON structure:

```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid credentials",
    "request_id": "req-988451f2-53b4-4e9e-8390-349f7ba3ad57"
  }
}
```

Common Error Codes:
- `NOT_FOUND`: Resource does not exist or unauthorized to view (404).
- `AUTHENTICATION_REQUIRED`: Missing or invalid session (401).
- `INSUFFICIENT_PERMISSIONS`: User lacks required role (403).
- `CSRF_VERIFICATION_FAILED`: Invalid or missing `X-CSRF-Token` header (403).
- `ACCOUNT_LOCKED`: Temporarily locked due to failed attempts (423).
- `RATE_LIMIT_EXCEEDED`: Too many requests (429).
- `VALIDATION_ERROR`: Malformed input payload (422).

---

## 4. Endpoints Reference

### 4.1 Authentication (`/auth`)

### 4.0 Public Information (`/public`, `/health`)

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/health` | Public | System health status. |
| `GET` | `/public/info` | Public | Public metadata: platform name, data as-of date, and legal disclaimer text. |

### 4.1 Authentication (`/auth`)

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `POST` | `/auth/register` | Public | Register new user. Always assigns role `user`. |
| `POST` | `/auth/login` | Public | Authenticate user; sets `access_token`, `refresh_token`, `csrf_token` cookies. |
| `POST` | `/auth/refresh` | Public | Rotates refresh token and issues new access token. |
| `POST` | `/auth/logout` | Public | Revokes current session and clears auth cookies. |
| `GET` | `/auth/me` | Authenticated User | Retrieve current user profile and role. |
| `POST` | `/auth/change-password` | Authenticated User | Change user password (requires current password). |

### 4.2 Market Overview (`/market`)

*Requires Authenticated User session (`401 Unauthorized` if unauthenticated).*

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/market/summary` | Authenticated User | Current market summary, as-of date, data version, tier counts, conditional volatility note, and disclaimer. |

### 4.3 Stocks & Pricing (`/stocks`)

*Requires Authenticated User session (`401 Unauthorized` if unauthenticated).*

> **Stock Identifier Standard**: Across all endpoints, the canonical stock identifier field `code` is strictly the Colombo Stock Exchange series ID (e.g., `"COMB.N"`, `"SEYB.X"`, `"JKH.N"`, `"DIAL.N"`). Never append suffixes or Bloomberg codes (e.g., `"COMB.N0000"` is invalid).

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/stocks` | Authenticated User | Paginated list of stocks with filtering by tier, risk status, search query. |
| `GET` | `/stocks/{code}` | Authenticated User | Stock detail: horizon percentiles, tiers, confidence labels, supporting & holding back drivers with consumer phrases, thin-trading note, 52-week price range. |
| `GET` | `/stocks/{code}/prices` | Authenticated User | Historical clean OHLCV with indicators (`SMA20/50/200`, `RSI14`, `MACD`, `Bollinger Bands`). Query param `range`: `1m`, `3m`, `6m`, `1y`, `5y`, `max`. |

### 4.4 Rankings & Insights (`/rankings`, `/leaders`, `/movers`)

*Requires Authenticated User session (`401 Unauthorized` if unauthenticated).*

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/rankings` | Authenticated User | Multi-horizon rankings (`short`, `mid`, `long`, `overall`). Supports custom weights query `weights=short:0.5,mid:0.3,long:0.2`. |
| `GET` | `/leaders` | Authenticated User | Consistent Leaders: stocks meeting multi-horizon threshold criteria. |
| `GET` | `/movers` | Authenticated User | Rank changes over the past 21 market days (labeled "indicative"). |
| `GET` | `/rankings/export.csv` | Authenticated User | Download CSV export of current ranked universe (clean user columns only). |

### 4.5 Methodology (`/methodology`)

*Requires Authenticated User session (`401 Unauthorized` if unauthenticated).*

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/methodology` | Authenticated User | Returns plain English user methodology documentation and threshold definitions. |

### 4.6 User Watchlist (`/watchlist`)

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/watchlist` | Authenticated User | List current user's tracked stocks with latest ranking data. |
| `POST` | `/watchlist` | Authenticated User | Add a stock to user's watchlist with optional notes (`series_id` = `"COMB.N"`). |
| `DELETE` | `/watchlist/{item_id}`| Authenticated User | Remove a stock from user's watchlist. |

### 4.7 Alerts & Notifications (`/alerts`, `/notifications`)

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/alerts` | Authenticated User | Retrieve all active/inactive alert configurations for user. |
| `POST` | `/alerts` | Authenticated User | Create an alert (`price_above`, `price_below`, `daily_change_above`, `daily_change_below`, `enters_high_tier`, `becomes_consistent_leader`, `risk_changes`). |
| `DELETE` | `/alerts/{id}` | Authenticated User | Delete an alert (object-level authorization enforced). |
| `GET` | `/notifications` | Authenticated User | List triggered alerts and notifications for user. |
| `POST` | `/notifications/{id}/read`| Authenticated User | Mark notification as read. |
| `POST` | `/notifications/read-all` | Authenticated User | Mark all unread notifications as read. |

---

## 5. Admin Endpoints (`/admin`)

*Strictly restricted to users with `role: "admin"`. Unauthorized requests receive `403 Forbidden`.*

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/admin/models` | Admin | Complete model registry with dev IC, top-tier hit rate, fold t-statistic, Holm p-values, holdout IC, production-config comparison, per_fold_ic, and servable flags. |
| `GET` | `/admin/models/{model_id}/preview` | Admin | Preview top 5 stocks under candidate model vs currently active model and return changed stock count (`changed_count`). Query param: `horizon=short\|mid\|long`. |
| `GET` | `/admin/model-settings` | Admin | Inspect active model selection per horizon (`auto` or manual override). |
| `PUT` | `/admin/model-settings` | Admin | Update model assignment. Requires `confirm=true` and `reason`. If model has lower evidence or t-stat than Auto, `acknowledge_warning=true` is required. |
| `POST` | `/admin/model-settings/reset-auto` | Admin | Reset all horizons back to default Auto models. |
| `GET` | `/admin/users` | Admin | List registered users and their roles/active states. |
| `POST` | `/admin/users` | Admin | Manually provision a user or administrator. |
| `PATCH` | `/admin/users/{id}` | Admin | Update user active status or role (last-admin protection enforced). |
| `POST` | `/admin/users/{id}/reset-password` | Admin | Issue temporary one-time password and force reset. |
| `GET` | `/admin/data/status` | Admin | Current data versions, as-of dates, and refresh history. |
| `POST` | `/admin/data/upload` | Admin | Upload new market data (supports `append` or `replace` mode with `dry_run` validation). |
| `GET` | `/admin/jobs/{id}` | Admin | Query background pipeline job status. |
| `POST` | `/admin/alerts/evaluate` | Admin | Trigger ad-hoc edge-triggered alert evaluation across all active user alerts. |
| `GET` | `/admin/audit-log` | Admin | Query tamper-evident audit logs with actor and action filters. |
| `GET` | `/admin/methodology` | Admin | View technical methodology documentation for platform administrators. |
| `GET` | `/admin/health` | Admin | Deep system health check (database connection, serving data readiness). |

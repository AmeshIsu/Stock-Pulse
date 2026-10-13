# Stock Pulse — Deployment Guide

This guide details the production deployment architecture for the **Stock Pulse** application.

---

## 1. Architecture Overview

```
   [ Browser User ]
          |
          v
   +---------------------------------------+
   | Vercel Edge Network                   |
   | - Serves static React/Vite SPA bundle |
   | - Rewrites /api/* to Container Host   |
   | - Same-origin cookies & CSRF preserved|
   +---------------------------------------+
          |
          | HTTPS (Reverse Proxy / Rewrite)
          v
   +---------------------------------------+
   | Container Host (Render / Fly.io / AWS)|
   | - FastAPI Uvicorn service             |
   | - Persistent Volume mounted at /data  |
   |   (data/serving, data/app, registry)  |
   +---------------------------------------+
          |
          v
   +---------------------------------------+
   | Managed PostgreSQL (RDS / Supabase)   |
   | - User accounts, sessions, alerts,    |
   |   watchlists, notifications, audit log|
   +---------------------------------------+
```

### Key Principles
1. **Same-Origin Authentication**: The Vercel frontend proxies requests matching `/api/:path*` directly to the backend container. Because the browser communicates only with the Vercel domain, session cookies (`stockpulse_access`, `stockpulse_refresh`) and the CSRF token cookie (`stockpulse_csrf`) are treated as first-party same-origin cookies without cross-site cookie restrictions (`SameSite=Lax`).
2. **Persistent Storage for Serving Models & Registry**: The backend container host must mount a persistent volume at `/app/data` to ensure `data/serving/`, `data/app/`, and `model_registry.json` are retained across container deployments.
3. **Database Migrations**: Alembic migrations run automatically on container startup or via release hooks before accepting traffic.

---

## 2. Frontend Deployment (Vercel)

1. Connect the repository to Vercel.
2. Set Root Directory to `app/frontend` (or use the root `vercel.json`).
3. Build Settings:
   - Framework Preset: `Vite`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Environment Variables:
   - `VITE_API_BASE_URL`: `/api/v1`
5. Vercel Configuration (`vercel.json`):
   ```json
   {
     "rewrites": [
       {
         "source": "/api/:path*",
         "destination": "https://<your-backend-container-domain>/api/:path*"
       },
       {
         "source": "/(.*)",
         "destination": "/index.html"
       }
     ]
   }
   ```

---

## 3. Backend Deployment (Container Host with Persistent Disk)

### Container Configuration
- Dockerfile builds the FastAPI application with Python 3.11+.
- Mount volume:
  - Container path: `/app/data` (persistent SSD volume).
- Port: `8000`

### Required Environment Variables
| Variable | Description | Example |
| :--- | :--- | :--- |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@db-host:5432/stockpulse` |
| `SECRET_KEY` | JWT and cryptographic token secret | `64-char random hex string` |
| `ENVIRONMENT` | Environment mode | `production` |
| `CORS_ORIGINS` | Allowed origins (Vercel domain) | `https://stockpulse.vercel.app` |
| `COOKIE_SECURE` | Set `True` for HTTPS-only cookies | `True` |

### Entrypoint Script
```bash
#!/bin/bash
set -e

# Run database migrations
python -m alembic upgrade head

# Ensure serving defaults and data structures exist
python scripts/ensure_serving_data.py

# Start Uvicorn
exec uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 4. Operational Health & Monitoring

- **Health Check**: `GET /health` returns `{ "status": "ok" }`.
- **Public Metadata**: `GET /public/info` returns app name, data as-of date, and legal disclaimer without requiring authentication.
- **Admin Audit Trail**: Admin model changes, user management actions, and dataset updates are logged persistently into the `audit_logs` table.

"""app/backend/main.py
Stock Pulse FastAPI application entrypoint.
Configures CORS, security headers, standardized error handlers,
SlowAPI rate limiting, database tables initialization / first admin creation,
and registers all route modules under /api/v1.
"""
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from app.backend.core.config import settings
from app.backend.core.rate_limit import limiter
from app.backend.db.session import engine, SessionLocal
from app.backend.db.base import Base
from app.backend.models.entities import User
from app.backend.core.security import get_password_hash

# Routers
from app.backend.routers import (
    public,
    auth,
    market,
    stocks,
    rankings,
    methodology,
    watchlist,
    alerts,
    notifications,
    admin
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and create first admin if not present
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    try:
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count == 0:
            first_admin = User(
                email=settings.ADMIN_EMAIL.lower().strip(),
                hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
                role="admin",
                is_active=True
            )
            db.add(first_admin)
            db.commit()
            print(f"[Startup] Initialized primary admin: {settings.ADMIN_EMAIL}")
    finally:
        db.close()
    yield
    # Shutdown


# In production, interactive docs are disabled
app = FastAPI(
    title="Stock Pulse API",
    description="Colombo Stock Exchange Institutional Upside-Ranking & Intelligence Platform",
    version="1.0.0",
    docs_url=None if settings.is_prod else "/docs",
    redoc_url=None if settings.is_prod else "/redoc",
    openapi_url=None if settings.is_prod else "/openapi.json",
    lifespan=lifespan
)

# SlowAPI rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-CSRF-Token"]
)


# Request ID and Security Headers Middleware
@app.middleware("http")
async def security_and_request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id

    # Check upload body size limit (MAX_UPLOAD_MB)
    content_length = request.headers.get("content-length")
    if content_length:
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if int(content_length) > max_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": f"Upload size exceeds maximum allowed limit of {settings.MAX_UPLOAD_MB}MB.",
                        "request_id": req_id
                    }
                },
                headers={"X-Request-ID": req_id}
            )

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Global Exception Handlers ensuring uniform JSON shape: {error: {code, message, request_id}}
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR"
    }
    code = code_map.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": code,
                "message": exc.detail if isinstance(exc.detail, str) else "Request error",
                "request_id": req_id
            }
        },
        headers={"X-Request-ID": req_id}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters or payload format.",
                "request_id": req_id
            }
        },
        headers={"X-Request-ID": req_id}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred.",
                "request_id": req_id
            }
        },
        headers={"X-Request-ID": req_id}
    )


# Register API v1 routes
API_PREFIX = "/api/v1"
app.include_router(public.router, prefix=API_PREFIX)
app.include_router(public.router)  # Also allow direct /public/info
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(market.router, prefix=API_PREFIX)
app.include_router(stocks.router, prefix=API_PREFIX)
app.include_router(rankings.router, prefix=API_PREFIX)
app.include_router(methodology.router, prefix=API_PREFIX)
app.include_router(watchlist.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(notifications.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)


@app.get("/health", tags=["system"])
def public_health():
    return {"status": "ok", "service": "Stock Pulse API"}

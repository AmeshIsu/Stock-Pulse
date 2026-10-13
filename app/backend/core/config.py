"""app/backend/core/config.py
Configuration loaded solely from environment variables.
"""
import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    ENV: str = os.getenv("ENV", "dev")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev_secret_key_change_in_production_min_32_chars_long_123456")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stock_pulse.db")
    CORS_ORIGINS_RAW: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    )
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@stockpulse.lk")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "AdminPassword123!")
    ACCESS_TOKEN_MINUTES: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "15"))
    REFRESH_TOKEN_DAYS: int = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

    @property
    def is_prod(self) -> bool:
        return self.ENV.lower() == "prod"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]


settings = Settings()

"""app/backend/models/entities.py
SQLAlchemy 2 models for Stock Pulse platform.
Includes: User, RefreshToken, WatchlistItem, Alert, Notification,
ModelSetting, AuditLog, and DataJob.
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, JSON
)
from sqlalchemy.orm import relationship

from app.backend.db.base import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="user", nullable=False)  # "admin" | "user"
    is_active = Column(Boolean, default=True, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, index=True, nullable=False)
    family_id = Column(String(64), index=True, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    series_id = Column(String(32), nullable=False, index=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="watchlist_items")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    series_id = Column(String(32), nullable=False, index=True)
    type = Column(String(64), nullable=False)
    params = Column(JSON, nullable=True)
    horizon = Column(String(16), nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    cooldown = Column(Integer, default=1, nullable=False)  # in data refreshes or days
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="alerts")
    notifications = relationship("Notification", back_populates="alert", cascade="all, delete-orphan")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id", ondelete="SET NULL"), nullable=True, index=True)
    message = Column(Text, nullable=False)
    data_version = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    read_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="notifications")
    alert = relationship("Alert", back_populates="notifications")


class ModelSetting(Base):
    __tablename__ = "model_settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    horizon = Column(String(16), unique=True, index=True, nullable=False)  # "short", "mid", "long"
    mode = Column(String(16), default="auto", nullable=False)  # "auto" | "manual"
    model_id = Column(String(64), nullable=True)
    updated_by = Column(String(255), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    actor_id = Column(Integer, nullable=True)
    actor_email = Column(String(255), nullable=False, index=True)
    action = Column(String(64), nullable=False, index=True)
    target = Column(String(255), nullable=True)
    details = Column(JSON, nullable=True)
    ip = Column(String(64), nullable=True)


class DataJob(Base):
    __tablename__ = "data_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type = Column(String(64), nullable=False)  # "refresh", "rollback", "recompute"
    status = Column(String(32), default="pending", nullable=False)  # "pending", "running", "completed", "failed"
    started = Column(DateTime(timezone=True), nullable=True)
    finished = Column(DateTime(timezone=True), nullable=True)
    input_name = Column(String(255), nullable=True)
    result = Column(JSON, nullable=True)
    created_by = Column(String(255), nullable=True)

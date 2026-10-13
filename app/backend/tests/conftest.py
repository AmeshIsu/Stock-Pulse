import pytest
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.backend.main import app

from app.backend.db.base import Base
from app.backend.db.session import get_db
from app.backend.models.entities import User, WatchlistItem, Alert, Notification, ModelSetting, AuditLog
from app.backend.core.security import get_password_hash, create_access_token, generate_csrf_token

# In-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_user(db_session) -> User:
    user = db_session.query(User).filter(User.email == "admin@example.com").first()
    if not user:
        user = User(
            email="admin@example.com",
            hashed_password=get_password_hash("AdminPass123!"),
            role="admin",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture
def normal_user_1(db_session) -> User:
    user = db_session.query(User).filter(User.email == "user1@example.com").first()
    if not user:
        user = User(
            email="user1@example.com",
            hashed_password=get_password_hash("UserOnePass123!"),
            role="user",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture
def normal_user_2(db_session) -> User:
    user = db_session.query(User).filter(User.email == "user2@example.com").first()
    if not user:
        user = User(
            email="user2@example.com",
            hashed_password=get_password_hash("UserTwoPass123!"),
            role="user",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


def get_auth_headers_and_cookies(user: User):
    token = create_access_token({"sub": str(user.id), "role": user.role})
    csrf = generate_csrf_token()
    cookies = {
        "access_token": token,
        "csrf_token": csrf
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-CSRF-Token": csrf
    }
    return headers, cookies

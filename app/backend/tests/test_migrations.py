"""app/backend/tests/test_migrations.py
Alembic migration lifecycle and database schema tests.
Verifies:
- Alembic upgrade to head applies all migrations.
- Alembic downgrade to base removes all created tables cleanly.
- Alembic upgrade back to head succeeds idempotently.
"""
import pytest
from pathlib import Path
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.backend.db.session import engine


def test_alembic_migrations_up_down_cycle():
    """Verify clean upgrade and downgrade cycle through Alembic."""
    alembic_cfg_path = Path(__file__).resolve().parents[3] / "alembic.ini"
    assert alembic_cfg_path.exists(), f"Missing alembic.ini at {alembic_cfg_path}"

    alembic_cfg = Config(str(alembic_cfg_path))

    from sqlalchemy import text
    # Cleanly downgrade or drop all tables including alembic_version for true greenfield test
    try:
        command.downgrade(alembic_cfg, "base")
    except Exception:
        pass
    from app.backend.db.base import Base
    Base.metadata.drop_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    # 1. Upgrade to head
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    assert "users" in existing_tables

    # 2. Downgrade to base
    command.downgrade(alembic_cfg, "base")

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    # Users table should be gone after downgrade to base
    assert "users" not in existing_tables

    # 3. Upgrade back to head
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(engine)
    tables_after = inspector.get_table_names()
    expected_tables = {
        "users", "refresh_tokens", "watchlist_items", "alerts",
        "notifications", "model_settings", "audit_log", "data_jobs"
    }
    for t in expected_tables:
        assert t in tables_after, f"Expected table '{t}' missing after migration upgrade!"


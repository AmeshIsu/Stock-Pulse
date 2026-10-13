"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-12 21:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False, server_default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_id', 'users', ['id'])

    # 2. refresh_tokens
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('family_id', sa.String(length=64), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False)
    )
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)
    op.create_index('ix_refresh_tokens_family_id', 'refresh_tokens', ['family_id'])
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])

    # 3. watchlist_items
    op.create_table(
        'watchlist_items',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('series_id', sa.String(length=32), nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
    )
    op.create_index('ix_watchlist_items_user_id', 'watchlist_items', ['user_id'])
    op.create_index('ix_watchlist_items_series_id', 'watchlist_items', ['series_id'])

    # 4. alerts
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('series_id', sa.String(length=32), nullable=False),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('params', sa.JSON(), nullable=True),
        sa.Column('horizon', sa.String(length=16), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('cooldown', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
    )
    op.create_index('ix_alerts_user_id', 'alerts', ['user_id'])
    op.create_index('ix_alerts_series_id', 'alerts', ['series_id'])

    # 5. notifications
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('alert_id', sa.Integer(), sa.ForeignKey('alerts.id', ondelete='SET NULL'), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('data_version', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'])
    op.create_index('ix_notifications_alert_id', 'notifications', ['alert_id'])

    # 6. model_settings
    op.create_table(
        'model_settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('horizon', sa.String(length=16), unique=True, nullable=False),
        sa.Column('mode', sa.String(length=16), nullable=False, server_default='auto'),
        sa.Column('model_id', sa.String(length=64), nullable=True),
        sa.Column('updated_by', sa.String(length=255), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
    )
    op.create_index('ix_model_settings_horizon', 'model_settings', ['horizon'])

    # 7. audit_log
    op.create_table(
        'audit_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('actor_id', sa.Integer(), nullable=True),
        sa.Column('actor_email', sa.String(length=255), nullable=False),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('target', sa.String(length=255), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('ip', sa.String(length=64), nullable=True)
    )
    op.create_index('ix_audit_log_ts', 'audit_log', ['ts'])
    op.create_index('ix_audit_log_action', 'audit_log', ['action'])

    # 8. data_jobs
    op.create_table(
        'data_jobs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column('type', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('started', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished', sa.DateTime(timezone=True), nullable=True),
        sa.Column('input_name', sa.String(length=255), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=True)
    )

def downgrade() -> None:
    op.drop_table('data_jobs')
    op.drop_table('audit_log')
    op.drop_table('model_settings')
    op.drop_table('notifications')
    op.drop_table('alerts')
    op.drop_table('watchlist_items')
    op.drop_table('refresh_tokens')
    op.drop_table('users')

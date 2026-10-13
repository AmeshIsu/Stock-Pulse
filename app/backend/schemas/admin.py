"""app/backend/schemas/admin.py
Admin request and response schemas (admin-only endpoints).
Includes full model registry, model settings, user management,
data refresh jobs, and audit logs.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class HorizonMetric(BaseModel):
    dev_mean_ic: Optional[float] = None
    hit_rate: Optional[float] = None
    hit_rate_ci: Optional[List[float]] = None
    folds_positive_ratio: Optional[str] = None
    folds_positive_count: Optional[int] = None
    folds_total_count: Optional[int] = None
    fold_t_stat: Optional[float] = None
    raw_p_value: Optional[float] = None
    holm_adjusted_p_value: Optional[float] = None
    evidence_level: Optional[str] = None
    net_excess: Optional[float] = None
    holdout_ic: Optional[float] = None
    holdout_n_eff: Optional[int] = None
    holdout_status: Optional[str] = None
    production_config_metrics: Optional[Dict[str, Any]] = None
    per_fold_ic: Optional[Dict[str, float]] = None
    is_auto_default: bool = False
    is_currently_active: bool = False


class ModelPreviewResponse(BaseModel):
    model_id: str
    horizon: str
    active_model_id: str
    top_5_model: List[str]
    top_5_active: List[str]
    changed_count: int


class AdminModelRegistryItem(BaseModel):
    model_id: str
    admin_display_name: str
    family: str
    short_description: str
    servable: bool
    servable_reason: Optional[str] = None
    horizons: Dict[str, HorizonMetric]


class AdminModelRegistryResponse(BaseModel):
    active_models: Dict[str, str]
    models: List[AdminModelRegistryItem]


class ModelSettingItem(BaseModel):
    horizon: str
    mode: str  # "auto" | "manual"
    model_id: Optional[str] = None
    effective_model_id: str
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None


class ModelSettingsResponse(BaseModel):
    settings: List[ModelSettingItem]


class ModelSettingUpdateRequest(BaseModel):
    horizon: str
    mode: str  # "auto" | "manual"
    model_id: Optional[str] = None
    confirm: bool = False
    reason: str
    acknowledge_warning: bool = False


class AdminUserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=10)
    role: str = "user"  # "admin" | "user"
    is_active: bool = True


class AdminUserUpdate(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None


class AdminUserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    failed_login_attempts: int
    locked_until: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PasswordResetResponse(BaseModel):
    user_id: int
    temporary_password: str
    message: str


class DataStatusResponse(BaseModel):
    current_as_of: str
    data_version: str
    available_versions: List[str]
    last_refresh_summary: Optional[Dict[str, Any]] = None


class DataJobResponse(BaseModel):
    id: int
    type: str
    status: str
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
    input_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None

    class Config:
        from_attributes = True


class AuditLogItem(BaseModel):
    id: int
    ts: datetime
    actor_id: Optional[int] = None
    actor_email: str
    action: str
    target: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip: Optional[str] = None

    class Config:
        from_attributes = True

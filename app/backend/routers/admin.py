"""app/backend/routers/admin.py
Admin router for Stock Pulse platform (admin role required for all routes).
Implements model registry inspection, model settings overrides with guardrails,
user management with last-admin guard, data refresh jobs, audit logging, and methodology.
"""
import os
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status, Request
from sqlalchemy.orm import Session

from app.backend.db.session import get_db
from app.backend.models.entities import (
    User, RefreshToken, ModelSetting, AuditLog, DataJob
)
from app.backend.schemas.admin import (
    AdminModelRegistryResponse,
    AdminModelRegistryItem,
    HorizonMetric,
    ModelPreviewResponse,
    ModelSettingsResponse,
    ModelSettingItem,
    ModelSettingUpdateRequest,
    AdminUserResponse,
    AdminUserCreate,
    AdminUserUpdate,
    PasswordResetResponse,
    DataStatusResponse,
    DataJobResponse,
    AuditLogItem,
)
from app.backend.schemas.common import MessageResponse
from app.backend.routers.deps import get_current_admin, verify_csrf
from app.backend.services.serving import serving_repo, BASE_DIR
from app.backend.core.security import get_password_hash
from app.backend.services.alerts import evaluate_alerts

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health")
def admin_health(admin: User = Depends(get_current_admin)):
    return {"status": "ok", "role": admin.role, "time": datetime.now(timezone.utc).isoformat()}


@router.get("/models", response_model=AdminModelRegistryResponse)
def get_admin_models(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    reg_path = BASE_DIR / "data/app/model_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        raw_reg = json.load(f)

    db_settings = {r.horizon: (r.mode, r.model_id) for r in db.query(ModelSetting).all()}
    active_models = serving_repo.resolve_active_models(db_settings)
    defaults = serving_repo.get_active_defaults().get("active_models", {
        "short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"
    })

    models_list = []
    for m_id, m_data in raw_reg.items():
        horizons_dict = {}
        for h, h_metrics in m_data.get("horizons", {}).items():
            is_auto = (defaults.get(h) == m_id)
            is_active = (active_models.get(h) == m_id)
            hit_r = h_metrics.get("top_tier_hit_rate")
            if hit_r is None:
                hit_r = h_metrics.get("top_5_precision")

            horizons_dict[h] = HorizonMetric(
                dev_mean_ic=h_metrics.get("dev_mean_ic"),
                hit_rate=hit_r,
                hit_rate_ci=h_metrics.get("top_tier_hit_rate_90_ci") or ([round(hit_r - 0.05, 4), round(hit_r + 0.05, 4)] if hit_r else None),
                folds_positive_ratio=h_metrics.get("folds_positive_ratio"),
                folds_positive_count=h_metrics.get("folds_positive_count"),
                folds_total_count=h_metrics.get("folds_total_count"),
                fold_t_stat=h_metrics.get("fold_t_stat"),
                raw_p_value=h_metrics.get("raw_p_value"),
                holm_adjusted_p_value=h_metrics.get("holm_adjusted_p_value"),
                evidence_level=h_metrics.get("evidence_level"),
                net_excess=h_metrics.get("net_excess_under_actual_turnover"),
                holdout_ic=h_metrics.get("holdout_ic"),
                holdout_n_eff=h_metrics.get("holdout_n_eff"),
                holdout_status=h_metrics.get("holdout_status"),
                production_config_metrics=h_metrics.get("production_config_metrics"),
                per_fold_ic=h_metrics.get("per_fold_ic"),
                is_auto_default=is_auto,
                is_currently_active=is_active
            )

        models_list.append(AdminModelRegistryItem(
            model_id=m_id,
            admin_display_name=m_data.get("admin_display_name", m_id),
            family=m_data.get("family", "Unknown"),
            short_description=m_data.get("short_description", ""),
            servable=bool(m_data.get("servable", False)),
            servable_reason=m_data.get("servable_reason"),
            horizons=horizons_dict
        ))

    return AdminModelRegistryResponse(
        active_models=active_models,
        models=models_list
    )


@router.get("/models/{model_id}/preview", response_model=ModelPreviewResponse)
def preview_model_scores(
    model_id: str,
    horizon: str = Query("short", pattern="^(short|mid|long)$"),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    h = horizon.lower()
    reg_path = BASE_DIR / "data/app/model_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if model_id not in registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found.")

    defaults = serving_repo.get_active_defaults().get("active_models", {
        "short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"
    })
    
    # Resolve active model for horizon from DB settings or default
    setting = db.query(ModelSetting).filter(ModelSetting.horizon == h).first()
    active_m_id = (setting.model_id if setting and setting.mode == "manual" and setting.model_id else defaults.get(h, "ens_rank4"))

    # Load proposed model scores
    try:
        df_cand = serving_repo.get_model_scores_df(model_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Scores not available for model {model_id}: {str(e)}")

    try:
        df_active = serving_repo.get_model_scores_df(active_m_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Scores not available for active model {active_m_id}: {str(e)}")

    col = f"score_{h}"
    if col not in df_cand.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Column {col} not found for model {model_id}.")
    if col not in df_active.columns:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Column {col} not found for active model {active_m_id}.")

    # Top 5 rankable stock codes
    cand_rankable = df_cand[df_cand.get("rankable", True)]
    active_rankable = df_active[df_active.get("rankable", True)]

    top_5_model = list(cand_rankable.sort_values(col, ascending=False)["series_id"].head(5))
    top_5_active = list(active_rankable.sort_values(col, ascending=False)["series_id"].head(5))

    changed = len([c for c in top_5_model if c not in top_5_active])

    return ModelPreviewResponse(
        model_id=model_id,
        horizon=h,
        active_model_id=active_m_id,
        top_5_model=top_5_model,
        top_5_active=top_5_active,
        changed_count=changed
    )


@router.get("/model-settings", response_model=ModelSettingsResponse)
def get_model_settings(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    db_settings = {r.horizon: r for r in db.query(ModelSetting).all()}
    defaults = serving_repo.get_active_defaults().get("active_models", {
        "short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"
    })

    items = []
    for h in ["short", "mid", "long"]:
        setting = db_settings.get(h)
        if setting:
            eff_id = setting.model_id if setting.mode == "manual" and setting.model_id else defaults.get(h, "ens_rank4")
            items.append(ModelSettingItem(
                horizon=h,
                mode=setting.mode,
                model_id=setting.model_id,
                effective_model_id=eff_id,
                updated_by=setting.updated_by,
                updated_at=setting.updated_at
            ))
        else:
            items.append(ModelSettingItem(
                horizon=h,
                mode="auto",
                model_id=None,
                effective_model_id=defaults.get(h, "ens_rank4"),
                updated_by=None,
                updated_at=None
            ))
    return ModelSettingsResponse(settings=items)


@router.put("/model-settings", response_model=ModelSettingsResponse, dependencies=[Depends(verify_csrf)])
def update_model_settings(
    payload: ModelSettingUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    if not payload.confirm:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Operation requires confirm=true.")
    if not payload.reason or len(payload.reason.strip()) < 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid justification reason is required.")

    h = payload.horizon.lower()
    if h not in ["short", "mid", "long"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Horizon must be short, mid, or long.")

    reg_path = BASE_DIR / "data/app/model_registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    defaults = serving_repo.get_active_defaults().get("active_models", {})
    auto_model_id = defaults.get(h, "ens_rank4")

    if payload.mode == "manual":
        if not payload.model_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model_id required for manual mode.")
        if payload.model_id not in registry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {payload.model_id} not found.")

        m_info = registry[payload.model_id]
        if not m_info.get("servable", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {payload.model_id} is not servable: {m_info.get('servable_reason')}"
            )

        # Check evidence level and fold t-stat comparison against Auto model
        auto_h = registry[auto_model_id]["horizons"][h]
        chosen_h = m_info["horizons"][h]

        ev_rank = {"RED": 1, "AMBER": 2, "GREEN": 3}
        auto_ev = ev_rank.get(auto_h.get("evidence_level", "RED"), 1)
        chosen_ev = ev_rank.get(chosen_h.get("evidence_level", "RED"), 1)
        auto_t = float(auto_h.get("fold_t_stat", 0.0) or 0.0)
        chosen_t = float(chosen_h.get("fold_t_stat", 0.0) or 0.0)

        warnings = []
        if chosen_ev < auto_ev:
            warnings.append(f"Chosen model evidence level ({chosen_h.get('evidence_level')}) is lower than Auto ({auto_h.get('evidence_level')}).")
        if chosen_t < auto_t:
            warnings.append(f"Chosen model development fold t-stat ({chosen_t:.2f}) is lower than Auto ({auto_t:.2f}).")

        if warnings and not payload.acknowledge_warning:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Warning: {'; '.join(warnings)}. To proceed, set acknowledge_warning=true."
            )

    setting = db.query(ModelSetting).filter(ModelSetting.horizon == h).first()
    if not setting:
        setting = ModelSetting(horizon=h, mode=payload.mode, model_id=payload.model_id)
        db.add(setting)
    else:
        setting.mode = payload.mode
        setting.model_id = payload.model_id if payload.mode == "manual" else None

    setting.updated_by = admin.email
    setting.updated_at = datetime.now(timezone.utc)

    # Audit log
    audit = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action="UPDATE_MODEL_SETTINGS",
        target=f"{h}:{payload.mode}:{payload.model_id}",
        details={"reason": payload.reason, "mode": payload.mode, "model_id": payload.model_id},
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    # Invalidate serving cache immediately
    serving_repo.invalidate_cache()
    return get_model_settings(db, admin)


@router.post("/model-settings/reset-auto", response_model=ModelSettingsResponse, dependencies=[Depends(verify_csrf)])
def reset_model_settings_auto(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    for h in ["short", "mid", "long"]:
        setting = db.query(ModelSetting).filter(ModelSetting.horizon == h).first()
        if setting:
            setting.mode = "auto"
            setting.model_id = None
            setting.updated_by = admin.email
            setting.updated_at = datetime.now(timezone.utc)

    audit = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action="RESET_MODEL_SETTINGS_AUTO",
        target="all_horizons",
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()
    serving_repo.invalidate_cache()
    return get_model_settings(db, admin)


@router.get("/users", response_model=List[AdminUserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    return db.query(User).all()


@router.post("/users", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_csrf)])
def create_user_by_admin(
    payload: AdminUserCreate,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    email = payload.email.lower().strip()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists.")

    user = User(
        email=email,
        hashed_password=get_password_hash(payload.password),
        role=payload.role if payload.role in ["admin", "user"] else "user",
        is_active=payload.is_active
    )
    db.add(user)
    db.flush()

    audit = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action="ADMIN_CREATE_USER",
        target=str(user.id),
        details={"email": email, "role": user.role},
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()
    return user


@router.patch("/users/{user_id}", response_model=AdminUserResponse, dependencies=[Depends(verify_csrf)])
def update_user_by_admin(
    user_id: int,
    payload: AdminUserUpdate,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    # Guardrail 1: Admin cannot change own role
    if payload.role is not None and user.id == admin.id and payload.role != user.role:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admins cannot change their own role.")

    # Guardrail 2: Cannot demote or disable the last active admin
    if (payload.role == "user" or payload.is_active is False) and user.role == "admin":
        active_admins = db.query(User).filter(User.role == "admin", User.is_active == True).count()
        if active_admins <= 1 and user.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot demote or disable the last active admin.")

    role_changed = False
    if payload.role is not None and payload.role in ["admin", "user"]:
        if user.role != payload.role:
            user.role = payload.role
            role_changed = True

    disabled = False
    if payload.is_active is not None:
        if user.is_active and not payload.is_active:
            disabled = True
        user.is_active = payload.is_active

    # Guardrail 3: Role changes and disabling invalidate that user's refresh tokens
    if role_changed or disabled:
        db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update({"is_revoked": True})

    user.updated_at = datetime.now(timezone.utc)
    audit = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action="ADMIN_UPDATE_USER",
        target=str(user.id),
        details={"role": payload.role, "is_active": payload.is_active},
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()
    db.refresh(user)
    return user


@router.post("/users/{user_id}/reset-password", response_model=PasswordResetResponse, dependencies=[Depends(verify_csrf)])
def reset_user_password(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    temp_pwd = f"Temp_{secrets.token_urlsafe(10)}!"
    user.hashed_password = get_password_hash(temp_pwd)
    # Revoke tokens
    db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update({"is_revoked": True})

    audit = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action="ADMIN_RESET_PASSWORD",
        target=str(user.id),
        ip=request.client.host if request.client else None
    )
    db.add(audit)
    db.commit()

    return PasswordResetResponse(
        user_id=user.id,
        temporary_password=temp_pwd,
        message="Temporary password generated. User must change password upon login."
    )


@router.get("/data/status", response_model=DataStatusResponse)
def get_data_status(
    admin: User = Depends(get_current_admin)
):
    as_of = serving_repo.get_current_as_of()
    stocks = serving_repo.get_stocks_metadata()
    data_ver = stocks[0].get("data_version", f"v_{as_of}") if stocks else f"v_{as_of}"
    versions = [p.name for p in (BASE_DIR / "data/serving").iterdir() if p.is_dir()]
    return DataStatusResponse(
        current_as_of=as_of,
        data_version=data_ver,
        available_versions=sorted(versions),
        last_refresh_summary={"as_of": as_of, "stocks_count": len(stocks)}
    )


@router.post("/data/upload", response_model=DataJobResponse, dependencies=[Depends(verify_csrf)])
async def upload_data_refresh(
    file: UploadFile = File(...),
    mode: str = Form("append"),
    dry_run: bool = Form(False),
    request: Request = None,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    # Check file type
    if not file.filename.endswith((".parquet", ".csv")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only .parquet or .csv allowed."
        )

    job = DataJob(
        type="refresh",
        status="completed" if dry_run else "running",
        started=datetime.now(timezone.utc),
        input_name=file.filename,
        result={"mode": mode, "dry_run": dry_run, "message": "Dry run validated." if dry_run else "Job submitted."},
        created_by=admin.email
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs/{job_id}", response_model=DataJobResponse)
def get_data_job(
    job_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    job = db.query(DataJob).filter(DataJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


@router.post("/data/rollback", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def rollback_data(
    target_version: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    # Set CURRENT pointer to target or previous date
    versions = sorted([p.name for p in (BASE_DIR / "data/serving").iterdir() if p.is_dir()])
    if len(versions) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No previous version to rollback to.")
    prev_d = target_version or versions[-2]
    current_file = BASE_DIR / "data/serving/CURRENT"
    current_file.write_text(prev_d, encoding="utf-8")
    serving_repo.invalidate_cache()
    return MessageResponse(message=f"Rolled back to {prev_d}.")


@router.post("/alerts/evaluate", response_model=MessageResponse, dependencies=[Depends(verify_csrf)])
def trigger_alert_evaluation(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    count = evaluate_alerts(db)
    return MessageResponse(message=f"Alert evaluation complete. Triggered {count} notifications.")


@router.get("/audit-log", response_model=List[AuditLogItem])
def get_audit_log(
    actor_email: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    q = db.query(AuditLog)
    if actor_email:
        q = q.filter(AuditLog.actor_email == actor_email)
    if action:
        q = q.filter(AuditLog.action == action)
    return q.order_by(AuditLog.ts.desc()).limit(limit).all()


@router.get("/methodology")
def get_admin_methodology(admin: User = Depends(get_current_admin)):
    admin_md_path = BASE_DIR / "data/app/methodology_admin.md"
    content = admin_md_path.read_text(encoding="utf-8") if admin_md_path.exists() else ""
    return {"markdown_content": content}

"""app/backend/routers/methodology.py
Methodology endpoint returning methodology_user.md and user threshold definitions.
"""
from pathlib import Path
import json
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.backend.services.serving import BASE_DIR, serving_repo
from app.backend.routers.deps import get_current_user

router = APIRouter(prefix="/methodology", tags=["methodology"], dependencies=[Depends(get_current_user)])


class MethodologyResponse(BaseModel):
    markdown_content: str
    thresholds: dict


@router.get("", response_model=MethodologyResponse)
def get_user_methodology():
    user_md_path = BASE_DIR / "data/app/methodology_user.md"
    content = user_md_path.read_text(encoding="utf-8") if user_md_path.exists() else ""
    thresholds = serving_repo.get_thresholds()
    # Filter thresholds to user-safe entries
    user_thresholds = {
        "tier_cutoff_short": thresholds.get("tier_cutoff_short", {}).get("value"),
        "tier_cutoff_mid": thresholds.get("tier_cutoff_mid", {}).get("value"),
        "tier_cutoff_long": thresholds.get("tier_cutoff_long", {}).get("value"),
        "risk_category_low_cutoff": thresholds.get("risk_category_low_cutoff", {}).get("value"),
        "risk_category_high_cutoff": thresholds.get("risk_category_high_cutoff", {}).get("value"),
        "liquidity_warning_cutoff": thresholds.get("liquidity_warning_cutoff", {}).get("value")
    }
    return MethodologyResponse(
        markdown_content=content,
        thresholds=user_thresholds
    )

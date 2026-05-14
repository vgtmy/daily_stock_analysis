# -*- coding: utf-8 -*-
"""AI recommendation attribution endpoint."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Query

from api.v1.schemas.attribution import AttributionResponse
from src.services.attribution_service import AttributionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/attribution",
    response_model=AttributionResponse,
    summary="AI 推荐绩效归因",
    description="追踪历史 AI 买入/持有/卖出建议的实际表现，计算方向准确率和平均远期收益。",
)
def get_attribution(
    lookback_days: int = Query(90, ge=30, le=365, description="回溯天数"),
    min_forward_days: int = Query(5, ge=5, le=20, description="最小前瞻天数"),
) -> AttributionResponse:
    service = AttributionService()
    summary = service.get_attribution(
        lookback_days=lookback_days,
        min_forward_days=min_forward_days,
    )

    return AttributionResponse(
        total_recommendations=summary.total_recommendations,
        date_range=list(summary.date_range),
        by_decision=summary.by_decision,
        overall=summary.overall,
        recent_records=summary.recent_records,
    )

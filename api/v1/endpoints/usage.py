# -*- coding: utf-8 -*-
"""LLM usage tracking endpoint with cost estimation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from api.deps import get_database_manager
from api.v1.schemas.usage import (
    CallTypeBreakdown,
    ModelBreakdown,
    UsageSummaryResponse,
)
from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

_CST = timezone(timedelta(hours=8))

router = APIRouter()

_VALID_PERIODS = {"today", "month", "all"}

# Cost per 1M tokens (USD) — approximate, updated 2026Q2
_MODEL_COST_PER_MTOK: dict = {
    # Gemini
    "gemini-3-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    # Claude / Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    "claude-opus": (15.00, 75.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


def _match_model_cost(model: str):
    """Find the best matching cost entry for a model string."""
    model_lower = (model or "").lower()
    # Try exact match first, then prefix match
    for pattern, costs in _MODEL_COST_PER_MTOK.items():
        if pattern in model_lower:
            return costs
    # Default: assume unknown model priced like gpt-4o-mini
    return (0.15, 0.60)


def _estimate_cost(total_tokens: int, model: str) -> float:
    """Estimate USD cost based on model token usage (assume 3:1 input:output split)."""
    input_cost, output_cost = _match_model_cost(model)
    # Heuristic: ~75% input tokens, ~25% output tokens
    input_tokens = total_tokens * 0.75
    output_tokens = total_tokens * 0.25
    return round(
        (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost,
        6,
    )


def _date_range(period: str):
    """Return (from_dt, to_dt) as naive datetimes in Beijing time (UTC+8)."""
    now = datetime.now(tz=_CST).replace(tzinfo=None)
    if period == "today":
        from_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        from_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        from_dt = datetime(2000, 1, 1)
    return from_dt, now


@router.get(
    "/summary",
    response_model=UsageSummaryResponse,
    summary="LLM 用量与成本概览",
    description="按周期、调用类型、模型汇总 token 消耗与预估费用。",
)
def get_usage_summary(
    period: str = Query("month", description="'today' | 'month' | 'all'"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> UsageSummaryResponse:
    if period not in _VALID_PERIODS:
        period = "month"

    from_dt, to_dt = _date_range(period)

    data = db_manager.get_llm_usage_summary(from_dt, to_dt)

    # Compute cost estimates
    total_cost = 0.0
    by_model_with_cost = []
    for item in data["by_model"]:
        cost = _estimate_cost(item["total_tokens"], item["model"])
        total_cost += cost
        by_model_with_cost.append(
            ModelBreakdown(
                model=item["model"],
                calls=item["calls"],
                total_tokens=item["total_tokens"],
                estimated_cost_usd=round(cost, 6),
            )
        )

    by_call_type = [
        CallTypeBreakdown(
            call_type=item["call_type"],
            calls=item["calls"],
            total_tokens=item["total_tokens"],
        )
        for item in data["by_call_type"]
    ]

    return UsageSummaryResponse(
        period=period,
        from_date=from_dt.date().isoformat(),
        to_date=to_dt.date().isoformat(),
        total_calls=data["total_calls"],
        total_tokens=data["total_tokens"],
        total_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
        by_call_type=by_call_type,
        by_model=by_model_with_cost,
    )

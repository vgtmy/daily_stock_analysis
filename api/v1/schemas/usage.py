# -*- coding: utf-8 -*-
"""Schemas for LLM usage tracking API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CallTypeBreakdown(BaseModel):
    call_type: str = Field(..., description="'analysis' | 'agent' | 'market_review'")
    calls: int
    total_tokens: int


class ModelBreakdown(BaseModel):
    model: str
    calls: int
    total_tokens: int
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated USD cost")


class UsageSummaryResponse(BaseModel):
    period: str = Field(..., description="'today' | 'month' | 'all'")
    from_date: str = Field(..., description="ISO date string")
    to_date: str = Field(..., description="ISO date string")
    total_calls: int
    total_tokens: int
    total_cost_usd: Optional[float] = Field(None, description="Total estimated USD cost")
    by_call_type: List[CallTypeBreakdown]
    by_model: List[ModelBreakdown]

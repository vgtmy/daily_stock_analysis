# -*- coding: utf-8 -*-
"""Schemas for AI recommendation attribution API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionAttribution(BaseModel):
    count: int = 0
    avg_score: float = 0.0
    direction_accuracy_5d: Optional[float] = None
    direction_accuracy_10d: Optional[float] = None
    direction_accuracy_20d: Optional[float] = None
    avg_forward_return_5d: Optional[float] = None
    avg_forward_return_10d: Optional[float] = None
    avg_forward_return_20d: Optional[float] = None


class OverallAttribution(BaseModel):
    total_buy: int = 0
    total_hold: int = 0
    total_sell: int = 0
    overall_accuracy_5d: Optional[float] = None
    overall_avg_return_5d: Optional[float] = None


class RecentAttributionRecord(BaseModel):
    code: str
    name: str
    date: str
    decision: str
    score: int
    advice: str
    fwd_return_5d: Optional[float] = None
    fwd_return_10d: Optional[float] = None
    correct_5d: Optional[bool] = None
    correct_10d: Optional[bool] = None


class AttributionResponse(BaseModel):
    total_recommendations: int
    date_range: List[str]
    by_decision: Dict[str, DecisionAttribution]
    overall: OverallAttribution
    recent_records: List[RecentAttributionRecord]

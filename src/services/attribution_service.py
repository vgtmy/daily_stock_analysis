# -*- coding: utf-8 -*-
"""
AI recommendation performance attribution.

Tracks how AI buy/hold/sell recommendations perform over subsequent trading days.
Computes direction accuracy, average forward return, and hit rate by decision type.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from data_provider.base import normalize_stock_code
from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

_CST = timezone(timedelta(hours=8))


@dataclass
class AttributionRecord:
    code: str
    name: str
    analysis_date: date
    decision_type: str  # buy / hold / sell
    sentiment_score: int
    operation_advice: str
    forward_return_5d: Optional[float] = None
    forward_return_10d: Optional[float] = None
    forward_return_20d: Optional[float] = None
    direction_correct_5d: Optional[bool] = None
    direction_correct_10d: Optional[bool] = None
    direction_correct_20d: Optional[bool] = None
    actual_outcome: Optional[str] = None


@dataclass
class AttributionSummary:
    total_recommendations: int
    date_range: Tuple[str, str]
    by_decision: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall: Dict[str, Any] = field(default_factory=dict)
    recent_records: List[Dict[str, Any]] = field(default_factory=list)


class AttributionService:
    """Compute AI recommendation performance against actual price movements."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()

    def get_attribution(
        self,
        *,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        lookback_days: int = 90,
        min_forward_days: int = 5,
    ) -> AttributionSummary:
        """Return attribution summary for AI recommendations in the window."""
        if to_date is None:
            to_date = date.today()
        if from_date is None:
            from_date = to_date - timedelta(days=lookback_days)

        records = self._load_records(from_date, to_date, min_forward_days)
        if not records:
            return AttributionSummary(
                total_recommendations=0,
                date_range=(from_date.isoformat(), to_date.isoformat()),
            )

        return self._build_summary(records, from_date, to_date)

    def _load_records(
        self,
        from_date: date,
        to_date: date,
        min_forward_days: int,
    ) -> List[AttributionRecord]:
        """Load analysis history and compute forward returns."""
        from src.storage import AnalysisHistory

        cutoff = to_date - timedelta(days=min_forward_days)

        with self.db.session_scope() as session:
            from sqlalchemy import and_, select

            query = (
                select(AnalysisHistory)
                .where(
                    and_(
                        AnalysisHistory.analysis_date >= from_date,
                        AnalysisHistory.analysis_date <= cutoff,
                        AnalysisHistory.success == True,  # noqa: E712
                    )
                )
                .order_by(AnalysisHistory.analysis_date.desc())
                .limit(500)
            )
            rows = session.execute(query).scalars().all()

        records: List[AttributionRecord] = []
        for row in rows:
            decision = (row.decision_type or "").strip().lower()
            if decision not in ("buy", "hold", "sell"):
                continue

            rec = AttributionRecord(
                code=normalize_stock_code(row.code or ""),
                name=row.stock_name or row.code or "",
                analysis_date=row.analysis_date if isinstance(row.analysis_date, date) else row.analysis_date.date(),
                decision_type=decision,
                sentiment_score=row.sentiment_score or 50,
                operation_advice=row.operation_advice or "",
            )

            # Compute forward returns if price data available
            self._fill_forward_returns(rec, min_forward_days)
            records.append(rec)

        return records

    def _fill_forward_returns(self, rec: AttributionRecord, min_forward_days: int) -> None:
        """Compute N-day forward return using DB price data."""
        analysis_dt = rec.analysis_date
        if not isinstance(analysis_dt, date):
            return

        # Get price at analysis date
        analysis_price = self._get_close_price(rec.code, analysis_dt)
        if analysis_price is None or analysis_price <= 0:
            return

        for horizon, label in [(5, "5d"), (10, "10d"), (20, "20d")]:
            future_dt = analysis_dt + timedelta(days=horizon + 5)  # buffer for weekends
            future_price = self._get_closest_close(rec.code, analysis_dt, future_dt)
            if future_price is None:
                continue

            fwd_return = (future_price - analysis_price) / analysis_price
            setattr(rec, f"forward_return_{label}", round(fwd_return, 6))

            # Directional correctness
            if rec.decision_type == "buy":
                correct = fwd_return > 0
            elif rec.decision_type == "sell":
                correct = fwd_return < 0
            else:  # hold
                correct = abs(fwd_return) < 0.03  # within ±3% is "correct hold"

            setattr(rec, f"direction_correct_{label}", correct)

    def _get_close_price(self, code: str, target_date: date) -> Optional[float]:
        """Get closing price for a stock on a given date."""
        from src.storage import StockDaily

        normalized = normalize_stock_code(code)
        with self.db.session_scope() as session:
            from sqlalchemy import select

            query = select(StockDaily.close).where(
                StockDaily.code == normalized,
                StockDaily.date == target_date,
            ).limit(1)
            result = session.execute(query).scalar()
            return float(result) if result else None

    def _get_closest_close(self, code: str, from_date: date, to_date: date) -> Optional[float]:
        """Get the closest closing price within [from_date + 1, to_date]."""
        from src.storage import StockDaily

        normalized = normalize_stock_code(code)
        with self.db.session_scope() as session:
            from sqlalchemy import select

            query = (
                select(StockDaily.close)
                .where(
                    StockDaily.code == normalized,
                    StockDaily.date > from_date,
                    StockDaily.date <= to_date,
                )
                .order_by(StockDaily.date.desc())
                .limit(1)
            )
            result = session.execute(query).scalar()
            return float(result) if result else None

    def _build_summary(
        self,
        records: List[AttributionRecord],
        from_date: date,
        to_date: date,
    ) -> AttributionSummary:
        """Aggregate attribution records into summary statistics."""
        by_decision: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "avg_score": 0,
            "direction_accuracy_5d": None,
            "direction_accuracy_10d": None,
            "direction_accuracy_20d": None,
            "avg_forward_return_5d": None,
            "avg_forward_return_10d": None,
            "avg_forward_return_20d": None,
        })

        for rec in records:
            group = by_decision[rec.decision_type]
            group["count"] += 1
            group["avg_score"] += rec.sentiment_score

        # Compute averages
        for decision, group in by_decision.items():
            if group["count"] > 0:
                group["avg_score"] = round(group["avg_score"] / group["count"], 1)

        # Directional accuracy & average forward returns
        for horizon, label in [(5, "5d"), (10, "10d"), (20, "20d")]:
            for decision in by_decision:
                subset = [r for r in records if r.decision_type == decision]
                correct = [r for r in subset if getattr(r, f"direction_correct_{label}") is not None]
                if correct:
                    accuracy = sum(1 for r in correct if getattr(r, f"direction_correct_{label}")) / len(correct)
                    by_decision[decision][f"direction_accuracy_{label}"] = round(accuracy, 4)

                returns = [r for r in subset if getattr(r, f"forward_return_{label}") is not None]
                if returns:
                    avg_ret = sum(getattr(r, f"forward_return_{label}") for r in returns) / len(returns)
                    by_decision[decision][f"avg_forward_return_{label}"] = round(avg_ret, 4)

        # Overall stats
        all_direction_5d = [r for r in records if r.direction_correct_5d is not None]
        all_returns_5d = [r for r in records if r.forward_return_5d is not None]

        overall = {
            "total_buy": by_decision.get("buy", {}).get("count", 0),
            "total_hold": by_decision.get("hold", {}).get("count", 0),
            "total_sell": by_decision.get("sell", {}).get("count", 0),
            "overall_accuracy_5d": round(
                sum(1 for r in all_direction_5d if r.direction_correct_5d) / len(all_direction_5d), 4
            ) if all_direction_5d else None,
            "overall_avg_return_5d": round(
                sum(r.forward_return_5d for r in all_returns_5d) / len(all_returns_5d), 4
            ) if all_returns_5d else None,
        }

        # Recent records (last 30, sorted by date desc)
        recent = sorted(records, key=lambda r: r.analysis_date, reverse=True)[:30]
        recent_dicts = [
            {
                "code": r.code,
                "name": r.name,
                "date": r.analysis_date.isoformat(),
                "decision": r.decision_type,
                "score": r.sentiment_score,
                "advice": r.operation_advice,
                "fwd_return_5d": r.forward_return_5d,
                "fwd_return_10d": r.forward_return_10d,
                "correct_5d": r.direction_correct_5d,
                "correct_10d": r.direction_correct_10d,
            }
            for r in recent
        ]

        # Sort by_decision for consistent output
        by_decision_sorted = dict(sorted(by_decision.items()))

        return AttributionSummary(
            total_recommendations=len(records),
            date_range=(from_date.isoformat(), to_date.isoformat()),
            by_decision=by_decision_sorted,
            overall=overall,
            recent_records=recent_dicts,
        )

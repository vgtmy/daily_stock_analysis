# -*- coding: utf-8 -*-
"""
Circuit breaker for data source failover.

When a data source fails N times within a window, the breaker opens and
fast-fails subsequent calls for a cooldown period, allowing the failover
chain to skip it without wasted retries.

Pattern: standard three-state circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing fast
    HALF_OPEN = "half_open"  # Probing recovery


@dataclass
class CircuitBreaker:
    """
    Per-data-source circuit breaker.

    Configuration:
        failure_threshold: consecutive failures before opening
        cooldown_seconds:  how long to stay OPEN before trying HALF_OPEN
        half_open_max:     max probe calls in HALF_OPEN before deciding
    """

    name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    half_open_max: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_count: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow_request(self) -> bool:
        """Return True if the request should be attempted."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_count = 0
                    logger.info(
                        "[熔断器] %s 进入半开状态，尝试探测恢复", self.name
                    )
                    return True
                return False
            # HALF_OPEN: allow limited probes
            if self._half_open_count < self.half_open_max:
                self._half_open_count += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful call, closing the circuit."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("[熔断器] %s 恢复关闭状态", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call, potentially opening the circuit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("[熔断器] %s 探测失败，重新打开", self.name)
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "[熔断器] %s 连续失败 %d 次，熔断打开 (冷却 %ds)",
                    self.name,
                    self._failure_count,
                    self.cooldown_seconds,
                )

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN


class CircuitBreakerRegistry:
    """Thread-safe registry of circuit breakers keyed by data source name."""

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name)
            return self._breakers[name]

    def status(self) -> Dict[str, str]:
        """Return a snapshot of breaker states for diagnostics."""
        with self._lock:
            return {name: cb.state.value for name, cb in self._breakers.items()}


# Module-level singleton for use across the application.
_registry: Optional[CircuitBreakerRegistry] = None
_registry_lock = threading.Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = CircuitBreakerRegistry()
        return _registry

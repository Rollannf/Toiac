"""Typed structures shared across the D₁.₁ analytical core."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass(slots=True)
class TaskPayload:
    """Normalized representation of an incoming analytical task."""

    task_id: str
    task_type: str
    input: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)
    crc: Optional[int] = None
    received_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ExecutionPlan:
    """Semantic execution plan derived from the task payload."""

    route: str
    intent: str
    strategy: Mapping[str, Any]
    task: TaskPayload


@dataclass(slots=True)
class ExecutionResult:
    """Result produced by one of the downstream analytical executors."""

    task_id: str
    status: str
    result: Mapping[str, Any]
    crc: Optional[int]
    latency: float
    executor_route: str
    annotations: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PushDecision:
    """Decision describing a predictive push action."""

    should_push: bool
    resources: List[str]
    reason: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SLAStatus:
    """Snapshot of the SLA controller state."""

    latency_avg: float
    latency_p95: float
    violations: int
    window: int
    degraded: bool


@dataclass(slots=True)
class MetricPoint:
    """Point-in-time metric measurement."""

    name: str
    value: float
    labels: Mapping[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ControlCommand:
    """Normalized command coming from the CMV₁ control-plane."""

    type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    issued_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class WatchdogProbe:
    """Entry describing watchdog heartbeat checks."""

    name: str
    last_seen: datetime
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)


def ensure_fields(mapping: Mapping[str, Any], fields: Iterable[str]) -> None:
    """Ensure *fields* are present in *mapping* raising ``ValueError`` otherwise."""

    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

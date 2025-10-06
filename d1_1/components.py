"""Building blocks for the D₁.₁ analytical core."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

from .types import (
    ControlCommand,
    ExecutionPlan,
    ExecutionResult,
    MetricPoint,
    PushDecision,
    SLAStatus,
    TaskPayload,
    WatchdogProbe,
    ensure_fields,
)

logger = logging.getLogger(__name__)

RouteHandler = Callable[[TaskPayload, Mapping[str, Any]], Awaitable[ExecutionResult]]


class DiagnosticL1:
    """Performs integrity and contract validation for incoming payloads."""

    mandatory_fields: Iterable[str] = ("task_id", "task_type", "input")

    def check_integrity(self, payload: Mapping[str, Any]) -> bool:
        crc = payload.get("crc")
        if crc is None:
            return True
        computed = self._compute_crc(payload)
        if computed != crc:
            raise ValueError("CRC mismatch detected")
        return True

    def check_contract(self, payload: Mapping[str, Any]) -> bool:
        ensure_fields(payload, self.mandatory_fields)
        return True

    @staticmethod
    def _compute_crc(payload: Mapping[str, Any]) -> int:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.crc32(serialized)
        return digest


class InputParser:
    """Normalises inbound payloads from M₁.₁ into :class:`TaskPayload`."""

    def __init__(self, diagnostics: Optional[DiagnosticL1] = None, *, default_context: Optional[Mapping[str, Any]] = None) -> None:
        self.diagnostics = diagnostics or DiagnosticL1()
        self.default_context = dict(default_context or {})

    def parse(self, payload: Mapping[str, Any]) -> TaskPayload:
        """Validate payload and convert it into a :class:`TaskPayload`."""

        self.diagnostics.check_contract(payload)
        self.diagnostics.check_integrity(payload)

        context = dict(self.default_context)
        context.update(payload.get("context", {}))
        return TaskPayload(
            task_id=str(payload["task_id"]),
            task_type=str(payload["task_type"]),
            input=dict(payload["input"]),
            context=context,
            crc=payload.get("crc"),
        )


class OutputFormatter:
    """Formats :class:`ExecutionResult` objects for transmission to M₁.₁."""

    def __init__(self, *, include_metadata: bool = True) -> None:
        self.include_metadata = include_metadata

    def format(self, result: ExecutionResult, *, metadata: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "task_id": result.task_id,
            "status": result.status,
            "result": dict(result.result),
            "crc": result.crc,
        }
        if self.include_metadata and metadata:
            payload["meta"] = dict(metadata)
        return payload


class AnalyticalRouter:
    """Decides which downstream route should receive the task."""

    def __init__(self) -> None:
        self._overrides: Dict[str, str] = {}

    def register_override(self, task_type: str, route: str) -> None:
        self._overrides[task_type] = route

    def decide_route(self, payload: TaskPayload) -> str:
        if payload.task_type in self._overrides:
            return self._overrides[payload.task_type]
        task_type = payload.task_type.lower()
        if task_type in {"kpi", "statistics"}:
            return "S_1.1.1"
        if task_type in {"graphs", "trends", "forecast", "forecasts"}:
            return "S_1.1.2"
        if task_type in {"tables", "reports"}:
            return "S_1.1.3"
        return "S_1.1.4"


class SemanticPlanner:
    """Maps payloads to execution plans with semantic hints."""

    def plan(self, payload: TaskPayload, route: str) -> ExecutionPlan:
        context = dict(payload.context)
        intent = context.get("intent") or self._infer_intent(payload)
        strategy = self._build_strategy(payload, intent)
        return ExecutionPlan(route=route, intent=intent, strategy=strategy, task=payload)

    @staticmethod
    def _infer_intent(payload: TaskPayload) -> str:
        hints = [payload.task_type.lower(), *(payload.context.get("tags", []))]
        if any("trend" in hint for hint in hints):
            return "analyze_trend"
        if any("report" in hint for hint in hints):
            return "build_report"
        if any("forecast" in hint for hint in hints):
            return "predict_forecast"
        return "aggregate_metrics"

    @staticmethod
    def _build_strategy(payload: TaskPayload, intent: str) -> Mapping[str, Any]:
        precision = "high" if payload.context.get("priority") == "high" else "normal"
        cache_ttl = 600 if intent in {"aggregate_metrics", "build_report"} else 120
        return {
            "precision": precision,
            "cache_ttl": cache_ttl,
            "retries": 2 if precision == "high" else 1,
            "deadline": payload.context.get("deadline", 3.0),
        }


class AnalyticalExecutor:
    """Coordinates execution of tasks through registered route handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, RouteHandler] = {}

    def register_handler(self, route: str, handler: RouteHandler) -> None:
        self._handlers[route] = handler

    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        if plan.route not in self._handlers:
            raise RuntimeError(f"No handler registered for route {plan.route}")
        handler = self._handlers[plan.route]
        start = time.perf_counter()
        result = await handler(plan.task, plan.strategy)
        latency = time.perf_counter() - start
        result.annotations.setdefault("latency" if "latency" not in result.annotations else "latency_plan", latency)
        return ExecutionResult(
            task_id=result.task_id,
            status=result.status,
            result=result.result,
            crc=result.crc,
            latency=latency,
            executor_route=plan.route,
            annotations=result.annotations,
        )


class PredictiveEngine:
    """Learns latency and reuse statistics to drive predictive pushes."""

    def __init__(self, history_window: int = 100) -> None:
        self._latency_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=history_window))
        self._reuse_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=history_window))
        self._waste_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=history_window))
        self._bias: Dict[str, float] = defaultdict(float)
        self._safe_mode_until: Optional[datetime] = None

    def record(self, task_type: str, latency: float, reused: float, wasted: float) -> None:
        self._latency_history[task_type].append(latency)
        self._reuse_history[task_type].append(reused)
        self._waste_history[task_type].append(wasted)

    def decide(self, task_type: str) -> PushDecision:
        if self._in_safe_mode():
            return PushDecision(False, [], "safe_mode", 0.0)

        latencies = self._latency_history.get(task_type)
        reuse = self._reuse_history.get(task_type)
        waste = self._waste_history.get(task_type)
        if not latencies:
            return PushDecision(False, [], "insufficient_history", 0.0)
        avg_latency = statistics.fmean(latencies)
        reuse_rate = statistics.fmean(reuse) if reuse else 0.0
        waste_rate = statistics.fmean(waste) if waste else 0.0
        efficiency = reuse_rate - waste_rate
        adjusted_efficiency = efficiency + self._bias.get(task_type, 0.0)
        should_push = avg_latency > 0.8 or adjusted_efficiency > 0.2
        resources: Sequence[str] = []
        if should_push:
            resources = (
                f"/prefetch/{task_type.lower()}/schema.json",
                f"/prefetch/{task_type.lower()}/last_result.json",
            )
        reason = self._decide_reason(avg_latency, adjusted_efficiency)
        confidence = self._compute_confidence(avg_latency, reuse_rate, waste_rate, adjusted_efficiency)
        metadata = {
            "avg_latency": avg_latency,
            "reuse_rate": reuse_rate,
            "waste_rate": waste_rate,
            "efficiency": efficiency,
            "adjusted_efficiency": adjusted_efficiency,
            "bias": self._bias.get(task_type, 0.0),
        }
        return PushDecision(should_push, list(resources), reason, confidence, metadata)

    def boost(self, task_type: str, factor: float = 0.2) -> None:
        """Increase predictive bias for *task_type* to favour push."""

        self._bias[task_type] = min(1.0, self._bias.get(task_type, 0.0) + abs(factor))

    def suppress(self, task_type: str, factor: float = 0.2) -> None:
        """Decrease predictive bias for *task_type* to discourage push."""

        self._bias[task_type] = max(-1.0, self._bias.get(task_type, 0.0) - abs(factor))

    def enter_safe_mode(self, duration: float = 60.0) -> None:
        """Disable predictive push decisions temporarily."""

        self._safe_mode_until = datetime.utcnow() + timedelta(seconds=duration)

    def exit_safe_mode(self) -> None:
        self._safe_mode_until = None

    def _in_safe_mode(self) -> bool:
        return self._safe_mode_until is not None and datetime.utcnow() < self._safe_mode_until

    @staticmethod
    def _decide_reason(avg_latency: float, efficiency: float) -> str:
        if avg_latency > 0.8 and efficiency > 0.2:
            return "latency_and_efficiency"
        if avg_latency > 0.8:
            return "latency_high"
        if efficiency > 0.2:
            return "efficiency_positive"
        return "insufficient_signal"

    @staticmethod
    def _compute_confidence(
        avg_latency: float,
        reuse_rate: float,
        waste_rate: float,
        adjusted_efficiency: float,
    ) -> float:
        base = (reuse_rate * 0.6) - (waste_rate * 0.4)
        latency_component = min(0.5, max(0.0, avg_latency / 5))
        efficiency_component = max(0.0, min(0.5, adjusted_efficiency))
        return max(0.05, min(0.99, base + latency_component + efficiency_component))


class Metrics:
    """In-memory metrics aggregation."""

    def __init__(self) -> None:
        self._latencies: Deque[float] = deque(maxlen=500)
        self._sla_breaches = 0
        self._records: Deque[MetricPoint] = deque(maxlen=1000)

    def record_latency(self, value: float) -> None:
        self._latencies.append(value)
        self._records.append(MetricPoint(name="task_latency", value=value))

    def record_sla_breach(self) -> None:
        self._sla_breaches += 1
        self._records.append(MetricPoint(name="sla_breach", value=1.0))

    def export(self) -> Mapping[str, Any]:
        if self._latencies:
            avg = statistics.fmean(self._latencies)
            p95 = sorted(self._latencies)[int(0.95 * (len(self._latencies) - 1))]
        else:
            avg = 0.0
            p95 = 0.0
        return {
            "latency_avg": avg,
            "latency_p95": p95,
            "sla_breaches": self._sla_breaches,
            "samples": len(self._latencies),
        }


class StateStore:
    """Tracks task state and cached results."""

    def __init__(self) -> None:
        self._tasks: Dict[str, ExecutionResult] = {}
        self._task_meta: Dict[str, Dict[str, Any]] = {}
        self._push_state: Dict[str, str] = {}

    def save_result(self, result: ExecutionResult) -> None:
        self._tasks[result.task_id] = result

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        return self._tasks.get(task_id)

    def save_meta(self, task_id: str, meta: Mapping[str, Any]) -> None:
        self._task_meta[task_id] = dict(meta)

    def get_meta(self, task_id: str) -> Mapping[str, Any]:
        return self._task_meta.get(task_id, {})

    def update_push_state(self, task_id: str, state: str) -> None:
        self._push_state[task_id] = state

    def get_push_state(self, task_id: str) -> Optional[str]:
        return self._push_state.get(task_id)


class SLAController:
    """Monitors SLA compliance and exposes adaptive throttling information."""

    def __init__(self, metrics: Metrics, threshold: float = 3.0) -> None:
        self.metrics = metrics
        self.threshold = threshold
        self._latency_window: Deque[float] = deque(maxlen=200)
        self._violations: int = 0
        self._degraded: bool = False

    def record(self, latency: float) -> bool:
        """Record *latency* and return ``True`` if the sample breached the SLA."""

        self._latency_window.append(latency)
        self.metrics.record_latency(latency)
        if latency > self.threshold:
            self._violations += 1
            self.metrics.record_sla_breach()
            if self._violations > 5:
                self._degraded = True
            return True
        return False

    def reset(self) -> None:
        self._latency_window.clear()
        self._violations = 0
        self._degraded = False

    def status(self) -> SLAStatus:
        latency_avg = statistics.fmean(self._latency_window) if self._latency_window else 0.0
        if self._latency_window:
            sorted_samples = sorted(self._latency_window)
            index = int(0.95 * (len(sorted_samples) - 1))
            latency_p95 = sorted_samples[index]
        else:
            latency_p95 = 0.0
        return SLAStatus(
            latency_avg=latency_avg,
            latency_p95=latency_p95,
            violations=self._violations,
            window=len(self._latency_window),
            degraded=self._degraded,
        )

    def degraded(self) -> bool:
        return self._degraded


class Watchdog:
    """Performs periodic probes and triggers callbacks when timeouts occur."""

    def __init__(self, interval: float = 15.0, timeout: float = 45.0) -> None:
        self.interval = interval
        self.timeout = timeout
        self._probes: Dict[str, WatchdogProbe] = {}
        self._callbacks: Dict[str, Callable[[WatchdogProbe], Awaitable[None]]] = {}
        self._task: Optional[asyncio.Task[None]] = None

    def update_probe(self, name: str, status: str = "ok", **details: Any) -> None:
        self._probes[name] = WatchdogProbe(name=name, last_seen=datetime.utcnow(), status=status, details=details)

    def register_callback(self, name: str, callback: Callable[[WatchdogProbe], Awaitable[None]]) -> None:
        self._callbacks[name] = callback

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            await self._tick()

    async def _tick(self) -> None:
        now = datetime.utcnow()
        for name, probe in list(self._probes.items()):
            if (now - probe.last_seen) > timedelta(seconds=self.timeout):
                probe = WatchdogProbe(name=name, last_seen=probe.last_seen, status="timeout", details=probe.details)
                if name in self._callbacks:
                    await self._callbacks[name](probe)


class FeedbackChannel:
    """Sends feedback to CMV₁ via the parent M₁.₁ module (represented by callback)."""

    def __init__(self, emitter: Callable[[Mapping[str, Any]], Awaitable[None]]) -> None:
        self._emitter = emitter

    async def publish(self, payload: Mapping[str, Any]) -> None:
        await self._emitter(dict(payload))


class ControlPlane:
    """Processes control commands propagated by CMV₁."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[ControlCommand], Awaitable[None]]] = {}

    def register(self, command: str, handler: Callable[[ControlCommand], Awaitable[None]]) -> None:
        self._handlers[command.upper()] = handler

    async def dispatch(self, command: ControlCommand) -> None:
        handler = self._handlers.get(command.type.upper())
        if handler:
            await handler(command)
        else:
            logger.warning("Unknown control command received: %s", command.type)


class MRPClient:
    """Simple logger bridge imitating the behaviour of the real MRP system."""

    def __init__(self) -> None:
        self.records: Deque[Mapping[str, Any]] = deque(maxlen=1000)

    async def log(self, event: str, payload: Mapping[str, Any], level: str = "INFO") -> None:
        record = {
            "event": event,
            "payload": dict(payload),
            "level": level,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.records.append(record)
        logger.log(getattr(logging, level, logging.INFO), "MRP %s: %s", event, payload)


class ThrottleController:
    """Adjusts concurrency limits in reaction to THROTTLE commands."""

    def __init__(self, initial_rate: int = 5) -> None:
        self._baseline = max(1, initial_rate)
        self._rate = self._baseline
        self._semaphore = asyncio.Semaphore(self._rate)

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def baseline(self) -> int:
        return self._baseline

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    async def throttle(self, value: float) -> None:
        """Scale concurrency relative to the baseline by *value*."""

        value = max(0.1, min(2.0, value))
        target = max(1, int(round(self._baseline * value)))
        await self._update_rate(target)

    async def restore(self) -> None:
        await self._update_rate(self._baseline)

    async def _update_rate(self, target: int) -> None:
        if target == self._rate:
            return
        if target > self._rate:
            for _ in range(target - self._rate):
                self._semaphore.release()
        else:
            for _ in range(self._rate - target):
                await self._semaphore.acquire()
        self._rate = target


class ReplayBuffer:
    """Stores payloads for potential replay requests from CMV₁."""

    def __init__(self, size: int = 50) -> None:
        self._buffer: Deque[TaskPayload] = deque(maxlen=size)

    def append(self, payload: TaskPayload) -> None:
        self._buffer.append(payload)

    def recent(self, limit: int = 10) -> Iterable[TaskPayload]:
        return list(self._buffer)[-limit:]

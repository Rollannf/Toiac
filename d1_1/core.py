"""High-level orchestration logic for the D₁.₁ analytical core."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Mapping, Optional

from .components import (
    AnalyticalExecutor,
    AnalyticalRouter,
    ControlPlane,
    DiagnosticL1,
    FeedbackChannel,
    InputParser,
    MRPClient,
    Metrics,
    OutputFormatter,
    PredictiveEngine,
    ReplayBuffer,
    SLAController,
    SemanticPlanner,
    StateStore,
    ThrottleController,
    Watchdog,
)
from .types import ControlCommand, ExecutionResult, PushDecision, TaskPayload

logger = logging.getLogger(__name__)


class D11Core:
    """Implements the orchestration logic for the D₁.₁ child core."""

    def __init__(
        self,
        *,
        feedback_emitter: Optional[Callable[[Mapping[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self.diagnostics = DiagnosticL1()
        self.parser = InputParser(self.diagnostics)
        self.formatter = OutputFormatter()
        self.router = AnalyticalRouter()
        self.planner = SemanticPlanner()
        self.executor = AnalyticalExecutor()
        self.predictor = PredictiveEngine()
        self.metrics = Metrics()
        self.state = StateStore()
        self.sla = SLAController(metrics=self.metrics)
        self.control = ControlPlane()
        self.watchdog = Watchdog()
        self.throttle = ThrottleController()
        self.mrp = MRPClient()
        self.replay = ReplayBuffer()
        self.feedback = FeedbackChannel(feedback_emitter or self._default_feedback)

        self._register_control_handlers()

    async def _default_feedback(self, payload: Mapping[str, Any]) -> None:
        logger.info("Feedback: %s", payload)

    def register_route(self, route: str, handler: Callable[[TaskPayload, Mapping[str, Any]], Awaitable[ExecutionResult]]) -> None:
        self.executor.register_handler(route, handler)

    async def handle_execute(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Process incoming task payload and return the response payload."""

        task = self.parser.parse(payload)
        self.replay.append(task)

        route = self.router.decide_route(task)
        plan = self.planner.plan(task, route)
        await self.mrp.log("TaskPlanned", {"task_id": task.task_id, "route": route, "intent": plan.intent})

        await self.throttle.acquire()
        try:
            result = await self.executor.execute(plan)
        finally:
            self.throttle.release()

        self.state.save_result(result)
        self.state.save_meta(
            task.task_id,
            {
                "route": route,
                "intent": plan.intent,
                "strategy": dict(plan.strategy),
            },
        )

        breach = self.sla.record(result.latency)
        if breach:
            await self._on_sla_breach(task, result)
        elif self.sla.degraded() and result.latency < self.sla.threshold:
            self.predictor.exit_safe_mode()

        decision = self.predictor.decide(task.task_type)
        await self._handle_predictive_push(task.task_id, decision)
        await self.predictor_update(task.task_type, result)

        await self.feedback.publish(
            {
                "task_id": task.task_id,
                "status": result.status,
                "latency": result.latency,
                "route": route,
            }
        )

        payload = self.formatter.format(
            result,
            metadata={
                "route": route,
                "intent": plan.intent,
                "strategy": dict(plan.strategy),
            },
        )
        await self.mrp.log("TaskCompleted", payload)
        return payload

    async def predictor_update(self, task_type: str, result: ExecutionResult) -> None:
        reused = float(result.annotations.get("reuse_rate", 0.0))
        wasted = float(result.annotations.get("waste_rate", 0.0))
        self.predictor.record(task_type, result.latency, reused, wasted)

    async def _handle_predictive_push(self, task_id: str, decision: PushDecision) -> None:
        if not decision.should_push:
            return
        self.state.update_push_state(task_id, "predicted")
        await self.mrp.log(
            "PredictivePush",
            {
                "task_id": task_id,
                "resources": decision.resources,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "metadata": dict(decision.metadata),
            },
        )

    async def handle_control(self, command: Mapping[str, Any]) -> None:
        normalized = ControlCommand(type=command.get("type", ""), payload=command)
        await self.control.dispatch(normalized)

    def _register_control_handlers(self) -> None:
        self.control.register("THROTTLE", self._handle_throttle)
        self.control.register("RESET", self._handle_reset)
        self.control.register("REVOKE_TASK", self._handle_revoke)
        self.control.register("PUSH_STATS", self._handle_push_stats)
        self.control.register("ANALYZE_LATENCY", self._handle_analyze_latency)
        self.control.register("BOOST_PUSH", self._handle_boost_push)
        self.control.register("SAFE_MODE", self._handle_safe_mode)
        self.control.register("RESTORE_RATE", self._handle_restore_rate)

    async def _handle_throttle(self, command: ControlCommand) -> None:
        value = float(command.payload.get("value", 1.0))
        await self.throttle.throttle(value)
        await self.mrp.log("ThrottleUpdated", {"value": value})

    async def _handle_reset(self, command: ControlCommand) -> None:
        self.state = StateStore()
        self.sla.reset()
        self.predictor.exit_safe_mode()
        await self.throttle.restore()
        await self.mrp.log("ResetPerformed", {"source": command.payload.get("source", "ctrl")}, level="WARNING")

    async def _handle_revoke(self, command: ControlCommand) -> None:
        task_id = command.payload.get("task_id")
        if task_id and self.state.get_result(task_id):
            self.state.update_push_state(task_id, "revoked")
            await self.mrp.log("TaskRevoked", {"task_id": task_id}, level="WARNING")

    async def _handle_push_stats(self, command: ControlCommand) -> None:
        status = self.sla.status()
        await self.feedback.publish(
            {
                "type": "push_stats",
                "status": {
                    "latency_avg": status.latency_avg,
                    "latency_p95": status.latency_p95,
                    "violations": status.violations,
                    "window": status.window,
                    "degraded": status.degraded,
                },
            }
        )

    async def _handle_analyze_latency(self, command: ControlCommand) -> None:
        snapshot = self.sla.status()
        await self.mrp.log(
            "LatencyAnalyzed",
            {
                "latency_avg": snapshot.latency_avg,
                "latency_p95": snapshot.latency_p95,
                "violations": snapshot.violations,
                "window": snapshot.window,
                "degraded": snapshot.degraded,
            },
        )

    async def _handle_boost_push(self, command: ControlCommand) -> None:
        task_type = command.payload.get("task_type", "generic")
        factor = float(command.payload.get("factor", 0.2))
        action = command.payload.get("action", "boost").lower()
        if action == "suppress":
            self.predictor.suppress(task_type, factor)
        else:
            self.predictor.boost(task_type, factor)
        await self.mrp.log(
            "PredictorBiasUpdated",
            {"task_type": task_type, "factor": factor, "action": action},
        )

    async def _handle_safe_mode(self, command: ControlCommand) -> None:
        duration = float(command.payload.get("duration", 60.0))
        self.predictor.enter_safe_mode(duration)
        await self.mrp.log("PredictorSafeMode", {"duration": duration})

    async def _handle_restore_rate(self, command: ControlCommand) -> None:
        await self.throttle.restore()
        await self.mrp.log("ThrottleRestored", {"rate": self.throttle.rate})

    async def fetch_status(self, task_id: str) -> Optional[Mapping[str, Any]]:
        result = self.state.get_result(task_id)
        if not result:
            return None
        return {
            "task_id": task_id,
            "status": result.status,
            "result": dict(result.result),
            "crc": result.crc,
            "latency": result.latency,
            "route": result.executor_route,
        }

    async def metrics_snapshot(self) -> Mapping[str, Any]:
        status = self.sla.status()
        base = self.metrics.export()
        base.update(
            {
                "sla": {
                    "latency_avg": status.latency_avg,
                    "latency_p95": status.latency_p95,
                    "violations": status.violations,
                    "window": status.window,
                    "degraded": status.degraded,
                }
            }
        )
        return base

    async def replay_buffer(self) -> Mapping[str, Any]:
        tasks = [
            {
                "task_id": payload.task_id,
                "task_type": payload.task_type,
                "input": dict(payload.input),
                "context": dict(payload.context),
                "received_at": payload.received_at.isoformat(),
            }
            for payload in self.replay.recent()
        ]
        return {"recent_tasks": tasks}

    async def start_watchdog(self) -> None:
        async def _self_heal(probe) -> None:
            await self.mrp.log("SelfHealTriggered", {"reason": "watchdog_timeout", "probe": probe.name}, level="WARNING")

        self.watchdog.register_callback("core", _self_heal)
        self.watchdog.update_probe("core")
        asyncio.create_task(self.watchdog.run())

    async def _on_sla_breach(self, task: TaskPayload, result: ExecutionResult) -> None:
        await self.mrp.log(
            "SLA_Breach",
            {
                "task_id": task.task_id,
                "latency": result.latency,
                "threshold": self.sla.threshold,
                "route": result.executor_route,
            },
            level="WARNING",
        )
        await self.throttle.throttle(0.5)
        if self.sla.degraded():
            self.predictor.enter_safe_mode(120.0)
            await self.mrp.log(
                "SafeModeEnabled",
                {"reason": "sla_degraded", "task_id": task.task_id},
                level="WARNING",
            )


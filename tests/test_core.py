import asyncio
import pathlib
import sys
from typing import Mapping

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d1_1 import D11Core
from d1_1.types import ExecutionResult, TaskPayload


async def _dummy_handler(task: TaskPayload, strategy: Mapping[str, object]) -> ExecutionResult:
    await asyncio.sleep(0)
    annotations = {"reuse_rate": 0.7, "waste_rate": 0.1}
    return ExecutionResult(
        task_id=task.task_id,
        status="completed",
        result={"avg_revenue": 100.0, "growth_rate": 0.12},
        crc=1234,
        latency=0.01,
        executor_route="S_1.1.1",
        annotations=annotations,
    )


def test_handle_execute_round_trip():
    async def scenario():
        core = D11Core()
        core.register_route("S_1.1.1", _dummy_handler)

        payload = {
            "task_id": "1234",
            "task_type": "KPI",
            "input": {"region": "EU", "year": 2025},
            "context": {"priority": "high"},
            "crc": None,
        }

        response = await core.handle_execute(payload)
        assert response["status"] == "completed"
        assert response["task_id"] == payload["task_id"]
        assert "meta" in response
        assert response["meta"]["route"] == "S_1.1.1"

        status = await core.fetch_status(payload["task_id"])
        assert status is not None
        assert status["route"] == "S_1.1.1"

        metrics = await core.metrics_snapshot()
        assert metrics["sla"]["violations"] == 0

    asyncio.run(scenario())


def test_control_plane_throttle_and_reset():
    async def scenario():
        core = D11Core()
        core.register_route("S_1.1.1", _dummy_handler)

        await core.handle_control({"type": "THROTTLE", "value": 0.5})
        assert core.throttle.rate <= core.throttle.baseline

        await core.handle_control({"type": "RESET"})
        assert core.state.get_result("non-existent") is None
        assert core.throttle.rate == core.throttle.baseline

    asyncio.run(scenario())


def test_predictive_bias_and_safe_mode_controls():
    async def scenario():
        core = D11Core()
        core.register_route("S_1.1.1", _dummy_handler)

        payload = {
            "task_id": "1234",
            "task_type": "KPI",
            "input": {"region": "EU"},
        }

        for _ in range(3):
            await core.handle_execute({**payload, "task_id": f"{_}"})

        decision_before = core.predictor.decide("KPI")
        assert decision_before.metadata["bias"] == 0.0

        await core.handle_control({"type": "BOOST_PUSH", "task_type": "KPI", "factor": 0.5})
        decision_after = core.predictor.decide("KPI")
        assert decision_after.metadata["bias"] > decision_before.metadata["bias"]

        await core.handle_control({"type": "SAFE_MODE", "duration": 10})
        decision_safe = core.predictor.decide("KPI")
        assert not decision_safe.should_push
        assert decision_safe.reason == "safe_mode"

    asyncio.run(scenario())


def test_sla_breach_triggers_throttle_and_logging():
    async def scenario():
        core = D11Core()
        core.register_route("S_1.1.1", _dummy_handler)
        core.sla.threshold = 1e-6  # force violation with measured latency

        payload = {
            "task_id": "breach",
            "task_type": "KPI",
            "input": {"region": "EU"},
        }

        await core.handle_execute(payload)

        assert any(record["event"] == "SLA_Breach" for record in core.mrp.records)
        assert core.throttle.rate <= max(1, int(core.throttle.baseline * 0.5))

    asyncio.run(scenario())

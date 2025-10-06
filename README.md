# Toiac
Dlya testa cdex

## D₁.₁ Analytical Core Prototype

This repository now contains a reference implementation of the D₁.₁ child core
(`d1_1` package).  The module can be embedded into the transport module `M₁.₁`
by instantiating :class:`d1_1.D11Core`, registering execution route handlers, and
forwarding HTTP/2 task payloads to :meth:`D11Core.handle_execute`.

### Highlights

* **Semantic ingestion pipeline** – :class:`d1_1.components.InputParser` and
  :class:`d1_1.components.SemanticPlanner` normalise tasks, infer intent, and
  surface execution strategies for downstream executors.
* **Adaptive control-plane** – `THROTTLE`, `BOOST_PUSH`, `SAFE_MODE`,
  `RESTORE_RATE`, and `ANALYZE_LATENCY` commands are supported to manage the
  predictive layer, SLA behaviour, and concurrency envelope.
* **Predictive push heuristics** – the `PredictiveEngine` now maintains
  task-type specific bias, safe-mode windows, and multi-resource push bundles
  while logging SLA breaches through the MRP shim.
* **Extensive tests** – ``tests/test_core.py`` demonstrates task execution,
  control-plane flows, predictor bias adjustments, and SLA-triggered
  throttling.

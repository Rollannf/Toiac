"""CMV_1 : Core Matrix Vector Ψ-Resonant Module.

Полный модуль фазово-суммативной Ψ-архитектуры без использования
внешних математических библиотек. Все расчёты выполняются вручную,
исходя из заданных уравнений и итеративных операторов.
"""


class CMV_1:
    """Центральный модуль резонансного Ψ-поля.

    Реализует многомерно-вложенную архитектуру с двумя уровнями:
    базовые операторы (уровень 1) и функциональные Ψ-сервисы
    (уровень 2: SDE, AEC, SPS, IVG, ICE). Все вычисления ведутся
    при помощи собственных уравнений, без внешних библиотек.
    """

    def __init__(self):
        # -----------------------------
        #  Базовые параметры Ψ-ядра
        # -----------------------------
        self.Psi_0 = 1.0
        self.alpha = 0.003
        self.beta = 0.12
        self.xi = 0.85
        self.epsilon = 0.002
        self.S_n = 1.0
        self.rho_n = 0.9
        self.delta_theta = 0.2
        self.theta_0 = 0.25
        self.memory_coupling = 0.45
        self.t = 0.0
        self.base_dt = 0.1
        self.dt = self.base_dt
        self.phase_period = 6.283185307179586  # 2π

        # -----------------------------
        #  Конфигурация фрактального Ψ-поля
        # -----------------------------
        self.fractal_depth = 3
        self.fractal_branch_factor = 2
        self.max_fractal_iterations = 48
        self.fractal_state = self._build_fractal_state(
            level=self.fractal_depth,
            seed=self.Psi_0,
        )

        # -----------------------------
        #  Операторы уровня 1
        # -----------------------------
        self.operator_levels = {
            "phase_flux": self._phase_flux,
            "reverberation": self._reverberation,
            "coherence": self.coherence,
        }

        self.operator_sequence = ["phase_flux", "reverberation", "coherence"]

        # -----------------------------
        #  Мосты взаимодействия CMV₁ ↔ Mᵢ ↔ Cᵢ
        # -----------------------------
        self.M_i = {}
        self.C_i = {}
        self.monitor_frames = []

        # -----------------------------
        #  Инициализация MRP-модуля
        # -----------------------------
        self.MRP = self.MRP_Module(parent=self)
        self.logs = []

        # -----------------------------
        #  Контекст связи и состояние
        # -----------------------------
        self.context = self.PsiContext(phase_period=self.phase_period)
        self.frames = []
        self.last_plan_value = 0.0
        self.last_feedback = 0.0

    # ===============================================
    # ВНУТРЕННИЙ MRP — МОДУЛЬ РЕГИСТРАЦИИ ПРОЦЕССОВ
    # ===============================================
    class MRP_Module:
        """Структурированный регистратор процессов CMV₁."""

        def __init__(self, parent):
            self.parent = parent
            self.records = []
            self.max_records = 500
            self.counter = 0

        def log(self, process_name, data, t=None, metadata=None):
            self.counter += 1
            timestamp = self.parent.t if t is None else t
            entry = {
                "id": self.counter,
                "t": timestamp,
                "tag": process_name,
                "message": data,
            }
            if metadata is not None:
                entry["meta"] = metadata
            self.records.append(entry)
            if len(self.records) > self.max_records:
                self.records.pop(0)
            record = (
                f"[CMV1_LOG #{entry['id']:04d}] t={timestamp:.5f} "
                f"{process_name}: {data}"
            )
            print(record)

        def report(self):
            lines = []
            for entry in self.records:
                meta_suffix = ""
                if "meta" in entry:
                    meta_suffix = f" | meta={entry['meta']}"
                lines.append(
                    f"[{entry['id']:04d}] t={entry['t']:.5f} {entry['tag']}: "
                    f"{entry['message']}{meta_suffix}"
                )
            return "\n".join(lines)

        def snapshot(self):
            return [entry.copy() for entry in self.records]

    class PsiFrame:
        """Хранит слепок состояния Ψ-поля на шаге времени."""

        def __init__(self, t, psi, rho, gamma, plan, confidence, feedback):
            self.t = t
            self.psi = psi
            self.rho = rho
            self.gamma = gamma
            self.plan = plan
            self.confidence = confidence
            self.feedback = feedback

        def as_dict(self):
            return {
                "t": self.t,
                "psi": self.psi,
                "rho": self.rho,
                "gamma": self.gamma,
                "plan": self.plan,
                "confidence": self.confidence,
                "feedback": self.feedback,
            }

    class PsiContext:
        """Контекст синхронизации CMV₁ с другими модулями."""

        def __init__(self, phase_period):
            self.phase_period = phase_period
            self.link_cmv2 = None
            self.link_cmv3 = None
            self.feedback_queue = []

        def connect(self, cmv2=None, cmv3=None):
            if cmv2 is not None:
                self.link_cmv2 = cmv2
            if cmv3 is not None:
                self.link_cmv3 = cmv3

        def push_feedback(self, value):
            self.feedback_queue.append(value)

        def pull_feedback(self):
            if self.feedback_queue:
                return self.feedback_queue.pop(0)
            return None

        def emit_frame(self, frame):
            if self.link_cmv2 is not None:
                if hasattr(self.link_cmv2, "receive_from_cmv1"):
                    self.link_cmv2.receive_from_cmv1(frame)
                elif hasattr(self.link_cmv2, "ingest"):
                    payload = frame.as_dict()
                    self.link_cmv2.ingest(payload.get("psi", 0.0))
            if self.link_cmv3 is not None:
                if hasattr(self.link_cmv3, "ingest"):
                    self.link_cmv3.ingest(frame.psi, frame.rho)
                elif hasattr(self.link_cmv3, "receive_from_cmv1"):
                    self.link_cmv3.receive_from_cmv1(frame)

    # ===============================================
    # УРОВЕНЬ 1 — МАТЕМАТИКА И ОПЕРАТОРЫ Ψ-ПОЛЯ
    # ===============================================
    def Phi(self, t, log=False):
        """Фазовая функция Φ(t) — реверберационно-стереометрическая форма."""
        numerator = t * t
        denominator = 1 + 0.1 * t * t
        base = numerator / denominator
        modulation = 0.25 * self._sin(0.5 * t)
        fractal_factor = self._fractal_amplitude(self.fractal_state, t)
        normalized_fractal = self._limit_amplitude(fractal_factor, 2.0) * 0.05
        phi_value = base + modulation + normalized_fractal
        if log:
            self.MRP.log("Φ", f"t={t:.5f}, Φ={phi_value:.6f}")
        return phi_value

    def dPhi(self, t, dt=1e-3):
        """Первая производная Φ'(t)."""
        return (self.Phi(t + dt) - self.Phi(t - dt)) / (2 * dt)

    def d2Phi(self, t, dt=1e-3):
        """Вторая производная Φ''(t)."""
        return (self.Phi(t + dt) - 2 * self.Phi(t) + self.Phi(t - dt)) / (dt * dt)

    def Psi(self, t, log=True):
        """Фазово-суммативное Ψ(t) — центральное уравнение поля."""
        resonant = self._reverberation(t)
        phase_component = self._phase_flux(t)
        previous_psi = self._get_previous_psi()
        phi_value = self.Phi(t)
        retro_phase = self.beta * self._cos(phi_value)
        numerator = (
            self.Psi_0
            + (self.alpha * self.xi) / (1 + self.epsilon * self.S_n)
            + self.alpha * phi_value
            + retro_phase
            + self.rho_n * phase_component
            + self.memory_coupling * previous_psi
            + 0.15 * resonant
        )
        denominator = 1 + self.epsilon * (self._abs(self.d2Phi(t)) + self._abs(resonant))
        psi = numerator / denominator
        if log:
            self.MRP.log(
                "Ψ Calculation",
                f"t={t:.5f}, Ψ={psi:.6f}",
                metadata={"prev": previous_psi},
            )
        return psi

    def rho_Psi(self, t):
        """Плотность Ψ (градиентная величина)."""
        delta = 1e-3
        dp = (self.Psi(t + delta, log=False) - self.Psi(t - delta, log=False)) / (2 * delta)
        base_density = self._abs(dp)
        psi_value = self.Psi(t, log=False)
        normalized = base_density / (1 + self._abs(psi_value))
        density = normalized * (1 + 0.1 * self.rho_n)
        self.MRP.log("ρΨ", f"t={t:.5f}, ρΨ={density:.6f}")
        return density

    def coherence(self, t):
        """Когерентность фазы ΓΨ."""
        diff = self._abs(self.dPhi(t))
        cross_feedback = self._abs(self.last_feedback) * 0.1
        coherence_value = 1 / (1 + diff + self.epsilon + cross_feedback)
        self.MRP.log("ΓΨ", f"t={t:.5f}, ΓΨ={coherence_value:.6f}")
        return coherence_value

    # ===============================================
    # УРОВЕНЬ 2 — Ψ-СЕРВИСЫ
    # ===============================================
    def SDE(self, current_state, env_signals, policies):
        """Strategic Decision Engine."""
        self.MRP.log("SDE", "Strategic Decision Engine initiated")

        env_weight = 0.6
        policy_weight = 0.4
        aggregate_env = self._weighted_sum(env_signals, env_weight)
        aggregate_policies = self._weighted_sum(policies, policy_weight)
        combined = aggregate_env + aggregate_policies + 0.1 * self.last_plan_value
        normalized = self._normalize(combined)
        risk = self._resonant_risk(normalized, env_signals)
        confidence = self._clamp(1 - (risk * 0.5), 0.0, 1.0)
        plan_raw = (current_state * (1 + normalized)) / (1 + risk)
        plan_value = self._clamp(plan_raw, 0.0, 2.5)

        self.last_plan_value = plan_value
        plan = f"Ψ-plan[{plan_value:.5f}]"
        rationale = (
            "Aggregated="
            f"{combined:.5f}, Normalized={normalized:.5f}, Risk={risk:.5f}"
        )
        self.MRP.log(
            "SDE Result",
            f"Plan={plan}, Conf={confidence:.3f}",
            metadata={"risk": risk, "normalized": normalized},
        )
        return plan, rationale, confidence

    def AEC(self, plan, runtime_policies):
        """Adaptive Execution Coordinator."""
        self.MRP.log("AEC", "Adaptive Execution Coordinator started")
        t = self.t
        delay_factor = runtime_policies.get("delay_factor", 0.0)
        coherence_ref = runtime_policies.get("coherence_ref", 1.0)
        spectral = self._abs(self._sin(self.delta_theta * t + self.theta_0))
        latency = (spectral + delay_factor) / (1 + coherence_ref)
        latency += self._abs(self.last_feedback) * 0.1
        stability = 1 / (1 + latency)
        result = f"EXEC[{plan}]@{t:.2f} stability={stability:.4f}"
        self.MRP.log("AEC Result", result, metadata={"latency": latency})
        return {"status": "COMPLETED", "latency": latency, "stability": stability}

    def SPS(self, fn_ref):
        """Safe Probe and Sandbox."""
        self.MRP.log("SPS", f"Sandbox probing {fn_ref.__name__}")
        t = self.t
        probe_result = fn_ref(t)
        threshold = 5 + self._abs(self.last_feedback)
        safe = self._abs(probe_result) < threshold
        self.MRP.log(
            "SPS Result",
            f"{fn_ref.__name__} SAFE={safe}",
            metadata={"threshold": threshold},
        )
        return {"safe": safe, "value": probe_result}

    def IVG(self, artifact_id, version_current, version_target):
        """Integrity and Version Guardian."""
        self.MRP.log("IVG", f"Integrity check for {artifact_id}")
        diff = self._abs(version_target - version_current)
        sensitivity = 0.01
        normalized_diff = diff / sensitivity
        integrity = 1 / (1 + normalized_diff)
        signature = self._psi_signature()
        self.MRP.log(
            "IVG Result",
            f"Integrity={integrity:.4f}",
            metadata={"signature": signature},
        )
        return {"artifact": artifact_id, "integrity": integrity, "signature": signature}

    def ICE(self, operator_call, function_signature):
        """Inter-Level Contract Enforcer."""
        self.MRP.log("ICE", f"Contract enforcement for {operator_call}")
        if callable(function_signature):
            signature_name = function_signature.__name__
        else:
            signature_name = str(function_signature)
        match = operator_call == signature_name
        phase_alignment = self._cos(self.delta_theta * self.t + self.theta_0)
        enforced = match and phase_alignment > 0
        self.MRP.log(
            "ICE Result",
            f"Match={match}, Enforced={enforced}",
            metadata={"phase_alignment": phase_alignment},
        )
        return {"enforced": enforced, "match": match, "phase": phase_alignment}

    # ===============================================
    # СВЯЗЬ CMV₁ ↔ Mᵢ ↔ Cᵢ
    # ===============================================
    def bind_chain(self, m_label, c_label, payload):
        """Регистрация взаимодействия CMV₁ с внутренними узлами."""
        self.M_i[m_label] = payload
        self.C_i[c_label] = payload
        self.MRP.log(
            "CHAIN",
            f"Bound CMV₁ ↔ {m_label} ↔ {c_label} with payload="
            f"{self._format_payload(payload)}",
            metadata={"M": m_label, "C": c_label},
        )
        self._update_monitor(payload)

    def handshake(self, cmv2=None, cmv3=None):
        """Устанавливает резонансные связи с CMV₂ и CMV₃."""
        self.context.connect(cmv2=cmv2, cmv3=cmv3)
        self.MRP.log(
            "HANDSHAKE",
            "Context synchronized with external modules",
            metadata={
                "cmv2": bool(cmv2),
                "cmv3": bool(cmv3),
            },
        )

    def receive_feedback(self, value):
        """Приём обратной связи из CMV₂/CMV₃."""
        if value is None:
            return
        self.context.push_feedback(value)
        self.last_feedback = value
        self.MRP.log("FEEDBACK", f"Received external feedback={value:.6f}")

    def execute_operator(self, operator_name, *args, **kwargs):
        """Диспетчер вызова операторов уровня 1."""
        operator = self.operator_levels.get(operator_name)
        if operator is None:
            self.MRP.log("OPERATOR", f"Unknown operator {operator_name}")
            return None
        result = operator(*args, **kwargs)
        self.MRP.log(
            "OPERATOR",
            f"{operator_name} executed -> {result:.6f}" if result is not None else "None",
        )
        return result

    def _store_frame(self, t, psi, rho, gamma, plan, confidence, feedback):
        frame = self.PsiFrame(t, psi, rho, gamma, plan, confidence, feedback)
        self.frames.append(frame)
        self.monitor_frames.append(frame.as_dict())
        return frame

    def _emit_frame(self, frame):
        if frame is None:
            return
        self.context.emit_frame(frame)

    def _get_previous_psi(self):
        if self.frames:
            return self.frames[-1].psi
        return self.Psi_0

    def _adaptive_dt(self, gradient):
        modifier = 1 + self._abs(gradient)
        return self.base_dt / modifier

    def _collect_env_signals(self, rho_value, coherence_value, feedback):
        signals = [rho_value, coherence_value]
        if feedback is not None:
            signals.append(feedback)
        return signals

    def _update_monitor(self, payload):
        snapshot = {
            "t": self.t,
            "payload": payload,
        }
        self.monitor_frames.append(snapshot)

    def pull_feedback(self):
        feedback = self.context.pull_feedback()
        if feedback is not None:
            self.MRP.log("FEEDBACK", f"Pulled feedback={feedback:.6f}")
        return feedback

    # ===============================================
    # ЦИКЛ РАБОТЫ Ψ-КОНТУРА
    # ===============================================
    def run(self, steps=10, env_sequence=None):
        self.MRP.log("CMV1", "Ψ Resonant Loop Initiated", t=self.t)
        confidence = 1.0
        previous_psi = self._get_previous_psi()
        env_sequence = env_sequence or []
        for idx in range(steps):
            feedback = self.pull_feedback()
            if feedback is None:
                feedback = self.last_feedback
            else:
                self.last_feedback = feedback

            psi_value = self.Psi(self.t)
            rho_value = self.rho_Psi(self.t)
            coherence_value = self.coherence(self.t)
            self.MRP.log(
                "Ψ-state",
                f"step={idx}, t={self.t:.3f}, Ψ={psi_value:.6f}, "
                f"ρΨ={rho_value:.6f}, ΓΨ={coherence_value:.6f}",
            )

            env_inputs = self._collect_env_signals(rho_value, coherence_value, feedback)
            policy_inputs = [confidence, self.last_plan_value]
            if env_sequence:
                env_inputs.append(env_sequence[idx % len(env_sequence)])

            plan, rationale, confidence = self.SDE(
                psi_value,
                env_inputs,
                policy_inputs,
            )
            self.MRP.log("SDE Rationale", rationale)

            exec_report = self.AEC(
                plan,
                {
                    "delay_factor": 0.5,
                    "coherence_ref": coherence_value,
                },
            )
            sandbox_report = self.SPS(self.Psi)
            integrity_report = self.IVG("artifact_core", 1.0, 1.02)
            contract_report = self.ICE("operator_start", "operator_start")

            payload = {
                "ψ": psi_value,
                "ρψ": rho_value,
                "Γψ": coherence_value,
                "plan": plan,
                "confidence": confidence,
                "execution": exec_report,
                "sandbox": sandbox_report,
                "integrity": integrity_report,
                "contract": contract_report,
                "feedback": feedback,
            }
            self.bind_chain(f"M_{idx}", f"C_{idx}", payload)

            frame = self._store_frame(
                self.t,
                psi_value,
                rho_value,
                coherence_value,
                plan,
                confidence,
                feedback,
            )
            self._emit_frame(frame)

            gradient = (psi_value - previous_psi) / self.dt if idx > 0 else 0.0
            self.dt = self._adaptive_dt(gradient)
            previous_psi = psi_value
            self.t = (self.t + self.dt) % self.phase_period

        print("\n=== FINAL MRP LOG ===")
        print(self.MRP.report())

    # ===============================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ===============================================
    def _phase_flux(self, t):
        """Фазовый поток Ψ-поля."""
        phi_value = self.Phi(t)
        gradient = self.dPhi(t)
        flux = phi_value * (1 + 0.5 * gradient)
        return self._limit_amplitude(flux, 3.0)

    def _reverberation(self, t):
        """Реверберационно-резонансное вычисление."""
        base = self._sin(self.delta_theta * t + self.theta_0)
        echo = base
        attenuation = 0.55
        harmonics = 3
        for n in range(1, harmonics + 1):
            phase_shift = self.theta_0 * (n + 1)
            phase = (n + 1) * t * self.delta_theta + phase_shift
            echo += self._pow(attenuation, n) * self._sin(phase)
        return self._limit_amplitude(echo, 1.5)

    def _build_fractal_state(self, level, seed):
        """Рекурсивное построение фрактальной структуры Ψ-поля."""
        node = {
            "value": seed,
            "children": [],
            "level": level,
        }
        if level <= 1:
            return node
        for branch in range(self.fractal_branch_factor):
            modifier = (branch + 1) / self.fractal_branch_factor
            child_seed = seed * (0.7 + 0.1 * modifier)
            child = self._build_fractal_state(level - 1, child_seed)
            child["index"] = branch
            node["children"].append(child)
        return node

    def _fractal_amplitude(self, node, t, depth=0, counter=None):
        """Расчёт фрактальной амплитуды Ψ-поля."""
        if counter is None:
            counter = [0]
        counter[0] += 1
        if counter[0] > self.max_fractal_iterations:
            return 0.0
        attenuation = 0.9 ** depth
        amplitude = node["value"] * self._sin(t + node["value"] + depth * 0.1)
        total = amplitude * attenuation
        for child in node["children"]:
            total += 0.5 * self._fractal_amplitude(child, t * 1.1, depth + 1, counter)
        return total

    def _normalize(self, value):
        if value == 0:
            return 0.0
        magnitude = self._abs(value)
        normalized = magnitude / (1 + magnitude)
        return normalized if value > 0 else -normalized

    def _sum(self, sequence):
        total = 0.0
        for element in sequence:
            total += element
        return total

    def _pow(self, base, exponent):
        if exponent == 0:
            return 1.0
        if exponent < 0:
            return 1.0 / self._pow(base, -exponent)
        if self._abs(exponent - int(exponent)) < 1e-9:
            result = 1.0
            counter = 0
            exponent = int(exponent)
            while counter < exponent:
                result *= base
                counter += 1
            return result
        # Для дробных степеней используем exp(exponent * ln(base))
        return self._exp(exponent * self._ln(base))

    def _weighted_sum(self, sequence, weight):
        if not sequence:
            return 0.0
        scaled_weight = weight / len(sequence)
        total = 0.0
        for value in sequence:
            total += value * scaled_weight
        return total

    def _resonant_risk(self, normalized, env_signals):
        variance = 0.0
        mean = self._sum(env_signals) / (len(env_signals) or 1)
        for value in env_signals:
            diff = value - mean
            variance += diff * diff
        variance /= (len(env_signals) or 1)
        return self._abs(normalized) * 0.5 + variance * 0.1

    def _format_payload(self, payload):
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                items.append(f"{key}={{...}}")
            else:
                items.append(f"{key}={value}")
        return ", ".join(items)

    def _psi_signature(self):
        if not self.frames:
            base_value = self.Psi_0
        else:
            base_value = self.frames[-1].psi
        return self._sin(base_value + self.t)

    def _sin(self, x):
        """Синус, рассчитанный по ряду Тейлора до x^7 с нормализацией периода."""
        tau = 6.283185307179586
        while x > 3.141592653589793:
            x -= tau
        while x < -3.141592653589793:
            x += tau
        x2 = x * x
        x3 = x2 * x
        x5 = x3 * x2 * x2
        x7 = x5 * x2 * x2
        return x - (x3 / 6) + (x5 / 120) - (x7 / 5040)

    def _cos(self, x):
        return self._sin(1.5707963267948966 - x)

    def _exp(self, x):
        term = 1.0
        total = 1.0
        for n in range(1, 12):
            term *= x / n
            total += term
        return total

    def _ln(self, x):
        if x <= 0:
            raise ValueError("ln undefined for non-positive values")
        z = (x - 1) / (x + 1)
        z2 = z * z
        result = 0.0
        term = z
        denominator = 1
        for _ in range(6):
            result += term / denominator
            term *= z2
            denominator += 2
        return 2 * result

    def _limit_amplitude(self, value, limit):
        return self._clamp(value, -limit, limit)

    def _clamp(self, value, lower, upper):
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    def _abs(self, x):
        return x if x >= 0 else -x


if __name__ == "__main__":
    core = CMV_1()
    core.run(steps=5)

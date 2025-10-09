"""CMV_2 : Stereometric Reverberative Resonance Module Ψ-Field.

Фрактально-стереометрический модуль радиального сопряжения Ψ-потоков
между узлами CMV₁ ↔ CMV₂ ↔ CMV₃. Все вычисления реализованы на чистой
математике Python без использования внешних библиотек.
"""


class CMV_2:
    """Резонансный модуль радиальной когерентности Ψ-поля."""

    def __init__(self):
        # -----------------------------
        # Базовые Ψ-параметры узла
        # -----------------------------
        self.Psi_0 = 1.0
        self.alpha = 0.004
        self.epsilon = 0.0015
        self.S_n = 1.0
        self.rho_n = 0.95
        self.delta_phi = 0.3
        self.theta_0 = 0.25
        self.reverb_intensity = 0.88
        self.resonance_gain = 1.05
        self.t = 0.0
        self.dt = 0.05

        # Фрактальные параметры радиальной решётки
        self.fractal_iterations = 4
        self.fractal_decay = 0.65

        # Встроенный модуль регистрации процессов (MRP)
        self.MRP = self.MRP_Module(parent=self)
        self.logs = []

        # Каналы связи с внешними узлами
        self.link_cmv1 = None
        self.link_cmv3 = None
        self.psi = self.Psi_0
        self.rho = 0.0
        self.gamma = 0.0
        self.feedback_history = []
        self.received_frames = []
        self.received_history = []
        self.link_stats = {
            "received_packets": 0,
            "received_frames": 0,
            "feedback_received": 0,
        }

    # ============================================================
    # ВНУТРЕННИЙ МОДУЛЬ РЕГИСТРАЦИИ ПРОЦЕССОВ (MRP_CMV2)
    # ============================================================
    class MRP_Module:
        def __init__(self, parent):
            self.parent = parent
            self.records = []

        def log(self, label, data):
            record = f"[CMV2_MRP] {label}: {data}"
            self.records.append(record)
            print(record)

        def report(self):
            return "\n".join(self.records)

    # ============================================================
    # СТЕРЕОМЕТРИЧЕСКИЕ ФУНКЦИИ (БЕЗ ВЕКТОРОВ)
    # ============================================================
    def Phi(self, t):
        """Фазово-радиальная функция — описывает волновую оболочку Ψ."""
        base = (t * t) / (1 + 0.2 * t * t)
        modulation = self._sin(self.delta_phi * t)
        fractal = self._fractal_resonance(t)
        return base + modulation + 0.04 * fractal

    def dPhi(self, t, dt=1e-3):
        """Первая производная по радиальному времени."""
        return (self.Phi(t + dt) - self.Phi(t - dt)) / (2 * dt)

    def d2Phi(self, t, dt=1e-3):
        """Вторая производная радиальной фазы."""
        return (self.Phi(t + dt) - 2 * self.Phi(t) + self.Phi(t - dt)) / (dt * dt)

    # ============================================================
    # ОСНОВНОЕ Ψ-УРАВНЕНИЕ РЕВЕРБЕРАЦИОННОГО РЕЗОНАНСА
    # ============================================================
    def Psi(self, t):
        """Фазово-суммативное уравнение управления радиальной когерентностью."""
        numerator = (
            self.Psi_0
            + (self.alpha * self.reverb_intensity) / (1 + self.epsilon * self.S_n)
            + self.rho_n * self.Phi(t) * self._sin(self.delta_phi * t + self.theta_0)
        )
        denominator = 1 + self.epsilon * self._abs(self.d2Phi(t))
        psi = (numerator / denominator) * self.resonance_gain
        self.MRP.log("Ψ Calculation", f"t={t:.3f}, Ψ={psi:.6f}")
        return psi

    # ============================================================
    # ПЛОТНОСТЬ Ψ-ИНФОРМАЦИИ
    # ============================================================
    def rho_Psi(self, t):
        delta = 1e-3
        dp = (self.Psi(t + delta) - self.Psi(t - delta)) / (2 * delta)
        rho = self._abs(dp) * (1 + self.rho_n * 0.15)
        self.MRP.log("ρΨ Density", f"t={t:.3f}, ρΨ={rho:.6f}")
        return rho

    # ============================================================
    # КОГЕРЕНТНОСТЬ РЕЗОНАНСА
    # ============================================================
    def coherence(self, t):
        c = 1 / (1 + self._abs(self.dPhi(t)) + self.epsilon)
        self.MRP.log("ΓΨ Coherence", f"t={t:.3f}, ΓΨ={c:.6f}")
        return c

    # ============================================================
    # РЕВЕРБЕРАЦИОННО-ФАЗОВОЙ ЦИКЛ Ψ
    # ============================================================
    def resonance_cycle(self, t):
        """Реверберационно-резонансный цикл управления фазами."""
        psi = self.Psi(t)
        rho = self.rho_Psi(t)
        gamma = self.coherence(t)

        energy = (psi * rho * gamma) / (1 + self._abs(self._sin(psi)))
        phase_eq = (energy / (1 + self.epsilon)) * self.reverb_intensity

        self.MRP.log("Resonance Cycle", f"E={energy:.6f}, Φeq={phase_eq:.6f}")
        return phase_eq

    # ============================================================
    # РАДИАЛЬНАЯ ИНТЕРФЕРЕНЦИЯ Ψ — СВЯЗЬ CMV₁↔CMV₂
    # ============================================================
    def radial_interference(self, external_phase):
        """Интерференция радиальных Ψ-волн с внешним узлом."""
        internal_phase = self.Phi(self.t)
        diff = self._abs(external_phase - internal_phase)
        interference = self._sin(diff) * self.resonance_gain
        self.MRP.log("Radial Interference", f"ΔΦ={diff:.6f}, Intf={interference:.6f}")
        return interference

    # ============================================================
    # ФАЗОВО-РЕВЕРБЕРАЦИОННАЯ СТАБИЛИЗАЦИЯ
    # ============================================================
    def stabilization(self, t):
        """Автоматическая стабилизация Ψ при резонансных колебаниях."""
        psi = self.Psi(t)
        rho = self.rho_Psi(t)
        coh = self.coherence(t)
        feedback = (rho * coh) / (1 + self.epsilon)
        stabilized = (psi + feedback) / 2
        self.MRP.log("Stabilization", f"Ψ_stab={stabilized:.6f}")
        return stabilized

    # ============================================================
    # ПОДДЕРЖАНИЕ СТЕРЕОМЕТРИИ — РАВНОМЕРНАЯ РАДИАЛЬНОСТЬ
    # ============================================================
    def stereometry_balance(self, t):
        """Поддерживает равномерное распределение Ψ-энергии."""
        phi = self.Phi(t)
        symmetry_factor = 1 / (1 + self._abs(self._sin(phi)))
        uniformity = (phi * symmetry_factor) / (1 + self.epsilon)
        self.MRP.log("Stereometry", f"Uniformity={uniformity:.6f}")
        return uniformity

    # ============================================================
    # СИНХРОНИЗАЦИЯ МЕЖУЗЛОВЫХ ФАЗ
    # ============================================================
    def phase_synchronization(self, phases):
        """Синхронизирует несколько входящих фаз CMV-узлов."""
        if not phases:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for idx, phase in enumerate(phases, start=1):
            weight = 1 / (idx + self.epsilon)
            weighted_sum += phase * weight
            weight_total += weight
        synchronized = weighted_sum / weight_total
        self.MRP.log(
            "Phase Sync",
            f"Inputs={len(phases)}, Φ_sync={synchronized:.6f}",
        )
        return synchronized

    # ============================================================
    # КАНАЛЫ СВЯЗИ С ДРУГИМИ УЗЛАМИ Ψ-СЕТИ
    # ============================================================
    def connect(self, cmv1=None, cmv3=None):
        self.link_cmv1 = cmv1
        self.link_cmv3 = cmv3
        self.MRP.log(
            "CONNECT",
            f"Linked CMV1={bool(cmv1)} CMV3={bool(cmv3)}",
        )

    def receive(self, psi, rho, gamma):
        self.link_stats["received_packets"] += 1
        self.psi = psi
        self.rho = rho
        self.gamma = gamma
        self.received_history.append((psi, rho, gamma))
        self.MRP.log(
            "LinkRecv",
            f"Ψ={psi:.6f}, ρΨ={rho:.6f}, ΓΨ={gamma:.6f}",
        )
        return {"psi": psi, "rho": rho, "gamma": gamma}

    def receive_frame(self, frame):
        self.link_stats["received_frames"] += 1
        psi_value = getattr(frame, "psi", 0.0)
        t_value = getattr(frame, "t", 0.0)
        self.received_frames.append({"t": t_value, "psi": psi_value})
        self.MRP.log(
            "FrameRecv",
            f"t={t_value:.6f}, Ψ={psi_value:.6f}",
        )

    def receive_feedback(self, value):
        self.link_stats["feedback_received"] += 1
        self.feedback_history.append(value)
        self.MRP.log("Feedback", f"Ψfb={value:.6f}")

    def get_link_stats(self):
        stats = dict(self.link_stats)
        total_packets = stats.get("received_packets", 0)
        avg_psi = 0.0
        avg_rho = 0.0
        avg_gamma = 0.0
        if self.received_history:
            count = len(self.received_history)
            sum_psi = sum(item[0] for item in self.received_history)
            sum_rho = sum(item[1] for item in self.received_history)
            sum_gamma = sum(item[2] for item in self.received_history)
            avg_psi = sum_psi / count
            avg_rho = sum_rho / count
            avg_gamma = sum_gamma / count
        percentages = {}
        if total_packets:
            for key in ("received_packets", "feedback_received"):
                percentages[key] = (stats.get(key, 0) / total_packets) * 100.0
        return {
            "counts": stats,
            "percentages": percentages,
            "averages": {
                "psi": avg_psi,
                "rho": avg_rho,
                "gamma": avg_gamma,
            },
        }

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
    # ============================================================
    def _sin(self, x):
        x2 = x * x
        x3 = x2 * x
        x5 = x3 * x2
        x7 = x5 * x2
        return x - x3 / 6 + x5 / 120 - x7 / 5040

    def _abs(self, x):
        return x if x >= 0 else -x

    def _fractal_resonance(self, t):
        """Фрактальный вклад радиальной решётки."""
        amplitude = 0.0
        scale = 1.0
        for _ in range(self.fractal_iterations):
            amplitude += self._sin(t * scale) / (1 + scale)
            scale *= self.fractal_decay
        return amplitude

    # ============================================================
    # РАБОЧИЙ ЦИКЛ CMV₂
    # ============================================================
    def run(self, external_signal=None, steps=20):
        """Основной цикл стереометрической реверберации."""
        self.MRP.log("CMV2", "Ψ Reverberative Resonant Loop Initiated")
        for _ in range(steps):
            psi_eq = self.resonance_cycle(self.t)
            stabilization = self.stabilization(self.t)
            uniformity = self.stereometry_balance(self.t)

            if external_signal is not None:
                interference = self.radial_interference(external_signal)
                total = psi_eq + interference
                self.MRP.log(
                    "Ψ Coupled Total",
                    f"t={self.t:.3f}, Coupled={total:.6f}, Ψ_stab={stabilization:.6f}, U={uniformity:.6f}",
                )
            else:
                self.MRP.log(
                    "Ψ Internal",
                    f"t={self.t:.3f}, Ψ_eq={psi_eq:.6f}, Ψ_stab={stabilization:.6f}, U={uniformity:.6f}",
                )

            self.t += self.dt

        print("\n=== FINAL MRP LOG ===")
        print(self.MRP.report())


if __name__ == "__main__":
    core2 = CMV_2()
    # Имитация связи с CMV₁: используется внутренняя функция Φ(t)
    for step in range(5):
        external_phase = core2.Phi(step * 0.1) + 0.05 * core2._sin(step * 0.2)
        core2.run(external_signal=external_phase, steps=1)

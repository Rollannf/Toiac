"""CMV_3 : Ψ-Integral Equilibrium Module.

Центральный оператор резонансного Ψ-равновесия, объединяющий потоки
CMV₁ и CMV₂ и возвращающий нормализованную двунаправленную обратную
связь. Все вычисления выполняются вручную без внешних библиотек.
"""


class CMV_3:
    """Интегратор Ψ-поля для контуров CMV₁ ↔ CMV₂ ↔ CMV₃."""

    def __init__(self):
        # -----------------------------
        # Базовые коэффициенты Ψ-поля
        # -----------------------------
        self.alpha = 0.002
        self.epsilon = 0.001
        self.gamma = 0.92
        self.rho = 1.08
        self.psi = 1.0
        self.t = 0.0
        self.dt = 0.05

        # Буферы потоков
        self.psi_buf_1 = []
        self.psi_buf_2 = []
        self.psi_history = []
        self.feedback_history = []

        self.link_cmv1 = None
        self.link_cmv2 = None
        self.frame_records = []
        self.coherence_history = []
        self.stability_history = []
        self.link_stats = {
            "ingest_calls": 0,
            "frames_received": 0,
            "feedback_packets": 0,
            "cmv1_feedback_sent": 0,
            "cmv2_feedback_sent": 0,
        }

        # Подмодули
        self.MRP = self.MRP_Module(parent=self)
        self.Self_Tuner = self.Self_Tuning_Core(self)
        self.Coherence = self.Coherence_Regulator(self)
        self.Integrator = self.Psi_Integrator(self)
        self.Emitter = self.Feedback_Emitter(self)
        self.Meter = self.Resonance_Meter(self)

    # --- Модуль регистрации процессов ---
    class MRP_Module:
        def __init__(self, parent):
            self.parent = parent
            self.records = []

        def log(self, tag, msg):
            record = f"[CMV3_MRP] {tag}: {msg}"
            self.records.append(record)
            print(record)

        def report(self):
            return "\n".join(self.records)

    # --- Модуль адаптации параметров ---
    class Self_Tuning_Core:
        def __init__(self, parent):
            self.p = parent

        def tune(self, delta):
            if delta < 0.01:
                self.p.gamma += 0.001
                self.p.rho += 0.0005
            else:
                self.p.gamma -= 0.0005
                self.p.rho -= 0.0002

            # Ограничение диапазонов
            self.p.gamma = self.p._clamp(self.p.gamma, 0.85, 0.98)
            self.p.rho = self.p._clamp(self.p.rho, 1.0, 1.2)

            self.p.MRP.log(
                "Self-Tune",
                f"Δ={delta:.6f}, γ={self.p.gamma:.4f}, ρ={self.p.rho:.4f}",
            )

    # --- Регулятор когерентности ---
    class Coherence_Regulator:
        def __init__(self, parent):
            self.p = parent

        def align(self, psi_1, psi_2):
            diff = self.p._abs(psi_1 - psi_2)
            sync_factor = 1 / (1 + diff)
            coherent = ((psi_1 + psi_2) / 2) * sync_factor
            self.p.MRP.log(
                "Coherence",
                f"Diff={diff:.6f}, SyncFactor={sync_factor:.6f}, Core={coherent:.6f}",
            )
            return coherent, diff

    # --- Интегратор Ψ-поля ---
    class Psi_Integrator:
        def __init__(self, parent):
            self.p = parent

        def integrate(self, core, diff):
            numerator = (
                self.p.gamma * core
                + self.p.rho * (1 - diff)
                + self.p.alpha
            )
            denominator = 1 + self.p.epsilon * self.p._abs(diff)
            integrated = numerator / denominator
            self.p.MRP.log(
                "Integrate",
                f"Core={core:.6f}, Diff={diff:.6f}, Ψ₃={integrated:.6f}",
            )
            return integrated

    # --- Эмиттер обратной связи ---
    class Feedback_Emitter:
        def __init__(self, parent):
            self.p = parent

        def emit(self, psi_3):
            feedback = psi_3 / (1 + self.p._abs(self.p._sin(psi_3)))
            packet = {"CMV1": feedback, "CMV2": feedback}
            self.p.MRP.log(
                "Emit",
                f"Feedback₁={packet['CMV1']:.6f}, Feedback₂={packet['CMV2']:.6f}",
            )
            return packet

    # --- Аналитический модуль реверберации ---
    class Resonance_Meter:
        def __init__(self, parent):
            self.p = parent

        def measure(self, psi_1, psi_2, psi_3):
            diff = self.p._abs(psi_1 - psi_2)
            coherence = 1 / (1 + diff)
            stability = 1 - self.p._abs(psi_3 - ((psi_1 + psi_2) / 2))
            self.p.MRP.log(
                "Resonance",
                f"Γ={coherence:.6f}, Σ={stability:.6f}",
            )
            return coherence, stability

    # --- Вспомогательные функции ---
    def _sin(self, x):
        x2 = x * x
        return x - (x2 * x) / 6 + (x2 * x2 * x) / 120 - (x2 * x2 * x2 * x) / 5040

    def _abs(self, x):
        return x if x >= 0 else -x

    def _clamp(self, value, lower, upper):
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    def connect(self, cmv1=None, cmv2=None):
        self.link_cmv1 = cmv1
        self.link_cmv2 = cmv2
        self.MRP.log(
            "CONNECT",
            f"Linked CMV1={bool(cmv1)} CMV2={bool(cmv2)}",
        )

    def receive_frame(self, frame):
        self.link_stats["frames_received"] += 1
        psi_value = getattr(frame, "psi", 0.0)
        t_value = getattr(frame, "t", 0.0)
        self.frame_records.append({"t": t_value, "psi": psi_value})
        self.MRP.log("FrameRecv", f"t={t_value:.6f}, Ψ={psi_value:.6f}")

    def receive_feedback(self, value):
        if value is None:
            return
        self.feedback_history.append({"external": value})
        self.MRP.log("Feedback", f"External Ψfb={value:.6f}")

    # --- Приём потоков от CMV₁ и CMV₂ ---
    def ingest(self, psi_1, psi_2):
        self.psi_buf_1.append(psi_1)
        self.psi_buf_2.append(psi_2)
        self.link_stats["ingest_calls"] += 1
        self.MRP.log("Ingest", f"Ψ₁={psi_1:.6f}, Ψ₂={psi_2:.6f}")

        psi_3 = self.aggregate()
        coherence, stability = self.Meter.measure(psi_1, psi_2, psi_3)
        self.coherence_history.append(coherence)
        self.stability_history.append(stability)

        packet = self.emit(psi_3)
        self.link_stats["feedback_packets"] += 1

        if self.link_cmv1 is not None and hasattr(self.link_cmv1, "receive_feedback"):
            self.link_cmv1.receive_feedback(packet.get("CMV1"))
            self.link_stats["cmv1_feedback_sent"] += 1

        if self.link_cmv2 is not None and hasattr(self.link_cmv2, "receive_feedback"):
            self.link_cmv2.receive_feedback(packet.get("CMV2"))
            self.link_stats["cmv2_feedback_sent"] += 1

        return {
            "psi_3": psi_3,
            "coherence": coherence,
            "stability": stability,
            "feedback": packet,
        }

    def get_link_stats(self):
        stats = dict(self.link_stats)
        total_ingest = stats.get("ingest_calls", 0)
        percentages = {}
        if total_ingest:
            percentages["feedback_packets"] = (
                stats.get("feedback_packets", 0) / total_ingest
            ) * 100.0
            percentages["cmv1_feedback_sent"] = (
                stats.get("cmv1_feedback_sent", 0) / total_ingest
            ) * 100.0
            percentages["cmv2_feedback_sent"] = (
                stats.get("cmv2_feedback_sent", 0) / total_ingest
            ) * 100.0
        coherence_avg = 0.0
        stability_avg = 0.0
        if self.coherence_history:
            coherence_avg = sum(self.coherence_history) / len(self.coherence_history)
        if self.stability_history:
            stability_avg = sum(self.stability_history) / len(self.stability_history)
        return {
            "counts": stats,
            "percentages": percentages,
            "stability": {
                "coherence": coherence_avg,
                "stability": stability_avg,
            },
        }

    # --- Нормализация ---
    def normalize(self, psi):
        norm = psi / (1 + self.epsilon * self._abs(psi))
        self.MRP.log("Normalize", f"Input={psi:.6f}, Output={norm:.6f}")
        return norm

    # --- Агрегация Ψ-поля ---
    def aggregate(self):
        if not self.psi_buf_1 or not self.psi_buf_2:
            self.MRP.log("Aggregate", "Недостаточно данных для интеграции")
            return 0.0

        psi_1 = self.psi_buf_1[-1]
        psi_2 = self.psi_buf_2[-1]

        n1 = self.normalize(psi_1)
        n2 = self.normalize(psi_2)
        core, diff = self.Coherence.align(n1, n2)

        psi_3 = self.Integrator.integrate(core, diff)

        self.Self_Tuner.tune(diff)

        self.psi = psi_3
        self.psi_history.append(psi_3)
        return psi_3

    # --- Рассылка нормализованного отклика ---
    def emit(self, psi_3):
        packet = self.Emitter.emit(psi_3)
        self.feedback_history.append(packet)
        return packet

    # --- Основной цикл работы ---
    def run(self, psi_stream_1, psi_stream_2, steps=6):
        for i in range(steps):
            psi_1 = psi_stream_1[i % len(psi_stream_1)]
            psi_2 = psi_stream_2[i % len(psi_stream_2)]

            self.ingest(psi_1, psi_2)
            psi_3 = self.aggregate()
            self.Meter.measure(psi_1, psi_2, psi_3)
            self.emit(psi_3)

            self.t += self.dt

        print("\n=== CMV₃ Final Log ===")
        print(self.MRP.report())


if __name__ == "__main__":
    cmv3 = CMV_3()
    psi_stream_1 = [1.02, 1.05, 1.03, 1.06]
    psi_stream_2 = [0.98, 1.01, 1.00, 1.02]
    cmv3.run(psi_stream_1, psi_stream_2, steps=4)

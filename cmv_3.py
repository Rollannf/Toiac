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

    # --- Приём потоков от CMV₁ и CMV₂ ---
    def ingest(self, psi_1, psi_2):
        self.psi_buf_1.append(psi_1)
        self.psi_buf_2.append(psi_2)
        self.MRP.log("Ingest", f"Ψ₁={psi_1:.6f}, Ψ₂={psi_2:.6f}")

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

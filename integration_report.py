"""Интеграционный отчёт по взаимодействию модулей CMV₁↔CMV₂↔CMV₃.

Скрипт запускает согласованный цикл CMV₁, фиксируя обмены между
модулями, и выводит сводку по каналам связи в процентах, а также
усреднённые показатели устойчивости.
"""

from CMV_1 import CMV_1
from cmv_2 import CMV_2
from cmv_3 import CMV_3


def _percent(part, whole):
    if whole == 0:
        return 0.0
    return (part / whole) * 100.0


def run_integration(steps=12):
    cmv1 = CMV_1()
    cmv2 = CMV_2()
    cmv3 = CMV_3()

    cmv1.handshake(cmv2=cmv2, cmv3=cmv3)

    cmv1.run(steps=steps, cmv2=cmv2, cmv3=cmv3)

    stats1 = cmv1.get_link_metrics()
    stats2 = cmv2.get_link_stats()
    stats3 = cmv3.get_link_stats()

    steps_done = stats1["counts"].get("steps", steps)

    channel_rows = [
        (
            "CMV₁→CMV₂ Ψ/ρΨ/ΓΨ",
            stats1["counts"].get("psi_packets", 0),
            _percent(stats1["counts"].get("psi_packets", 0), steps_done),
        ),
        (
            "CMV₁→CMV₃ ingest",
            stats1["counts"].get("cmv3_ingest", 0),
            _percent(stats1["counts"].get("cmv3_ingest", 0), steps_done),
        ),
        (
            "Фреймы CMV₁→CMV₂",
            stats1["counts"].get("frames_to_cmv2", 0),
            _percent(stats1["counts"].get("frames_to_cmv2", 0), steps_done),
        ),
        (
            "Фреймы CMV₁→CMV₃",
            stats1["counts"].get("frames_to_cmv3", 0),
            _percent(stats1["counts"].get("frames_to_cmv3", 0), steps_done),
        ),
        (
            "Обратная связь CMV₃→CMV₁",
            stats1["counts"].get("feedback_received", 0),
            _percent(stats1["counts"].get("feedback_received", 0), steps_done),
        ),
        (
            "Обратная связь CMV₃→CMV₂",
            stats3["counts"].get("cmv2_feedback_sent", 0),
            _percent(stats3["counts"].get("cmv2_feedback_sent", 0), steps_done),
        ),
    ]

    stability_cmv1 = stats1.get("stability", {})
    stability_cmv3 = stats3.get("stability", {})
    averages_cmv2 = stats2.get("averages", {})

    report = {
        "channels": channel_rows,
        "handshake": {
            "cmv2": bool(stats1["counts"].get("handshake_cmv2")),
            "cmv3": bool(stats1["counts"].get("handshake_cmv3")),
        },
        "stability": {
            "cmv1_mean_coherence_percent": stability_cmv1.get(
                "mean_coherence_percent", 0.0
            ),
            "cmv1_mean_rho": stability_cmv1.get("mean_rho", 0.0),
            "cmv3_coherence_percent": stability_cmv3.get("coherence", 0.0) * 100.0,
            "cmv3_stability_percent": stability_cmv3.get("stability", 0.0) * 100.0,
            "cmv2_avg_gamma_percent": averages_cmv2.get("gamma", 0.0) * 100.0,
        },
        "raw": {
            "cmv1": stats1,
            "cmv2": stats2,
            "cmv3": stats3,
        },
        "steps": steps_done,
    }

    return report


def print_report(report):
    steps = report.get("steps", 0)
    print("=== Ψ-Integration Diagnostic Report ===")
    print(f"Шагов цикла: {steps}")
    print("Рукопожатия: CMV₂={0}, CMV₃={1}".format(
        "OK" if report["handshake"]["cmv2"] else "—",
        "OK" if report["handshake"]["cmv3"] else "—",
    ))
    print("\nКаналы передачи:")
    for label, count, percent in report["channels"]:
        print(f"  - {label:<24} : {count:>3} событий | {percent:6.2f}%")

    stability = report.get("stability", {})
    print("\nУстойчивость:")
    print(
        "  - Γ̄(CMV₁) = {0:6.2f}% | ρ̄(CMV₁) = {1:.4f}".format(
            stability.get("cmv1_mean_coherence_percent", 0.0),
            stability.get("cmv1_mean_rho", 0.0),
        )
    )
    print(
        "  - Γ̄(CMV₃) = {0:6.2f}% | Σ̄(CMV₃) = {1:6.2f}%".format(
            stability.get("cmv3_coherence_percent", 0.0),
            stability.get("cmv3_stability_percent", 0.0),
        )
    )
    print(
        "  - Γ̄(CMV₂) = {0:6.2f}%".format(
            stability.get("cmv2_avg_gamma_percent", 0.0)
        )
    )


if __name__ == "__main__":
    report = run_integration(steps=8)
    print_report(report)

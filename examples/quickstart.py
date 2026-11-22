"""Minimal end-to-end quickstart for ``rtmix``."""

from __future__ import annotations

from rtmix import (
    audit_dual_tail_forensics,
    compute_auroc,
    evaluate_ability_recovery,
    fit_classical_irt,
    fit_dtmix_irt,
    fit_vdl_speed_accuracy,
    load_benchmark_dataset,
    verify_numerical_equivalence,
)


def main() -> None:
    """Run bundled benchmark audit, baseline contrast, and numerical parity check."""
    data = load_benchmark_dataset()
    res = fit_dtmix_irt(
        data["Y"],
        data["log_rt"],
        data["entropy"],
        guessing=0.25,
        device="auto",
        compute_ci=True,
    )
    audit = audit_dual_tail_forensics(data["Y"], data["log_rt"], data["entropy"], res)
    irt = fit_classical_irt(data["Y"], guessing=0.25)
    vdl = fit_vdl_speed_accuracy(data["Y"], data["log_rt"], guessing=0.25)

    g = data["group_label"]
    sc_ai = (g == 1) | (g == 3)
    y_ai = (g[sc_ai] == 3).astype(int)

    print(f"Backend: {res.device_used} | Fit time: {res.fit_seconds:.3f}s")
    print(f"AI vs Slow-Careful AUROC -> DT-MixIRT: {compute_auroc(y_ai, res.pi_ai[sc_ai]):.4f} | vdL RT: {compute_auroc(y_ai, vdl.ai_score[sc_ai]):.4f}")
    print(f"Ability RMSE -> DT-MixIRT: {evaluate_ability_recovery(data['theta_true'], res.theta)['rmse']:.4f} | 3PL IRT: {evaluate_ability_recovery(data['theta_true'], irt.theta)['rmse']:.4f}")
    print(f"Flagged examinees -> RG: {int(audit.flagged_rg.sum())}, AI: {int(audit.flagged_ai.sum())}")

    eq = verify_numerical_equivalence()
    assert eq.all_passed
    print(f"Numerical parity passed (max M-step diff = {eq.max_abs_diff_theta_mstep:.2e}, speedup = {eq.speedup:.1f}x)")


if __name__ == "__main__":
    main()

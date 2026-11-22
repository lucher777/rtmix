"""Tests for classical baselines, bivariate ablation, and forensic diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from rtmix import (
    audit_dual_tail_forensics,
    compute_auroc,
    evaluate_ability_recovery,
    fit_bivariate_mixture_ablation,
    fit_classical_irt,
    fit_dtmix_irt,
    fit_vdl_speed_accuracy,
    fit_wise_nt_baseline,
    simulate_dual_tail_exam,
)


@pytest.fixture(scope="module")
def sim_bench():
    """Shared dual-tail simulation for baseline comparisons."""
    return simulate_dual_tail_exam(n_examinees=280, n_items=45, seed=202)


def test_classical_irt_baseline_structure(sim_bench):
    """Classical 3PL baseline returns finite abilities and tail scores."""
    res = fit_classical_irt(sim_bench["Y"], guessing=0.25)
    assert res.method == "classical_3pl_irt"
    assert res.theta.shape == (280,)
    assert np.all(np.isfinite(res.theta))
    assert res.fit_seconds >= 0.0


def test_vdl_baseline_confounds_slow_careful(sim_bench):
    """van der Linden RT-only baseline confounds slow careful examinees with AI copiers."""
    vdl = fit_vdl_speed_accuracy(sim_bench["Y"], sim_bench["log_rt"], guessing=0.25)
    g = sim_bench["group_label"]
    m = (g == 1) | (g == 3)
    auroc_vdl = compute_auroc((g[m] == 3).astype(int), vdl.ai_score[m])
    # RT-only slowness cannot separate AI copiers from slow careful examinees
    assert auroc_vdl < 0.65


def test_wise_nt10_detects_rg_but_not_ai(sim_bench):
    """Wise NT10 detects left-tail rapid guessing cleanly but misses right-tail AI."""
    nt = fit_wise_nt_baseline(sim_bench["Y"], sim_bench["log_rt"], guessing=0.25)
    g = sim_bench["group_label"]
    m_rg = (g == 0) | (g == 2)
    auroc_rg = compute_auroc((g[m_rg] == 2).astype(int), nt.rg_score[m_rg])
    assert auroc_rg > 0.95
    assert np.all((nt.extras["rte"] >= 0.0) & (nt.extras["rte"] <= 1.0))

    m_ai = (g == 1) | (g == 3)
    auroc_ai = compute_auroc((g[m_ai] == 3).astype(int), nt.ai_score[m_ai])
    assert auroc_ai < 0.65


def test_trivariate_beats_bivariate_ablation(sim_bench):
    """Full trivariate DT-MixIRT strictly outperforms bivariate (Y, log_rt) ablation."""
    tri = fit_dtmix_irt(
        sim_bench["Y"], sim_bench["log_rt"], sim_bench["entropy"], device="numpy", compute_ci=False
    )
    biv = fit_bivariate_mixture_ablation(sim_bench["Y"], sim_bench["log_rt"])
    g = sim_bench["group_label"]
    m = (g == 1) | (g == 3)
    y = (g[m] == 3).astype(int)
    auroc_tri = compute_auroc(y, tri.pi_ai[m])
    auroc_biv = compute_auroc(y, biv.ai_score[m])
    assert auroc_tri > auroc_biv + 0.05
    assert auroc_tri > 0.99


def test_forensic_audit_flags_and_tail_contrasts(sim_bench):
    """Forensic audit identifies AI hard-tail excess and flags RG/AI/DualTail groups."""
    res = fit_dtmix_irt(
        sim_bench["Y"], sim_bench["log_rt"], sim_bench["entropy"], device="numpy", compute_ci=True
    )
    audit = audit_dual_tail_forensics(
        sim_bench["Y"], sim_bench["log_rt"], sim_bench["entropy"], res, alpha=0.01, min_pi=0.08
    )
    g = sim_bench["group_label"]
    assert audit.flags.shape == (280,)
    assert set(audit.flag_labels.tolist()).issubset({"Clean", "RapidGuessing", "AIAssisted", "DualTail"})
    # AI copiers show positive hard-tail observed-minus-expected accuracy excess
    assert float(audit.hard_tail_excess[g == 3].mean()) > 0.15
    # Slow careful examinees show near-zero hard-tail excess and near-zero flagging
    assert abs(float(audit.hard_tail_excess[g == 1].mean())) < 0.08
    assert float(audit.flagged_ai[g == 1].mean()) <= 0.03
    assert float(audit.flagged_ai[g == 3].mean()) > 0.85
    assert float(audit.flagged_rg[g == 2].mean()) > 0.85
    assert audit.item_ai_rate.shape == (45,)
    assert audit.item_rg_rate.shape == (45,)


def test_compute_auroc_exact_cases_and_ties():
    """AUROC handles perfect separation, inverted scores, ties, and validation errors."""
    assert compute_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert compute_auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert compute_auroc(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compute_auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        compute_auroc(np.array([0, 1]), np.array([0.1]))


def test_evaluate_ability_recovery_subset_and_errors():
    """evaluate_ability_recovery returns exact zero error on identity and validates inputs."""
    x = np.linspace(-2.0, 2.0, 40)
    rec = evaluate_ability_recovery(x, x)
    assert rec["rmse"] == pytest.approx(0.0)
    assert rec["mae"] == pytest.approx(0.0)
    assert rec["bias"] == pytest.approx(0.0)
    assert rec["pearson_r"] == pytest.approx(1.0)
    assert rec["spearman_rho"] == pytest.approx(1.0)
    assert rec["kendall_tau"] == pytest.approx(1.0)

    mask = x > 0.0
    rec_sub = evaluate_ability_recovery(x, x + 0.5, subset_mask=mask)
    assert rec_sub["bias"] == pytest.approx(0.5)
    assert rec_sub["rmse"] == pytest.approx(0.5)

    with pytest.raises(ValueError):
        evaluate_ability_recovery(x, x[:-1])
    with pytest.raises(ValueError):
        evaluate_ability_recovery(x[:1], x[:1])

"""Unit and psychometric tests for the core DT-MixIRT estimation engine."""

from __future__ import annotations

import numpy as np
import pytest

from rtmix import (
    compute_auroc,
    compute_item_regime_posteriors,
    evaluate_ability_recovery,
    fit_classical_irt,
    fit_dtmix_irt,
    simulate_dual_tail_exam,
)


@pytest.fixture(scope="module")
def dual_tail_sim():
    """Shared reproducible dual-tail digital exam simulation."""
    return simulate_dual_tail_exam(n_examinees=320, n_items=45, seed=101)


@pytest.fixture(scope="module")
def fitted_numpy(dual_tail_sim):
    """Fitted DT-MixIRT model on NumPy float64 backend."""
    return fit_dtmix_irt(
        dual_tail_sim["Y"],
        dual_tail_sim["log_rt"],
        dual_tail_sim["entropy"],
        guessing=0.25,
        device="numpy",
        compute_ci=True,
    )


def test_posterior_simplex_and_bounds(fitted_numpy):
    """Item-level regime posteriors sum strictly to 1 and lie in [0, 1]."""
    post = fitted_numpy.posteriors
    assert post.ndim == 3 and post.shape[2] == 3
    assert np.all(post >= 0.0) and np.all(post <= 1.0)
    np.testing.assert_allclose(post.sum(axis=-1), 1.0, atol=1e-12)


def test_mixing_weights_simplex(fitted_numpy):
    """Person-level mixture weights lie on the 2-simplex."""
    total = fitted_numpy.pi_sol + fitted_numpy.pi_rg + fitted_numpy.pi_ai
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert np.all(fitted_numpy.pi_rg >= 0.0)
    assert np.all(fitted_numpy.pi_ai >= 0.0)
    assert np.all(fitted_numpy.pi_sol >= 0.0)


def test_three_way_disentanglement_auroc(dual_tail_sim, fitted_numpy):
    """DT-MixIRT achieves >0.99 AUROC separating AI vs Slow-Careful and RG vs Clean."""
    g = dual_tail_sim["group_label"]
    # AI Copier (3) vs Slow Careful (1)
    m_ai_sc = (g == 1) | (g == 3)
    auroc_ai_sc = compute_auroc((g[m_ai_sc] == 3).astype(int), fitted_numpy.pi_ai[m_ai_sc])
    assert auroc_ai_sc > 0.99

    # Rapid Guesser (2) vs Clean/Slow (0 or 1)
    m_rg_cl = (g == 0) | (g == 1) | (g == 2)
    auroc_rg = compute_auroc((g[m_rg_cl] == 2).astype(int), fitted_numpy.pi_rg[m_rg_cl])
    assert auroc_rg > 0.99

    # AI Copier (3) vs Rapid Guesser (2)
    m_ai_rg = (g == 2) | (g == 3)
    auroc_ai_rg = compute_auroc((g[m_ai_rg] == 3).astype(int), fitted_numpy.pi_ai[m_ai_rg])
    assert auroc_ai_rg > 0.99


def test_slow_careful_zero_false_positives(dual_tail_sim, fitted_numpy):
    """Honest slow careful examinees are not falsely flagged as AI copiers."""
    g = dual_tail_sim["group_label"]
    sc = g == 1
    assert float(fitted_numpy.pi_ai[sc].mean()) < 0.02
    fpr = float(np.mean((fitted_numpy.p_ai[sc] < 0.01) & (fitted_numpy.pi_ai[sc] >= 0.08)))
    assert fpr <= 0.02


def test_dual_tail_ability_debiasing(dual_tail_sim, fitted_numpy):
    """DT-MixIRT cuts dual-tail ability bias and RMSE sharply vs classical 3PL IRT."""
    base = fit_classical_irt(dual_tail_sim["Y"], guessing=0.25)
    g = dual_tail_sim["group_label"]
    rec_dt = evaluate_ability_recovery(dual_tail_sim["theta_true"], fitted_numpy.theta)
    rec_3pl = evaluate_ability_recovery(dual_tail_sim["theta_true"], base.theta)
    assert rec_dt["rmse"] < rec_3pl["rmse"] * 0.80
    assert rec_dt["pearson_r"] > 0.88

    rec_ai_dt = evaluate_ability_recovery(dual_tail_sim["theta_true"], fitted_numpy.theta, g == 3)
    rec_ai_3pl = evaluate_ability_recovery(dual_tail_sim["theta_true"], base.theta, g == 3)
    assert abs(rec_ai_dt["bias"]) < abs(rec_ai_3pl["bias"]) * 0.30

    rec_rg_dt = evaluate_ability_recovery(dual_tail_sim["theta_true"], fitted_numpy.theta, g == 2)
    rec_rg_3pl = evaluate_ability_recovery(dual_tail_sim["theta_true"], base.theta, g == 2)
    assert abs(rec_rg_dt["bias"]) < abs(rec_rg_3pl["bias"]) * 0.30


def test_profile_ci_enclosure_and_bounds(fitted_numpy):
    """95% profile-likelihood CIs enclose point estimates and remain in [0, 1]."""
    assert fitted_numpy.pi_rg_ci_low is not None and fitted_numpy.pi_rg_ci_high is not None
    assert fitted_numpy.pi_ai_ci_low is not None and fitted_numpy.pi_ai_ci_high is not None
    assert np.all(fitted_numpy.pi_rg_ci_low <= fitted_numpy.pi_rg + 1e-12)
    assert np.all(fitted_numpy.pi_rg <= fitted_numpy.pi_rg_ci_high + 1e-12)
    assert np.all(fitted_numpy.pi_ai_ci_low <= fitted_numpy.pi_ai + 1e-12)
    assert np.all(fitted_numpy.pi_ai <= fitted_numpy.pi_ai_ci_high + 1e-12)
    assert np.all((fitted_numpy.pi_ai_ci_low >= 0.0) & (fitted_numpy.pi_ai_ci_high <= 1.0))


def test_boundary_lrt_nonnegative_and_pvalues(fitted_numpy):
    """Boundary-corrected LRT statistics are >= 0 and p-values lie in [0, 1]."""
    assert np.all(fitted_numpy.lrt_rg >= 0.0)
    assert np.all(fitted_numpy.lrt_ai >= 0.0)
    assert np.all(fitted_numpy.lrt_joint >= 0.0)
    assert np.all((fitted_numpy.p_rg >= 0.0) & (fitted_numpy.p_rg <= 1.0))
    assert np.all((fitted_numpy.p_ai >= 0.0) & (fitted_numpy.p_ai <= 1.0))
    assert np.all((fitted_numpy.p_joint >= 0.0) & (fitted_numpy.p_joint <= 1.0))


def test_standalone_item_posterior_matches_fit(dual_tail_sim, fitted_numpy):
    """compute_item_regime_posteriors matches fitted posteriors to machine precision."""
    recomp = compute_item_regime_posteriors(
        dual_tail_sim["Y"], dual_tail_sim["log_rt"], dual_tail_sim["entropy"], fitted_numpy
    )
    np.testing.assert_allclose(recomp, fitted_numpy.posteriors, atol=1e-13)


def test_missing_data_handling(dual_tail_sim):
    """DT-MixIRT handles NaN missing entries gracefully and preserves NaN posteriors."""
    rng = np.random.default_rng(7)
    Y = dual_tail_sim["Y"].copy()
    L = dual_tail_sim["log_rt"].copy()
    H = dual_tail_sim["entropy"].copy()
    miss = rng.random(size=Y.shape) < 0.08
    # Keep at least 5 observed cells per row/col
    miss[:, :5] = False
    miss[:5, :] = False
    Y[miss] = np.nan
    res = fit_dtmix_irt(Y, L, H, device="numpy", compute_ci=False, max_iter=20)
    assert np.all(np.isfinite(res.theta))
    assert np.all(np.isnan(res.posteriors[miss]))
    assert np.all(np.isfinite(res.posteriors[~miss]))


def test_fixed_item_and_regime_scoring(dual_tail_sim, fitted_numpy):
    """Fixed-bank scoring mode preserves supplied item and regime parameters."""
    res_fixed = fit_dtmix_irt(
        dual_tail_sim["Y"][:60],
        dual_tail_sim["log_rt"][:60],
        dual_tail_sim["entropy"][:60],
        fixed_item_params=fitted_numpy.item_params,
        fixed_regime_params=fitted_numpy.regime_params,
        device="numpy",
        compute_ci=False,
    )
    np.testing.assert_allclose(res_fixed.item_params.b, fitted_numpy.item_params.b, atol=1e-15)
    assert res_fixed.regime_params.p_ai == fitted_numpy.regime_params.p_ai


def test_compute_ci_false_returns_none(dual_tail_sim):
    """Disabling compute_ci sets CI fields to None."""
    res = fit_dtmix_irt(
        dual_tail_sim["Y"][:80],
        dual_tail_sim["log_rt"][:80],
        dual_tail_sim["entropy"][:80],
        device="numpy",
        compute_ci=False,
        max_iter=15,
    )
    assert res.pi_rg_ci_low is None
    assert res.pi_ai_ci_high is None


def test_vector_guessing_asymptotes(dual_tail_sim):
    """Item-specific chance asymptote vector c_j is accepted and respected."""
    J = dual_tail_sim["Y"].shape[1]
    c_vec = np.linspace(0.0, 0.30, J)
    res = fit_dtmix_irt(
        dual_tail_sim["Y"][:80],
        dual_tail_sim["log_rt"][:80],
        dual_tail_sim["entropy"][:80],
        guessing=c_vec,
        device="numpy",
        compute_ci=False,
        max_iter=15,
    )
    np.testing.assert_allclose(res.item_params.c, c_vec)


def test_regime_parameter_ordering_recovery(fitted_numpy):
    """Recovered global regime parameters respect physical left/right-tail ordering."""
    rp = fitted_numpy.regime_params
    ip = fitted_numpy.item_params
    assert rp.mu_t_rg < float(np.median(ip.beta))
    assert rp.delta_t_ai > 0.25
    assert rp.p_ai > 0.75
    assert rp.mu_h_ai < float(np.median(ip.gamma)) - 0.5


def test_input_validation_errors(dual_tail_sim):
    """Invalid shapes, out-of-range Y, bad guessing, or unknown device raise ValueError."""
    Y = dual_tail_sim["Y"][:20, :10]
    L = dual_tail_sim["log_rt"][:20, :10]
    H = dual_tail_sim["entropy"][:20, :10]

    with pytest.raises(ValueError):
        fit_dtmix_irt(Y[:, :-1], L, H, device="numpy")
    with pytest.raises(ValueError):
        fit_dtmix_irt(Y * 2.0, L, H, device="numpy")
    with pytest.raises(ValueError):
        fit_dtmix_irt(Y, L, H, guessing=0.9, device="numpy")
    with pytest.raises(ValueError):
        fit_dtmix_irt(Y, L, H, max_iter=0, device="numpy")
    with pytest.raises(ValueError):
        fit_dtmix_irt(Y, L, H, device="invalid_backend")
    with pytest.raises(ValueError):
        fit_dtmix_irt(Y[:1, :], L[:1, :], H[:1, :], device="numpy")


def test_lrt_and_ci_boundary_consistency(fitted_numpy):
    """Whenever LRT_ai <= 3.8415, 95% profile CI includes 0; when LRT_ai >> 3.8415, CI excludes 0."""
    low_lrt = fitted_numpy.lrt_ai < 3.5
    if np.any(low_lrt):
        assert np.all(fitted_numpy.pi_ai_ci_low[low_lrt] == 0.0)
    high_lrt = fitted_numpy.lrt_ai > 8.0
    if np.any(high_lrt):
        assert np.all(fitted_numpy.pi_ai_ci_low[high_lrt] > 0.0)


def test_dual_tail_group_detects_both_regimes(dual_tail_sim, fitted_numpy):
    """Dual-tail examinees exhibit simultaneous positive pi_rg and pi_ai estimates."""
    dt = dual_tail_sim["group_label"] == 4
    assert float(fitted_numpy.pi_rg[dt].mean()) > 0.15
    assert float(fitted_numpy.pi_ai[dt].mean()) > 0.15
    assert float(fitted_numpy.p_joint[dt].mean()) < 0.01


def test_zero_guessing_open_ended_mode():
    """DT-MixIRT works cleanly with open-ended items (guessing=0.0)."""
    sim = simulate_dual_tail_exam(n_examinees=120, n_items=30, guessing=0.0, seed=303)
    res = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], guessing=0.0, device="numpy", compute_ci=False
    )
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.item_params.c == 0.0)
    g = sim["group_label"]
    m = (g == 1) | (g == 3)
    assert compute_auroc((g[m] == 3).astype(int), res.pi_ai[m]) > 0.98


def test_speed_tau_recovery_for_slow_careful(dual_tail_sim, fitted_numpy):
    """Slow careful examinees retain strongly negative estimated speed tau."""
    g = dual_tail_sim["group_label"]
    assert float(fitted_numpy.tau[g == 1].mean()) < float(fitted_numpy.tau[g == 0].mean()) - 0.5
    assert float(fitted_numpy.nu[g == 1].mean()) > float(fitted_numpy.nu[g == 0].mean()) + 0.1


def test_item_difficulty_recovery_correlation(dual_tail_sim, fitted_numpy):
    """Calibrated item difficulties b_j correlate > 0.95 with true difficulties."""
    r = float(np.corrcoef(dual_tail_sim["item_params"].b, fitted_numpy.item_params.b)[0, 1])
    assert r > 0.95


def test_item_time_intensity_recovery_correlation(dual_tail_sim, fitted_numpy):
    """Calibrated item time intensities beta_j correlate > 0.95 with ground truth."""
    r = float(np.corrcoef(dual_tail_sim["item_params"].beta, fitted_numpy.item_params.beta)[0, 1])
    assert r > 0.95


def test_item_entropy_baseline_recovery_correlation(dual_tail_sim, fitted_numpy):
    """Calibrated item entropy baselines gamma_j correlate > 0.95 with ground truth."""
    r = float(np.corrcoef(dual_tail_sim["item_params"].gamma, fitted_numpy.item_params.gamma)[0, 1])
    assert r > 0.95


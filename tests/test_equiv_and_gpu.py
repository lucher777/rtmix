"""Numerical parity (< 1e-8) and GPU/CPU backend consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from rtmix import fit_dtmix_irt, simulate_dual_tail_exam, verify_numerical_equivalence

try:
    import torch

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _HAS_CUDA = False


def test_numerical_equivalence_strict_parity():
    """Vectorized E-step, log-lik, theta M-step, and tau/nu match scalar reference < 1e-8."""
    rep = verify_numerical_equivalence(n_examinees=120, n_items=35, seed=42, tol=1e-8)
    assert rep.all_passed is True
    assert rep.max_abs_diff_estep < 1e-12
    assert rep.max_abs_diff_loglik < 1e-11
    assert rep.max_abs_diff_theta_mstep < 1e-8
    assert rep.max_abs_diff_tau_nu < 1e-12
    assert rep.speedup > 5.0


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
def test_torch_cpu_vs_numpy_full_fit_parity():
    """PyTorch CPU float64 full EM-Newton fit matches NumPy backend to < 1e-10."""
    sim = simulate_dual_tail_exam(n_examinees=100, n_items=30, seed=55)
    r_np = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], device="numpy", max_iter=15, compute_ci=True
    )
    r_th = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], device="cpu-torch", max_iter=15, compute_ci=True
    )
    assert r_np.device_used == "cpu-numpy"
    assert r_th.device_used == "cpu-torch"
    np.testing.assert_allclose(r_th.theta, r_np.theta, atol=1e-10)
    np.testing.assert_allclose(r_th.pi_ai, r_np.pi_ai, atol=1e-10)
    np.testing.assert_allclose(r_th.pi_rg, r_np.pi_rg, atol=1e-10)
    np.testing.assert_allclose(r_th.lrt_ai, r_np.lrt_ai, atol=1e-9)
    np.testing.assert_allclose(r_th.pi_ai_ci_low, r_np.pi_ai_ci_low, atol=1e-12)
    np.testing.assert_allclose(r_th.pi_ai_ci_high, r_np.pi_ai_ci_high, atol=1e-12)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_gpu_vs_numpy_full_fit_parity():
    """PyTorch CUDA float64 full fit matches NumPy backend to < 1e-10."""
    sim = simulate_dual_tail_exam(n_examinees=120, n_items=30, seed=56)
    r_np = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], device="numpy", max_iter=15, compute_ci=True
    )
    r_gpu = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], device="cuda:0", max_iter=15, compute_ci=True
    )
    assert r_gpu.device_used == "cuda:0"
    np.testing.assert_allclose(r_gpu.theta, r_np.theta, atol=1e-10)
    np.testing.assert_allclose(r_gpu.pi_ai, r_np.pi_ai, atol=1e-10)
    np.testing.assert_allclose(r_gpu.pi_rg, r_np.pi_rg, atol=1e-10)
    np.testing.assert_allclose(r_gpu.posteriors, r_np.posteriors, atol=1e-10)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_gpu_large_scale_smoke():
    """CUDA GPU handles 2,000 examinees x 60 items (120,000 trivariate records) cleanly."""
    sim = simulate_dual_tail_exam(n_examinees=2000, n_items=60, seed=77)
    r_gpu = fit_dtmix_irt(
        sim["Y"], sim["log_rt"], sim["entropy"], device="cuda:0", max_iter=20, compute_ci=True
    )
    assert r_gpu.device_used == "cuda:0"
    assert r_gpu.theta.shape == (2000,)
    assert np.all(np.isfinite(r_gpu.theta))
    assert r_gpu.fit_seconds < 10.0

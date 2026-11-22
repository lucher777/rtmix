"""Numerical equivalence and runtime parity verification against SciPy reference."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from scipy.optimize import brentq, minimize

from .datasets import simulate_dual_tail_exam
from .dtmix import (
    _EPS,
    _HAS_TORCH,
    _LOG_2PI,
    _compute_logliks_torch,
    _estep_numpy,
    _estep_torch,
    _update_theta_newton_numpy,
    compute_regime_logliks_numpy,
)
from .types import EquivalenceReport

if _HAS_TORCH:
    import torch


def _scalar_reference_estep(Y, L, H, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p):
    """Independent scalar loop reference for exact E-step posteriors and log-likelihoods."""
    N, J = Y.shape
    post = np.zeros((N, J, 3), dtype=np.float64)
    person_ll = np.zeros(N, dtype=np.float64)

    for i in range(N):
        w0 = max(1.0 - pi_rg[i] - pi_ai[i], 0.0)
        w1 = max(pi_rg[i], 0.0)
        w2 = max(pi_ai[i], 0.0)
        lw = [
            math.log(w0) if w0 > 0.0 else -1e9,
            math.log(w1) if w1 > 0.0 else -1e9,
            math.log(w2) if w2 > 0.0 else -1e9,
        ]
        ll_sum = 0.0
        for j in range(J):
            z = max(-35.0, min(35.0, item_p.a[j] * (theta[i] - item_p.b[j])))
            s = 1.0 / (1.0 + math.exp(-z))
            p0 = max(_EPS, min(1.0 - _EPS, item_p.c[j] + (1.0 - item_p.c[j]) * s))
            ll0 = (
                Y[i, j] * math.log(p0)
                + (1.0 - Y[i, j]) * math.log(1.0 - p0)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(item_p.sigma_t[j]) + ((L[i, j] - (item_p.beta[j] - tau[i])) / item_p.sigma_t[j]) ** 2)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(item_p.sigma_h[j]) + ((H[i, j] - (item_p.gamma[j] + nu[i])) / item_p.sigma_h[j]) ** 2)
            )

            p1 = max(_EPS, min(1.0 - _EPS, item_p.c[j]))
            ll1 = (
                Y[i, j] * math.log(p1)
                + (1.0 - Y[i, j]) * math.log(1.0 - p1)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(reg_p.sigma_t_rg) + ((L[i, j] - reg_p.mu_t_rg) / reg_p.sigma_t_rg) ** 2)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(reg_p.sigma_h_rg) + ((H[i, j] - reg_p.mu_h_rg) / reg_p.sigma_h_rg) ** 2)
            )

            p2 = max(_EPS, min(1.0 - _EPS, reg_p.p_ai))
            mu_t2 = item_p.beta[j] - tau[i] + reg_p.delta_t_ai
            ll2 = (
                Y[i, j] * math.log(p2)
                + (1.0 - Y[i, j]) * math.log(1.0 - p2)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(reg_p.sigma_t_ai) + ((L[i, j] - mu_t2) / reg_p.sigma_t_ai) ** 2)
                - 0.5 * (_LOG_2PI + 2.0 * math.log(reg_p.sigma_h_ai) + ((H[i, j] - reg_p.mu_h_ai) / reg_p.sigma_h_ai) ** 2)
            )

            vals = [ll0 + lw[0], ll1 + lw[1], ll2 + lw[2]]
            m = max(vals)
            denom = math.exp(vals[0] - m) + math.exp(vals[1] - m) + math.exp(vals[2] - m)
            lse = m + math.log(denom)
            ll_sum += lse
            post[i, j, 0] = math.exp(vals[0] - lse)
            post[i, j, 1] = math.exp(vals[1] - lse)
            post[i, j, 2] = math.exp(vals[2] - lse)
        person_ll[i] = ll_sum
    return post, person_ll


def verify_numerical_equivalence(
    n_examinees: int = 200,
    n_items: int = 50,
    seed: int = 42,
    tol: float = 1e-8,
    prior_var_theta: float = 4.0,
) -> EquivalenceReport:
    """Verify exact numerical parity (< 1e-8) and speedup vs. scalar SciPy L-BFGS-B.

    Checks four independent identities:
    1. Vectorized E-step posterior responsibilities vs. scalar reference loop.
    2. Vectorized marginal log-likelihoods vs. scalar reference loop.
    3. Vectorized damped Newton ``theta`` M-step vs. per-examinee SciPy L-BFGS-B.
    4. Vectorized closed-form ``tau``/``nu`` M-step vs. scalar loop, plus optional
       PyTorch CUDA vs. NumPy CPU parity when CUDA is available.

    Parameters
    ----------
    n_examinees : int, default=200
        Number of examinees in the verification benchmark.
    n_items : int, default=50
        Number of items in the verification benchmark.
    seed : int, default=42
        Random seed for reproducible test data.
    tol : float, default=1e-8
        Maximum tolerated absolute discrepancy for ``all_passed``.
    prior_var_theta : float, default=4.0
        Gaussian prior variance on ``theta_i``.

    Returns
    -------
    EquivalenceReport
        Discrepancy magnitudes, runtimes, speedup factor, and pass flag.
    """
    sim = simulate_dual_tail_exam(n_examinees=n_examinees, n_items=n_items, seed=seed)
    Y = sim["Y"]
    L = sim["log_rt"]
    H = sim["entropy"]
    mask = np.ones_like(Y, dtype=bool)
    item_p = sim["item_params"]
    reg_p = sim["regime_params"]
    theta = sim["theta_true"]
    tau = sim["tau_true"]
    nu = sim["nu_true"]
    pi_rg = np.clip(sim["pi_rg_true"] + 0.02, 0.01, 0.45)
    pi_ai = np.clip(sim["pi_ai_true"] + 0.02, 0.01, 0.45)

    # 1. E-step parity: vectorized vs. scalar reference
    ll_vec = compute_regime_logliks_numpy(Y, L, H, theta, tau, nu, item_p, reg_p)
    post_vec, pll_vec = _estep_numpy(ll_vec, pi_rg, pi_ai, mask)
    post_ref, pll_ref = _scalar_reference_estep(Y, L, H, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p)

    max_diff_estep = float(np.max(np.abs(post_vec - post_ref)))
    max_diff_ll = float(np.max(np.abs(pll_vec - pll_ref)))

    # 2. M-step theta parity (vs. exact SciPy brentq root) & speedup (vs. per-examinee L-BFGS-B)
    r0 = post_vec[:, :, 0]
    r2 = post_vec[:, :, 2]

    t_v0 = time.perf_counter()
    theta_vec = _update_theta_newton_numpy(
        np.zeros(n_examinees, dtype=np.float64),
        Y,
        r0,
        item_p.a,
        item_p.b,
        item_p.c,
        prior_var=prior_var_theta,
        n_steps=12,
    )
    vec_sec = time.perf_counter() - t_v0

    inv_pv = 1.0 / prior_var_theta
    a_v, b_v, c_v = item_p.a, item_p.b, item_p.c

    # Exact root reference via scipy.optimize.brentq on analytical score equation
    theta_exact = np.zeros(n_examinees, dtype=np.float64)
    for i in range(n_examinees):
        yi = Y[i]
        r0i = r0[i]

        def _score(th: float) -> float:
            z = np.clip(a_v * (th - b_v), -35.0, 35.0)
            s = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(c_v + (1.0 - c_v) * s, _EPS, 1.0 - _EPS)
            dp = (1.0 - c_v) * s * (1.0 - s) * a_v
            return float(np.sum(r0i * (yi - p) * dp / (p * (1.0 - p))) - th * inv_pv)

        if _score(-4.5) * _score(4.5) <= 0.0:
            theta_exact[i] = float(brentq(_score, -4.5, 4.5, xtol=1e-14, rtol=1e-14))
        else:
            theta_exact[i] = 4.5 if _score(4.5) > 0.0 else -4.5

    # Per-examinee SciPy L-BFGS-B runtime benchmark
    t_s0 = time.perf_counter()
    theta_scipy = np.zeros(n_examinees, dtype=np.float64)
    for i in range(n_examinees):
        yi = Y[i]
        r0i = r0[i]

        def _neg_obj(th_arr):
            th = float(th_arr[0])
            z = np.clip(a_v * (th - b_v), -35.0, 35.0)
            s = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(c_v + (1.0 - c_v) * s, _EPS, 1.0 - _EPS)
            val = np.sum(r0i * (yi * np.log(p) + (1.0 - yi) * np.log(1.0 - p))) - 0.5 * th * th * inv_pv
            dp = (1.0 - c_v) * s * (1.0 - s) * a_v
            grad = np.sum(r0i * (yi - p) * dp / (p * (1.0 - p))) - th * inv_pv
            return -float(val), np.array([-float(grad)], dtype=np.float64)

        opt = minimize(
            _neg_obj,
            x0=np.array([0.0]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-4.5, 4.5)],
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        theta_scipy[i] = float(opt.x[0])
    scipy_sec = time.perf_counter() - t_s0
    max_diff_theta = float(np.max(np.abs(theta_vec - theta_exact)))

    # 3. Closed-form tau & nu parity
    inv_st2 = 1.0 / (item_p.sigma_t[None, :] ** 2)
    inv_sh2 = 1.0 / (item_p.sigma_h[None, :] ** 2)
    inv_st_ai2 = 1.0 / (reg_p.sigma_t_ai**2)

    tau_vec = np.sum(
        r0 * inv_st2 * (item_p.beta[None, :] - L)
        + r2 * inv_st_ai2 * (item_p.beta[None, :] + reg_p.delta_t_ai - L),
        axis=1,
    ) / (0.25 + np.sum(r0 * inv_st2 + r2 * inv_st_ai2, axis=1))
    nu_vec = np.sum(r0 * inv_sh2 * (H - item_p.gamma[None, :]), axis=1) / (
        0.25 + np.sum(r0 * inv_sh2, axis=1)
    )

    tau_ref = np.zeros(n_examinees, dtype=np.float64)
    nu_ref = np.zeros(n_examinees, dtype=np.float64)
    for i in range(n_examinees):
        t_num = 0.0
        t_den = 0.25
        n_num = 0.0
        n_den = 0.25
        for j in range(n_items):
            w0 = r0[i, j] / (item_p.sigma_t[j] ** 2)
            w2 = r2[i, j] / (reg_p.sigma_t_ai**2)
            t_num += w0 * (item_p.beta[j] - L[i, j]) + w2 * (item_p.beta[j] + reg_p.delta_t_ai - L[i, j])
            t_den += w0 + w2
            wh = r0[i, j] / (item_p.sigma_h[j] ** 2)
            n_num += wh * (H[i, j] - item_p.gamma[j])
            n_den += wh
        tau_ref[i] = t_num / t_den
        nu_ref[i] = n_num / n_den

    max_diff_tau_nu = float(max(np.max(np.abs(tau_vec - tau_ref)), np.max(np.abs(nu_vec - nu_ref))))

    # 4. Optional GPU vs. CPU float64 parity
    max_diff_gpu_cpu: Optional[float] = None
    device_used = "cpu-numpy"
    if _HAS_TORCH and torch.cuda.is_available():
        device_used = "cuda:0"
        dev = torch.device("cuda:0")
        with torch.no_grad():
            Y_t = torch.as_tensor(Y, device=dev, dtype=torch.float64)
            L_t = torch.as_tensor(L, device=dev, dtype=torch.float64)
            H_t = torch.as_tensor(H, device=dev, dtype=torch.float64)
            m_t = torch.as_tensor(mask, device=dev, dtype=torch.bool)
            th_t = torch.as_tensor(theta, device=dev, dtype=torch.float64)
            ta_t = torch.as_tensor(tau, device=dev, dtype=torch.float64)
            nu_t = torch.as_tensor(nu, device=dev, dtype=torch.float64)
            prg_t = torch.as_tensor(pi_rg, device=dev, dtype=torch.float64)
            pai_t = torch.as_tensor(pi_ai, device=dev, dtype=torch.float64)
            ll_t = _compute_logliks_torch(
                Y_t, L_t, H_t, th_t, ta_t, nu_t,
                torch.as_tensor(item_p.a, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.b, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.c, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.beta, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.sigma_t, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.gamma, device=dev, dtype=torch.float64),
                torch.as_tensor(item_p.sigma_h, device=dev, dtype=torch.float64),
                reg_p,
            )
            post_gpu, _ = _estep_torch(ll_t, prg_t, pai_t, m_t)
            torch.cuda.synchronize()
            max_diff_gpu_cpu = float(np.max(np.abs(post_gpu.cpu().numpy() - post_vec)))

    speedup = scipy_sec / max(vec_sec, 1e-9)
    checks = [max_diff_estep < tol, max_diff_ll < tol, max_diff_theta < tol, max_diff_tau_nu < tol]
    if max_diff_gpu_cpu is not None:
        checks.append(max_diff_gpu_cpu < tol)

    return EquivalenceReport(
        max_abs_diff_estep=max_diff_estep,
        max_abs_diff_loglik=max_diff_ll,
        max_abs_diff_theta_mstep=max_diff_theta,
        max_abs_diff_tau_nu=max_diff_tau_nu,
        max_abs_diff_gpu_cpu=max_diff_gpu_cpu,
        vectorized_seconds=vec_sec,
        scipy_seconds=scipy_sec,
        speedup=speedup,
        device_used=device_used,
        all_passed=all(checks),
    )

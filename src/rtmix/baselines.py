"""Classical psychometric baselines for head-to-head dual-tail evaluation.

Provides:
- ``fit_classical_irt``: Accuracy-only 2PL/3PL IRT (Bock & Aitkin, 1981).
- ``fit_vdl_speed_accuracy``: Bivariate speed-accuracy log-normal IRT baseline
  (van der Linden, 2007).
- ``fit_wise_nt_baseline``: Normative-threshold effort-moderated IRT baseline
  (Wise & Kong, 2005; Wise & DeMars, 2006).
- ``fit_bivariate_mixture_ablation``: Bivariate ``(Y, log_rt)`` ablation without
  edit/action entropy.
"""

from __future__ import annotations

import time
from typing import Union

import numpy as np

from .dtmix import (
    _EPS,
    _sigmoid_np,
    _update_items_newton_numpy,
    _update_theta_newton_numpy,
    _validate_inputs,
    fit_dtmix_irt,
)
from .types import BaselineResult


def fit_classical_irt(
    Y: np.ndarray,
    guessing: Union[float, np.ndarray] = 0.25,
    max_iter: int = 35,
    prior_var_theta: float = 4.0,
) -> BaselineResult:
    """Fit a standard accuracy-only 3PL/2PL IRT model via alternating Newton updates.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)`` (``NaN`` allowed).
    guessing : float or np.ndarray, default=0.25
        Known lower chance asymptote(s) ``c_j``.
    max_iter : int, default=35
        Number of alternating person/item Newton iterations.
    prior_var_theta : float, default=4.0
        Gaussian prior variance on examinee ability ``theta_i``.

    Returns
    -------
    BaselineResult
        Estimated abilities and accuracy-only left/right tail anomaly scores.
    """
    t0 = time.perf_counter()
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D; got shape {Y_arr.shape}.")
    N, J = Y_arr.shape
    mask = np.isfinite(Y_arr)
    mask_f = mask.astype(np.float64)
    Y_c = np.where(mask, Y_arr, 0.0)
    c_arr = (
        np.full(J, float(guessing), dtype=np.float64)
        if np.ndim(guessing) == 0
        else np.asarray(guessing, dtype=np.float64)
    )

    # Initialize
    p_j = np.clip((Y_c.sum(axis=0) + 0.5) / (mask_f.sum(axis=0) + 1.0), c_arr + 0.03, 0.97)
    p_adj = np.clip((p_j - c_arr) / np.maximum(1.0 - c_arr, 0.1), 0.03, 0.97)
    b = np.clip(-np.log(p_adj / (1.0 - p_adj)), -3.0, 3.0)
    a = np.ones(J, dtype=np.float64)

    p_i = np.clip((Y_c.sum(axis=1) + 0.5) / (mask_f.sum(axis=1) + 1.0), 0.05, 0.95)
    theta = np.log(p_i / (1.0 - p_i))
    theta = (theta - theta.mean()) / max(float(theta.std()), 0.5)

    for _ in range(max_iter):
        theta = _update_theta_newton_numpy(
            theta, Y_c, mask_f, a, b, c_arr, prior_var=prior_var_theta, n_steps=3
        )
        sd = float(theta.std())
        if sd > 1e-3:
            mu = float(theta.mean())
            theta = (theta - mu) / sd
            b = (b - mu) / sd
            a = np.clip(a * sd, 0.25, 3.50)
        a, b = _update_items_newton_numpy(a, b, c_arr, Y_c, mask_f, theta, n_steps=2)

    # Accuracy-only heuristic anomaly scores:
    # easy-item unexpected errors -> RG score; hard-item unexpected successes -> AI score
    z = a[None, :] * (theta[:, None] - b[None, :])
    p_hat = np.clip(c_arr[None, :] + (1.0 - c_arr[None, :]) * _sigmoid_np(z), _EPS, 1.0 - _EPS)
    hard_mask = b >= np.percentile(b, 75.0)
    easy_mask = b <= np.percentile(b, 25.0)

    ai_score = np.sum(mask_f[:, hard_mask] * (Y_c[:, hard_mask] - p_hat[:, hard_mask]), axis=1) / np.maximum(
        mask_f[:, hard_mask].sum(axis=1), 1.0
    )
    rg_score = np.sum(mask_f[:, easy_mask] * (p_hat[:, easy_mask] - Y_c[:, easy_mask]), axis=1) / np.maximum(
        mask_f[:, easy_mask].sum(axis=1), 1.0
    )

    return BaselineResult(
        method="classical_3pl_irt",
        theta=theta,
        rg_score=rg_score,
        ai_score=ai_score,
        fit_seconds=time.perf_counter() - t0,
        extras={"a": a, "b": b, "c": c_arr},
    )


def fit_vdl_speed_accuracy(
    Y: np.ndarray,
    log_rt: np.ndarray,
    guessing: Union[float, np.ndarray] = 0.25,
    max_iter: int = 35,
) -> BaselineResult:
    """Fit a van der Linden (2007) log-normal speed-accuracy hierarchical baseline.

    Estimates separate accuracy abilities (``theta``) and log-normal working speed
    (``tau``), scoring left-tail rapid guessing via excess speed (``tau``) and
    right-tail slowness/copying via negative speed (``-tau``) plus positive RT
    tail skew.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    guessing : float or np.ndarray, default=0.25
        Known lower chance asymptote(s).
    max_iter : int, default=35
        Iterations for accuracy IRT calibration.

    Returns
    -------
    BaselineResult
        Baseline ability estimates and RT-based speed/slowness anomaly scores.
    """
    t0 = time.perf_counter()
    irt_res = fit_classical_irt(Y, guessing=guessing, max_iter=max_iter)
    L = np.asarray(log_rt, dtype=np.float64)
    mask = np.isfinite(Y) & np.isfinite(L)
    mask_f = mask.astype(np.float64)
    L_c = np.where(mask, L, 0.0)

    # Log-normal RT model: L_ij = beta_j - tau_i + e_ij
    beta = np.sum(L_c * mask_f, axis=0) / np.maximum(mask_f.sum(axis=0), 1.0)
    tau = -np.sum((L_c - beta[None, :]) * mask_f, axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)
    tau = tau - np.mean(tau)

    # RT-only right-tail score: slowness (-tau) + excess hard-item accuracy
    # Fundamentally confounds slow careful examinees (tau << 0) with AI copiers!
    slowness = -tau
    rt_res = np.where(mask, L_c - beta[None, :], 0.0)
    fast_frac = np.sum(mask & (rt_res < -1.0), axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)
    slow_frac = np.sum(mask & (rt_res > 0.8), axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)

    return BaselineResult(
        method="vdl_speed_accuracy",
        theta=irt_res.theta,
        rg_score=fast_frac + 0.25 * tau,
        ai_score=0.5 * slowness + 0.5 * slow_frac,
        fit_seconds=time.perf_counter() - t0,
        extras={"tau": tau, "beta": beta, "slowness": slowness, "fast_frac": fast_frac},
    )


def fit_wise_nt_baseline(
    Y: np.ndarray,
    log_rt: np.ndarray,
    threshold_ratio: float = 0.10,
    guessing: Union[float, np.ndarray] = 0.25,
    max_iter: int = 35,
) -> BaselineResult:
    """Fit the Wise & Kong (2005) / Wise & DeMars (2006) effort-moderated NT10 baseline.

    Flags item responses faster than ``threshold_ratio`` of each item's mean
    response time (on the raw RT scale ``exp(log_rt)``) as rapid guesses, computes
    examinee Response Time Effort (``RTE``), filters rapid-guessed responses, and
    fits effort-moderated 3PL IRT on remaining solution responses.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    threshold_ratio : float, default=0.10
        Normative threshold proportion of mean item RT (default ``0.10`` for NT10).
    guessing : float or np.ndarray, default=0.25
        Known lower chance asymptote(s).
    max_iter : int, default=35
        Iterations for effort-moderated IRT calibration.

    Returns
    -------
    BaselineResult
        Effort-moderated ability estimates and left-tail ``1 - RTE`` scores.
    """
    t0 = time.perf_counter()
    Y_arr = np.asarray(Y, dtype=np.float64)
    L_arr = np.asarray(log_rt, dtype=np.float64)
    mask = np.isfinite(Y_arr) & np.isfinite(L_arr)

    rt = np.where(mask, np.exp(np.clip(L_arr, -10.0, 12.0)), np.nan)
    mean_rt_j = np.nanmean(rt, axis=0)
    # NT10 threshold capped per Wise (2017) conventions
    thresh_j = np.maximum(threshold_ratio * mean_rt_j, np.nanpercentile(rt, 5.0, axis=0))
    is_rg = mask & (rt < thresh_j[None, :])
    sb_mask = mask & (~is_rg)

    n_obs = np.maximum(mask.sum(axis=1), 1.0)
    rte = np.sum(sb_mask, axis=1) / n_obs
    rg_rate = 1.0 - rte

    # Effort-moderated IRT: drop rapid guesses (weight 0 on RG cells)
    Y_em = np.where(sb_mask, Y_arr, np.nan)
    # Guard examinees with all items filtered
    empty_rows = np.sum(sb_mask, axis=1) == 0
    if np.any(empty_rows):
        Y_em[empty_rows] = Y_arr[empty_rows]
    em_irt = fit_classical_irt(Y_em, guessing=guessing, max_iter=max_iter)

    # Wise NT10 has no right-tail AI mechanism; proxy slow-tail fraction for comparison
    med_rt_j = np.nanmedian(rt, axis=0)
    slow_tail = np.sum(mask & (rt > 2.0 * med_rt_j[None, :]), axis=1) / n_obs

    return BaselineResult(
        method="wise_nt10_effort_moderated",
        theta=em_irt.theta,
        rg_score=rg_rate,
        ai_score=slow_tail,
        fit_seconds=time.perf_counter() - t0,
        extras={"rte": rte, "is_rg": is_rg.astype(np.int8), "thresh_rt": thresh_j},
    )


def fit_bivariate_mixture_ablation(
    Y: np.ndarray,
    log_rt: np.ndarray,
    guessing: Union[float, np.ndarray] = 0.25,
    max_iter: int = 35,
    device: str = "cpu-numpy",
) -> BaselineResult:
    """Fit a bivariate ``(Y, log_rt)`` ablation of DT-MixIRT without edit entropy.

    Removes all information from the edit/action-entropy channel by passing a
    degenerate zero-variance constant stream, isolating the exact marginal gain
    contributed by trivariate keystroke/action entropy (Theorem 1).

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    guessing : float or np.ndarray, default=0.25
        Known lower chance asymptote(s).
    max_iter : int, default=35
        Maximum EM-Newton iterations.
    device : str, default="cpu-numpy"
        Compute backend passed to ``fit_dtmix_irt``.

    Returns
    -------
    BaselineResult
        Bivariate ablation ability estimates and regime weights.
    """
    t0 = time.perf_counter()
    H_dummy = np.zeros_like(Y, dtype=np.float64)
    res = fit_dtmix_irt(
        Y,
        log_rt,
        H_dummy,
        guessing=guessing,
        max_iter=max_iter,
        device=device,
        compute_ci=False,
        use_entropy=False,
    )
    return BaselineResult(
        method="bivariate_rt_acc_mixture",
        theta=res.theta,
        rg_score=res.pi_rg,
        ai_score=res.pi_ai,
        fit_seconds=time.perf_counter() - t0,
        extras={"lrt_rg": res.lrt_rg, "lrt_ai": res.lrt_ai},
    )

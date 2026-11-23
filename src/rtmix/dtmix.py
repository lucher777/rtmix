"""Dual-Tail Process-Data Mixture IRT (DT-MixIRT) estimation engine.

Implements GPU-accelerated (PyTorch CUDA/CPU) and vectorized NumPy alternating
EM-Newton estimation for the trivariate three-regime process-data mixture model
over binary accuracy (Y), log response time (log T), and keystroke/action edit
entropy (H), separating:
- Regime 0: Genuine solution behavior (3PL IRT x log-normal RT x baseline entropy)
- Regime 1: Left-tail rapid guessing (chance floor, fast RT, minimal entropy)
- Regime 2: Right-tail AI-assisted copying (elevated accuracy floor, prompting
  pause latency shift, collapsed edit/revision entropy)
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2

from .types import DTMixIRTResult, ItemParameters, RegimeParameters

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

_LOG_2PI = math.log(2.0 * math.pi)
_EPS = 1e-12


def _resolve_device(device: str) -> str:
    """Resolve requested compute backend to a concrete runtime identifier.

    Parameters
    ----------
    device : str
        One of ``"auto"``, ``"cuda"``, ``"cuda:0"``, ``"cpu"``, ``"cpu-torch"``,
        or ``"numpy"``.

    Returns
    -------
    str
        Concrete backend tag: ``"cuda:0"``, ``"cpu-torch"``, or ``"cpu-numpy"``.
    """
    dev = device.lower().strip()
    if dev in ("numpy", "cpu-numpy"):
        return "cpu-numpy"
    if dev == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda:0"
        if _HAS_TORCH:
            return "cpu-torch"
        return "cpu-numpy"
    if dev.startswith("cuda"):
        if not (_HAS_TORCH and torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but PyTorch CUDA is not available.")
        return dev if ":" in dev else "cuda:0"
    if dev in ("cpu", "cpu-torch", "torch"):
        if _HAS_TORCH:
            return "cpu-torch"
        return "cpu-numpy"
    raise ValueError(f"Unsupported device specification: {device!r}")


def _validate_inputs(
    Y: np.ndarray,
    log_rt: np.ndarray,
    entropy: np.ndarray,
    guessing: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce trivariate process-data arrays to float64.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)`` with optional ``NaN`` missing values.
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    entropy : np.ndarray
        Keystroke/action edit entropy matrix of shape ``(N, J)``.
    guessing : float or np.ndarray
        Scalar or length-``J`` chance asymptote array in ``[0, 0.5]``.

    Returns
    -------
    tuple
        Validated ``(Y_arr, L_arr, H_arr, mask, c_arr)`` arrays.
    """
    Y_arr = np.asarray(Y, dtype=np.float64)
    L_arr = np.asarray(log_rt, dtype=np.float64)
    H_arr = np.asarray(entropy, dtype=np.float64)

    if Y_arr.ndim != 2 or L_arr.shape != Y_arr.shape or H_arr.shape != Y_arr.shape:
        raise ValueError(
            "Y, log_rt, and entropy must be 2D arrays with identical shape (N, J); "
            f"got {Y_arr.shape}, {L_arr.shape}, {H_arr.shape}."
        )
    N, J = Y_arr.shape
    if N < 2 or J < 2:
        raise ValueError(f"Expected at least 2 examinees and 2 items; got ({N}, {J}).")

    mask = np.isfinite(Y_arr) & np.isfinite(L_arr) & np.isfinite(H_arr)
    if not np.any(mask):
        raise ValueError("All entries in (Y, log_rt, entropy) are non-finite.")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("At least one examinee has zero valid trivariate observations.")
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError("At least one item has zero valid trivariate observations.")

    y_obs = Y_arr[mask]
    if np.any((y_obs < 0.0) | (y_obs > 1.0)):
        raise ValueError("Observed entries of Y must lie in [0, 1].")

    if np.ndim(guessing) == 0:
        c_arr = np.full(J, float(guessing), dtype=np.float64)
    else:
        c_arr = np.asarray(guessing, dtype=np.float64)
        if c_arr.shape != (J,):
            raise ValueError(f"guessing must be scalar or shape ({J},); got {c_arr.shape}.")
    if np.any((c_arr < 0.0) | (c_arr >= 0.75)):
        raise ValueError("guessing asymptotes must lie in [0, 0.75).")

    Y_clean = np.where(mask, Y_arr, 0.0)
    L_clean = np.where(mask, L_arr, 0.0)
    H_clean = np.where(mask, H_arr, 0.0)
    return Y_clean, L_clean, H_clean, mask, c_arr


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid in NumPy."""
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _initialize_parameters(
    Y: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    c_arr: np.ndarray,
    use_entropy: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ItemParameters, RegimeParameters]:
    """Compute robust quantile-based starting values immune to dual-tail outliers.

    Uses the upper-entropy core (when ``use_entropy=True``) or central RT core
    (when ``use_entropy=False``) to seed genuine solution baselines.
    """
    N, J = Y.shape
    mask_f = mask.astype(np.float64)

    beta = np.zeros(J, dtype=np.float64)
    sigma_t = np.zeros(J, dtype=np.float64)
    gamma = np.zeros(J, dtype=np.float64)
    sigma_h = np.zeros(J, dtype=np.float64)
    b = np.zeros(J, dtype=np.float64)

    for j in range(J):
        mj = mask[:, j]
        lj = L[mj, j]
        hj = H[mj, j]
        yj = Y[mj, j]
        if use_entropy:
            h_cut = np.percentile(hj, 35.0)
            core = hj >= h_cut
        else:
            lo_q, hi_q = np.percentile(lj, [15.0, 85.0])
            core = (lj >= lo_q) & (lj <= hi_q)
        if core.sum() < 5:
            core = np.ones_like(hj, dtype=bool)
        beta[j] = float(np.median(lj[core]))
        mad_t = float(np.median(np.abs(lj[core] - beta[j]))) * 1.4826
        sigma_t[j] = float(np.clip(mad_t, 0.20, 1.50))

        gamma[j] = float(np.median(hj[core]))
        mad_h = float(np.median(np.abs(hj[core] - gamma[j]))) * 1.4826
        sigma_h[j] = float(np.clip(mad_h, 0.25, 1.50))

        p_obs = float(np.clip(yj[core].mean(), c_arr[j] + 0.03, 0.97))
        p_adj = float(np.clip((p_obs - c_arr[j]) / max(1.0 - c_arr[j], 0.1), 0.03, 0.97))
        b[j] = float(np.clip(-math.log(p_adj / (1.0 - p_adj)), -3.0, 3.0))

    a = np.ones(J, dtype=np.float64)

    L_res = np.where(mask, L - beta[None, :], 0.0)
    H_res = np.where(mask, H - gamma[None, :], 0.0)
    if use_entropy:
        core_cell = mask & (H_res > -1.0 * sigma_h[None, :])
    else:
        core_cell = mask & (L_res > -1.0 * sigma_t[None, :]) & (L_res < 1.2 * sigma_t[None, :])
    core_cnt = np.maximum(core_cell.sum(axis=1), 1.0)

    tau = -np.sum(np.where(core_cell, L_res, 0.0), axis=1) / core_cnt
    tau = tau - np.mean(tau)
    tau = np.clip(tau, -2.5, 2.5)

    nu = np.sum(np.where(core_cell, H_res, 0.0), axis=1) / core_cnt
    nu = nu - np.mean(nu)
    nu = np.clip(nu, -2.0, 2.0)

    p_person = np.clip(
        (np.sum(np.where(core_cell, Y, 0.0), axis=1) + 0.5) / (core_cnt + 1.0),
        0.05,
        0.95,
    )
    theta = np.log(p_person / (1.0 - p_person))
    theta = (theta - np.mean(theta)) / max(float(np.std(theta)), 0.5)
    theta = np.clip(theta, -3.5, 3.5)

    rt_dev = L - (beta[None, :] - tau[:, None])
    h_dev = H - (gamma[None, :] + nu[:, None])

    rg_seed = mask & (rt_dev < -1.0) & ((h_dev < -0.7) if use_entropy else True)
    if rg_seed.sum() >= 15:
        mu_t_rg = float(np.median(L[rg_seed]))
        sigma_t_rg = float(np.clip(np.std(L[rg_seed]), 0.20, 1.00))
        mu_h_rg = float(np.median(H[rg_seed]))
        sigma_h_rg = float(np.clip(np.std(H[rg_seed]), 0.20, 1.00))
    else:
        mu_t_rg = float(np.median(beta) - 1.8)
        sigma_t_rg = 0.40
        mu_h_rg = float(np.median(gamma) - 2.0)
        sigma_h_rg = 0.45

    ai_seed = mask & (rt_dev > 0.35) & ((h_dev < -1.0) if use_entropy else (Y > 0.5))
    if ai_seed.sum() >= 15:
        delta_t_ai = float(np.clip(np.median(rt_dev[ai_seed]), 0.35, 2.20))
        sigma_t_ai = float(np.clip(np.std(rt_dev[ai_seed]), 0.20, 1.00))
        mu_h_ai = float(np.median(H[ai_seed]))
        sigma_h_ai = float(np.clip(np.std(H[ai_seed]), 0.20, 1.00))
        p_ai = float(np.clip(np.mean(Y[ai_seed]), 0.75, 0.97))
    else:
        delta_t_ai = 0.85
        sigma_t_ai = 0.40
        mu_h_ai = float(np.median(gamma) - 2.2)
        sigma_h_ai = 0.40
        p_ai = 0.90

    mu_t_rg = min(mu_t_rg, float(np.median(beta)) - 0.75)
    mu_h_rg = min(mu_h_rg, float(np.median(gamma)) - 0.75)
    mu_h_ai = min(mu_h_ai, float(np.median(gamma)) - 0.90)

    pi_rg = np.clip((rg_seed.sum(axis=1) + 0.5) / (mask_f.sum(axis=1) + 10.0), 0.01, 0.45)
    pi_ai = np.clip((ai_seed.sum(axis=1) + 0.5) / (mask_f.sum(axis=1) + 10.0), 0.01, 0.45)
    excess = pi_rg + pi_ai
    scale = np.where(excess > 0.85, 0.85 / np.maximum(excess, _EPS), 1.0)
    pi_rg *= scale
    pi_ai *= scale

    item_params = ItemParameters(
        a=a, b=b, c=c_arr.copy(), beta=beta, sigma_t=sigma_t, gamma=gamma, sigma_h=sigma_h
    )
    regime_params = RegimeParameters(
        mu_t_rg=mu_t_rg,
        sigma_t_rg=sigma_t_rg,
        mu_h_rg=mu_h_rg,
        sigma_h_rg=sigma_h_rg,
        p_ai=p_ai,
        delta_t_ai=delta_t_ai,
        sigma_t_ai=sigma_t_ai,
        mu_h_ai=mu_h_ai,
        sigma_h_ai=sigma_h_ai,
    )
    return theta, tau, nu, pi_rg, pi_ai, item_params, regime_params


def compute_regime_logliks_numpy(
    Y: np.ndarray,
    log_rt: np.ndarray,
    entropy: np.ndarray,
    theta: np.ndarray,
    tau: np.ndarray,
    nu: np.ndarray,
    item_params: ItemParameters,
    regime_params: RegimeParameters,
    use_entropy: bool = True,
) -> np.ndarray:
    """Compute exact component log-densities ``log f_k(Y_ij, L_ij, H_ij)`` in NumPy.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix, shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix, shape ``(N, J)``.
    entropy : np.ndarray
        Keystroke/action edit entropy matrix, shape ``(N, J)``.
    theta, tau, nu : np.ndarray
        Examinee latent traits of shape ``(N,)``.
    item_params : ItemParameters
        Item-level parameters.
    regime_params : RegimeParameters
        Global left-tail and right-tail regime parameters.
    use_entropy : bool, default=True
        If False, drops the edit-entropy channel to obtain a pure bivariate
        ``(Y, log_rt)`` three-regime mixture likelihood.

    Returns
    -------
    np.ndarray
        Component log-likelihood tensor of shape ``(N, J, 3)`` for
        ``(0=Solution, 1=RapidGuessing, 2=AIAssisted)``.
    """
    a = item_params.a[None, :]
    b = item_params.b[None, :]
    c = item_params.c[None, :]
    beta = item_params.beta[None, :]
    st = item_params.sigma_t[None, :]
    gamma = item_params.gamma[None, :]
    sh = item_params.sigma_h[None, :]

    # Regime 0: Genuine solution behavior
    z = a * (theta[:, None] - b)
    p0 = np.clip(c + (1.0 - c) * _sigmoid_np(z), _EPS, 1.0 - _EPS)
    ll_y0 = Y * np.log(p0) + (1.0 - Y) * np.log(1.0 - p0)
    mu_t0 = beta - tau[:, None]
    ll_t0 = -0.5 * (_LOG_2PI + 2.0 * np.log(st) + ((log_rt - mu_t0) / st) ** 2)
    if use_entropy:
        mu_h0 = gamma + nu[:, None]
        ll_h0 = -0.5 * (_LOG_2PI + 2.0 * np.log(sh) + ((entropy - mu_h0) / sh) ** 2)
    else:
        ll_h0 = 0.0
    ll0 = ll_y0 + ll_t0 + ll_h0

    # Regime 1: Left-tail rapid guessing
    p1 = np.clip(c, _EPS, 1.0 - _EPS)
    ll_y1 = Y * np.log(p1) + (1.0 - Y) * np.log(1.0 - p1)
    ll_t1 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(regime_params.sigma_t_rg)
        + ((log_rt - regime_params.mu_t_rg) / regime_params.sigma_t_rg) ** 2
    )
    if use_entropy:
        ll_h1 = -0.5 * (
            _LOG_2PI
            + 2.0 * math.log(regime_params.sigma_h_rg)
            + ((entropy - regime_params.mu_h_rg) / regime_params.sigma_h_rg) ** 2
        )
    else:
        ll_h1 = 0.0
    ll1 = ll_y1 + ll_t1 + ll_h1

    # Regime 2: Right-tail AI-assisted copying
    p2 = float(np.clip(regime_params.p_ai, _EPS, 1.0 - _EPS))
    ll_y2 = Y * math.log(p2) + (1.0 - Y) * math.log(1.0 - p2)
    mu_t2 = mu_t0 + regime_params.delta_t_ai
    ll_t2 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(regime_params.sigma_t_ai)
        + ((log_rt - mu_t2) / regime_params.sigma_t_ai) ** 2
    )
    if use_entropy:
        ll_h2 = -0.5 * (
            _LOG_2PI
            + 2.0 * math.log(regime_params.sigma_h_ai)
            + ((entropy - regime_params.mu_h_ai) / regime_params.sigma_h_ai) ** 2
        )
    else:
        ll_h2 = 0.0
    ll2 = ll_y2 + ll_t2 + ll_h2

    return np.stack([ll0, ll1, ll2], axis=-1)


def compute_item_regime_posteriors(
    Y: np.ndarray,
    log_rt: np.ndarray,
    entropy: np.ndarray,
    result: DTMixIRTResult,
) -> np.ndarray:
    """Compute item-level forensic posterior probabilities ``P(Z_ij = k | Y, L, H)``.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    entropy : np.ndarray
        Edit/action entropy matrix of shape ``(N, J)``.
    result : DTMixIRTResult
        Fitted ``DT-MixIRT`` model result.

    Returns
    -------
    np.ndarray
        Posterior responsibility tensor of shape ``(N, J, 3)`` summing to 1 along
        the last axis on all observed entries (and ``NaN`` on missing entries).
    """
    Y_c, L_c, H_c, mask, _ = _validate_inputs(Y, log_rt, entropy, result.item_params.c)
    ll = compute_regime_logliks_numpy(
        Y_c, L_c, H_c, result.theta, result.tau, result.nu, result.item_params, result.regime_params
    )
    weights = np.stack([result.pi_sol, result.pi_rg, result.pi_ai], axis=-1)
    log_w = np.where(weights > 0.0, np.log(np.maximum(weights, 1e-300)), -1e9)
    joint = ll + log_w[:, None, :]
    m = np.max(joint, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(joint - m), axis=-1, keepdims=True))
    post = np.exp(joint - lse)
    post = np.where(mask[:, :, None], post, np.nan)
    return post


def _estep_numpy(
    ll: np.ndarray,
    pi_rg: np.ndarray,
    pi_ai: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized NumPy E-step returning posteriors (N, J, 3) and person log-likelihoods (N,)."""
    pi_sol = np.clip(1.0 - pi_rg - pi_ai, 0.0, 1.0)
    w = np.stack([pi_sol, np.clip(pi_rg, 0.0, 1.0), np.clip(pi_ai, 0.0, 1.0)], axis=-1)
    log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), -1e9)
    joint = ll + log_w[:, None, :]
    m = np.max(joint, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(joint - m), axis=-1, keepdims=True))
    post = np.exp(joint - lse)
    post = np.where(mask[:, :, None], post, 0.0)
    person_ll = np.sum(np.where(mask, lse[:, :, 0], 0.0), axis=1)
    return post, person_ll


def _update_theta_newton_numpy(
    theta: np.ndarray,
    Y: np.ndarray,
    r0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    prior_var: float = 4.0,
    n_steps: int = 4,
) -> np.ndarray:
    """Safeguarded exact observed Newton-Raphson update for examinee abilities."""
    th = theta.copy()
    inv_pv = 1.0 / prior_var
    a_row = a[None, :]
    b_row = b[None, :]
    c_row = c[None, :]
    a2_row = a_row * a_row
    for _ in range(n_steps):
        z = a_row * (th[:, None] - b_row)
        s = _sigmoid_np(z)
        p = np.clip(c_row + (1.0 - c_row) * s, _EPS, 1.0 - _EPS)
        # Exact 3PL cancellation: dp / (p * (1 - p)) = a * s / p
        grad = np.sum(r0 * a_row * (s / p) * (Y - p), axis=1) - th * inv_pv
        # Exact observed negative Hessian safeguarded by expected Fisher information
        s1ms = s * (1.0 - s)
        obs_h = np.sum(r0 * a2_row * s1ms * (1.0 - (c_row * Y) / (p * p)), axis=1) + inv_pv
        fish_h = np.sum(r0 * a2_row * (1.0 - c_row) * (s * s1ms) / p, axis=1) + inv_pv
        hess = np.maximum(obs_h, np.maximum(0.25 * fish_h, inv_pv))
        step = np.clip(grad / np.maximum(hess, 1e-8), -0.85, 0.85)
        th = np.clip(th + step, -4.5, 4.5)
    return th


def _update_items_newton_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    Y: np.ndarray,
    r0: np.ndarray,
    theta: np.ndarray,
    n_steps: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized damped Newton-Raphson update for 3PL item parameters (a_j, b_j)."""
    a_cur = a.copy()
    b_cur = b.copy()
    th_col = theta[:, None]
    c_row = c[None, :]
    for _ in range(n_steps):
        z = a_cur[None, :] * (th_col - b_cur[None, :])
        s = _sigmoid_np(z)
        p = np.clip(c_row + (1.0 - c_row) * s, _EPS, 1.0 - _EPS)
        common = (1.0 - c_row) * s * (1.0 - s)
        dp_da = common * (th_col - b_cur[None, :])
        dp_db = -common * a_cur[None, :]
        denom = p * (1.0 - p)
        resid = (Y - p) / denom

        # Weak regularizing priors: log(a) ~ N(0, 0.5^2), b ~ N(0, 2.5^2)
        g_a = np.sum(r0 * resid * dp_da, axis=0) - (np.log(a_cur) / (0.25 * a_cur))
        g_b = np.sum(r0 * resid * dp_db, axis=0) - b_cur / 6.25

        h_aa = np.sum(r0 * (dp_da * dp_da) / denom, axis=0) + 1.0 / (0.25 * a_cur * a_cur) + 1e-4
        h_bb = np.sum(r0 * (dp_db * dp_db) / denom, axis=0) + 1.0 / 6.25 + 1e-4
        h_ab = np.sum(r0 * (dp_da * dp_db) / denom, axis=0)

        det = np.maximum(h_aa * h_bb - h_ab * h_ab, 1e-6)
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det

        a_cur = np.clip(a_cur + np.clip(da, -0.35, 0.35), 0.25, 3.50)
        b_cur = np.clip(b_cur + np.clip(db, -0.50, 0.50), -4.0, 4.0)
    return a_cur, b_cur


def _fit_numpy_backend(
    Y: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    c_arr: np.ndarray,
    max_iter: int,
    tol: float,
    fixed_item_params: Optional[ItemParameters],
    fixed_regime_params: Optional[RegimeParameters],
    compute_ci: bool,
    ci_grid_size: int,
    prior_var_theta: float,
    prior_prec_tau: float,
    prior_prec_nu: float,
    use_entropy: bool = True,
) -> DTMixIRTResult:
    """Pure NumPy vectorized float64 DT-MixIRT estimation backend."""
    t0 = time.perf_counter()
    N, J = Y.shape
    mask_f = mask.astype(np.float64)
    n_obs_i = np.maximum(mask_f.sum(axis=1), 1.0)

    theta, tau, nu, pi_rg, pi_ai, item_p, reg_p = _initialize_parameters(
        Y, L, H, mask, c_arr, use_entropy=use_entropy
    )
    if fixed_item_params is not None:
        item_p = ItemParameters(
            a=fixed_item_params.a.copy(),
            b=fixed_item_params.b.copy(),
            c=fixed_item_params.c.copy(),
            beta=fixed_item_params.beta.copy(),
            sigma_t=fixed_item_params.sigma_t.copy(),
            gamma=fixed_item_params.gamma.copy(),
            sigma_h=fixed_item_params.sigma_h.copy(),
        )
    if fixed_regime_params is not None:
        reg_p = RegimeParameters(**fixed_regime_params.__dict__)

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    post = np.zeros((N, J, 3), dtype=np.float64)
    total_ll = -np.inf

    for it in range(1, max_iter + 1):
        n_iter = it
        ll = compute_regime_logliks_numpy(
            Y, L, H, theta, tau, nu, item_p, reg_p, use_entropy=use_entropy
        )
        post, person_ll = _estep_numpy(ll, pi_rg, pi_ai, mask)
        total_ll = float(np.sum(person_ll))

        r0 = post[:, :, 0]
        r1 = post[:, :, 1]
        r2 = post[:, :, 2]

        # Person M-step
        pi_rg = np.clip(np.sum(r1, axis=1) / n_obs_i, 0.0, 0.99)
        pi_ai = np.clip(np.sum(r2, axis=1) / n_obs_i, 0.0, 0.99)
        s_pi = pi_rg + pi_ai
        over = s_pi > 0.995
        if np.any(over):
            pi_rg[over] *= 0.995 / s_pi[over]
            pi_ai[over] *= 0.995 / s_pi[over]

        inv_st2 = 1.0 / (item_p.sigma_t[None, :] ** 2)
        inv_sh2 = 1.0 / (item_p.sigma_h[None, :] ** 2)
        inv_st_ai2 = 1.0 / (reg_p.sigma_t_ai**2)

        tau_num = np.sum(
            r0 * inv_st2 * (item_p.beta[None, :] - L)
            + r2 * inv_st_ai2 * (item_p.beta[None, :] + reg_p.delta_t_ai - L),
            axis=1,
        )
        tau_den = prior_prec_tau + np.sum(r0 * inv_st2 + r2 * inv_st_ai2, axis=1)
        tau = tau_num / np.maximum(tau_den, 1e-8)
        if fixed_item_params is None:
            tau = tau - np.mean(tau)
        tau = np.clip(tau, -3.5, 3.5)

        if use_entropy:
            nu_num = np.sum(r0 * inv_sh2 * (H - item_p.gamma[None, :]), axis=1)
            nu_den = prior_prec_nu + np.sum(r0 * inv_sh2, axis=1)
            nu = nu_num / np.maximum(nu_den, 1e-8)
            if fixed_item_params is None:
                nu = nu - np.mean(nu)
            nu = np.clip(nu, -3.0, 3.0)
        else:
            nu = np.zeros_like(nu)

        theta = _update_theta_newton_numpy(
            theta, Y, r0, item_p.a, item_p.b, item_p.c, prior_var=prior_var_theta, n_steps=4
        )

        # Item M-step (unless fixed)
        if fixed_item_params is None:
            theta_sd = float(np.std(theta))
            if theta_sd > 1e-3:
                theta_mean = float(np.mean(theta))
                theta = (theta - theta_mean) / theta_sd
                item_p.b = (item_p.b - theta_mean) / theta_sd
                item_p.a = np.clip(item_p.a * theta_sd, 0.25, 3.50)

            item_p.a, item_p.b = _update_items_newton_numpy(
                item_p.a, item_p.b, item_p.c, Y, r0, theta, n_steps=3
            )

            w_beta0 = r0 * inv_st2
            w_beta2 = r2 * inv_st_ai2
            beta_num = np.sum(
                w_beta0 * (L + tau[:, None]) + w_beta2 * (L + tau[:, None] - reg_p.delta_t_ai),
                axis=0,
            )
            beta_den = np.sum(w_beta0 + w_beta2, axis=0)
            item_p.beta = np.where(beta_den > 1.0, beta_num / np.maximum(beta_den, 1e-8), item_p.beta)

            r0_j = np.sum(r0, axis=0)
            res_t0 = L - (item_p.beta[None, :] - tau[:, None])
            st2 = (np.sum(r0 * (res_t0**2), axis=0) + 0.16) / (r0_j + 1.0)
            item_p.sigma_t = np.clip(np.sqrt(st2), 0.18, 1.50)

            if use_entropy:
                gamma_num = np.sum(r0 * (H - nu[:, None]), axis=0)
                item_p.gamma = np.where(r0_j > 1.0, gamma_num / np.maximum(r0_j, 1e-8), item_p.gamma)
                res_h0 = H - (item_p.gamma[None, :] + nu[:, None])
                sh2 = (np.sum(r0 * (res_h0**2), axis=0) + 0.16) / (r0_j + 1.0)
                item_p.sigma_h = np.clip(np.sqrt(sh2), 0.18, 1.50)

        # Global left-tail & right-tail regime M-step (unless fixed)
        if fixed_regime_params is None:
            med_beta = float(np.median(item_p.beta))
            med_gamma = float(np.median(item_p.gamma))

            w1_sum = float(np.sum(r1))
            if w1_sum > 5.0:
                mu_t_rg = float(np.sum(r1 * L) / w1_sum)
                reg_p.mu_t_rg = min(mu_t_rg, med_beta - 0.55)
                var_t1 = (float(np.sum(r1 * ((L - reg_p.mu_t_rg) ** 2))) + 0.25) / (w1_sum + 2.0)
                reg_p.sigma_t_rg = float(np.clip(math.sqrt(var_t1), 0.18, 1.20))

                if use_entropy:
                    mu_h_rg = float(np.sum(r1 * H) / w1_sum)
                    reg_p.mu_h_rg = min(mu_h_rg, med_gamma - 0.55)
                    var_h1 = (float(np.sum(r1 * ((H - reg_p.mu_h_rg) ** 2))) + 0.25) / (w1_sum + 2.0)
                    reg_p.sigma_h_rg = float(np.clip(math.sqrt(var_h1), 0.18, 1.20))

            w2_sum = float(np.sum(r2))
            if w2_sum > 5.0:
                p_ai = (float(np.sum(r2 * Y)) + 4.5) / (w2_sum + 5.0)
                reg_p.p_ai = float(np.clip(p_ai, 0.70, 0.99))

                rt_shift = L - (item_p.beta[None, :] - tau[:, None])
                delta_t_ai = (float(np.sum(r2 * rt_shift)) + 1.5) / (w2_sum + 2.0)
                reg_p.delta_t_ai = float(np.clip(delta_t_ai, 0.30, 2.50))
                var_t2 = (float(np.sum(r2 * ((rt_shift - reg_p.delta_t_ai) ** 2))) + 0.25) / (w2_sum + 2.0)
                reg_p.sigma_t_ai = float(np.clip(math.sqrt(var_t2), 0.18, 1.20))

                if use_entropy:
                    mu_h_ai = float(np.sum(r2 * H) / w2_sum)
                    reg_p.mu_h_ai = min(mu_h_ai, med_gamma - 0.70)
                    var_h2 = (float(np.sum(r2 * ((H - reg_p.mu_h_ai) ** 2))) + 0.25) / (w2_sum + 2.0)
                    reg_p.sigma_h_ai = float(np.clip(math.sqrt(var_h2), 0.18, 1.20))

        if it > 2 and abs(total_ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            converged = True
            break
        prev_ll = total_ll

    # Final person-level inference (Proposition 3: LRT + 95% profile CIs)
    (
        theta,
        tau,
        nu,
        pi_rg,
        pi_ai,
        post,
        total_ll,
        lrt_rg,
        p_rg,
        lrt_ai,
        p_ai_pval,
        lrt_joint,
        p_joint,
        pi_rg_lo,
        pi_rg_hi,
        pi_ai_lo,
        pi_ai_hi,
    ) = _person_inference_numpy(
        Y,
        L,
        H,
        mask,
        theta,
        tau,
        nu,
        pi_rg,
        pi_ai,
        item_p,
        reg_p,
        compute_ci=compute_ci,
        ci_grid_size=ci_grid_size,
        prior_var_theta=prior_var_theta,
        prior_prec_tau=prior_prec_tau,
        prior_prec_nu=prior_prec_nu,
        use_entropy=use_entropy,
    )

    pi_sol = np.clip(1.0 - pi_rg - pi_ai, 0.0, 1.0)
    post_out = np.where(mask[:, :, None], post, np.nan)
    fit_seconds = time.perf_counter() - t0

    return DTMixIRTResult(
        theta=theta,
        tau=tau,
        nu=nu,
        pi_rg=pi_rg,
        pi_ai=pi_ai,
        pi_sol=pi_sol,
        item_params=item_p,
        regime_params=reg_p,
        posteriors=post_out,
        lrt_rg=lrt_rg,
        p_rg=p_rg,
        lrt_ai=lrt_ai,
        p_ai=p_ai_pval,
        lrt_joint=lrt_joint,
        p_joint=p_joint,
        pi_rg_ci_low=pi_rg_lo,
        pi_rg_ci_high=pi_rg_hi,
        pi_ai_ci_low=pi_ai_lo,
        pi_ai_ci_high=pi_ai_hi,
        log_likelihood=total_ll,
        n_iter=n_iter,
        converged=converged,
        device_used="cpu-numpy",
        fit_seconds=fit_seconds,
    )


def _optimize_persons_constrained_numpy(
    Y: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    theta_init: np.ndarray,
    tau_init: np.ndarray,
    nu_init: np.ndarray,
    pi_rg_init: np.ndarray,
    pi_ai_init: np.ndarray,
    item_p: ItemParameters,
    reg_p: RegimeParameters,
    allow_rg: bool,
    allow_ai: bool,
    prior_var_theta: float,
    prior_prec_tau: float,
    prior_prec_nu: float,
    n_iter: int = 18,
    use_entropy: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimize person parameters under optional boundary constraints (pi_rg=0 and/or pi_ai=0)."""
    theta = theta_init.copy()
    tau = tau_init.copy()
    nu = nu_init.copy() if use_entropy else np.zeros_like(nu_init)
    pi_rg = pi_rg_init.copy() if allow_rg else np.zeros_like(pi_rg_init)
    pi_ai = pi_ai_init.copy() if allow_ai else np.zeros_like(pi_ai_init)

    n_obs_i = np.maximum(mask.astype(np.float64).sum(axis=1), 1.0)
    inv_st2 = 1.0 / (item_p.sigma_t[None, :] ** 2)
    inv_sh2 = 1.0 / (item_p.sigma_h[None, :] ** 2)
    inv_st_ai2 = 1.0 / (reg_p.sigma_t_ai**2)

    for _ in range(n_iter):
        ll = compute_regime_logliks_numpy(
            Y, L, H, theta, tau, nu, item_p, reg_p, use_entropy=use_entropy
        )
        post, _ = _estep_numpy(ll, pi_rg, pi_ai, mask)
        r0, r1, r2 = post[:, :, 0], post[:, :, 1], post[:, :, 2]

        if allow_rg:
            pi_rg = np.clip(np.sum(r1, axis=1) / n_obs_i, 0.0, 0.99)
        if allow_ai:
            pi_ai = np.clip(np.sum(r2, axis=1) / n_obs_i, 0.0, 0.99)
        if allow_rg and allow_ai:
            s = pi_rg + pi_ai
            over = s > 0.995
            if np.any(over):
                pi_rg[over] *= 0.995 / s[over]
                pi_ai[over] *= 0.995 / s[over]

        tau_num = np.sum(
            r0 * inv_st2 * (item_p.beta[None, :] - L)
            + r2 * inv_st_ai2 * (item_p.beta[None, :] + reg_p.delta_t_ai - L),
            axis=1,
        )
        tau_den = prior_prec_tau + np.sum(r0 * inv_st2 + r2 * inv_st_ai2, axis=1)
        tau = np.clip(tau_num / np.maximum(tau_den, 1e-8), -3.5, 3.5)

        if use_entropy:
            nu_num = np.sum(r0 * inv_sh2 * (H - item_p.gamma[None, :]), axis=1)
            nu_den = prior_prec_nu + np.sum(r0 * inv_sh2, axis=1)
            nu = np.clip(nu_num / np.maximum(nu_den, 1e-8), -3.0, 3.0)

        theta = _update_theta_newton_numpy(
            theta, Y, r0, item_p.a, item_p.b, item_p.c, prior_var=prior_var_theta, n_steps=3
        )

    ll = compute_regime_logliks_numpy(
        Y, L, H, theta, tau, nu, item_p, reg_p, use_entropy=use_entropy
    )
    post, person_ll = _estep_numpy(ll, pi_rg, pi_ai, mask)
    pen = (
        -0.5 * (theta**2) / prior_var_theta
        - 0.5 * prior_prec_tau * (tau**2)
        - 0.5 * prior_prec_nu * (nu**2)
    )
    return theta, tau, nu, pi_rg, pi_ai, post, person_ll + pen


def _profile_ci_numpy(
    ll_full: np.ndarray,
    ll_null: np.ndarray,
    pi_target: np.ndarray,
    pi_other: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    grid_size: int,
    pen_full: np.ndarray,
    pen_null: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact 95% profile-likelihood confidence bounds on a vectorized grid."""
    N = ll_full.shape[0]
    grid = np.linspace(0.0, 0.99, grid_size, dtype=np.float64)
    other_idx = 2 if target_idx == 1 else 1

    # Evaluate profile objective at both full-MLE and null-MLE nuisance anchors
    # so the profile curve coincides exactly with full and boundary-null LRT endpoints.
    def _eval_slice(ll_tensor: np.ndarray, pen: np.ndarray, p_oth: np.ndarray) -> np.ndarray:
        out = np.empty((N, grid_size), dtype=np.float64)
        for g_idx, g in enumerate(grid):
            oth = np.minimum(p_oth, max(0.0, 0.995 - g))
            sol = np.clip(1.0 - g - oth, 0.0, 1.0)
            if target_idx == 1:
                w = np.stack([sol, np.full(N, g), oth], axis=-1)
            else:
                w = np.stack([sol, oth, np.full(N, g)], axis=-1)
            log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), -1e9)
            joint = ll_tensor + log_w[:, None, :]
            m = np.max(joint, axis=-1, keepdims=True)
            lse = m + np.log(np.sum(np.exp(joint - m), axis=-1, keepdims=True))
            out[:, g_idx] = np.sum(np.where(mask, lse[:, :, 0], 0.0), axis=1) + pen
        return out

    prof_full = _eval_slice(ll_full, pen_full, pi_other)
    prof_null = _eval_slice(ll_null, pen_null, pi_other)
    prof = np.maximum(prof_full, prof_null)

    max_prof = np.max(prof, axis=1, keepdims=True)
    # 95% profile-likelihood chi2_1 cutoff = 3.841458820694124
    in_ci = (2.0 * (max_prof - prof)) <= 3.841458820694124
    lo = np.zeros(N, dtype=np.float64)
    hi = np.zeros(N, dtype=np.float64)
    for i in range(N):
        idx = np.flatnonzero(in_ci[i])
        if idx.size == 0:
            lo[i] = float(pi_target[i])
            hi[i] = float(pi_target[i])
        else:
            lo[i] = float(min(grid[idx[0]], pi_target[i]))
            hi[i] = float(max(grid[idx[-1]], pi_target[i]))
    return lo, hi


def _person_inference_numpy(
    Y: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    tau: np.ndarray,
    nu: np.ndarray,
    pi_rg: np.ndarray,
    pi_ai: np.ndarray,
    item_p: ItemParameters,
    reg_p: RegimeParameters,
    compute_ci: bool,
    ci_grid_size: int,
    prior_var_theta: float,
    prior_prec_tau: float,
    prior_prec_nu: float,
    use_entropy: bool = True,
) -> Tuple:
    """Compute boundary-corrected LRTs and 95% profile CIs in NumPy."""
    th_n0, ta_n0, nu_n0, _, _, _, obj_clean = _optimize_persons_constrained_numpy(
        Y, L, H, mask, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p,
        allow_rg=False, allow_ai=False,
        prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        use_entropy=use_entropy,
    )
    th_nrg, ta_nrg, nu_nrg, _, pi_ai_nrg, _, obj_nrg = _optimize_persons_constrained_numpy(
        Y, L, H, mask, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p,
        allow_rg=False, allow_ai=True,
        prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        use_entropy=use_entropy,
    )
    th_nai, ta_nai, nu_nai, pi_rg_nai, _, _, obj_nai = _optimize_persons_constrained_numpy(
        Y, L, H, mask, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p,
        allow_rg=True, allow_ai=False,
        prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        use_entropy=use_entropy,
    )
    th_f, ta_f, nu_f, prg_f, pai_f, post_f, obj_f = _optimize_persons_constrained_numpy(
        Y, L, H, mask, theta, tau, nu, pi_rg, pi_ai, item_p, reg_p,
        allow_rg=True, allow_ai=True,
        prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        use_entropy=use_entropy,
    )

    best_obj = obj_f.copy()
    for cand_obj, c_th, c_ta, c_nu, c_prg, c_pai in [
        (obj_nai, th_nai, ta_nai, nu_nai, pi_rg_nai, np.zeros_like(pai_f)),
        (obj_nrg, th_nrg, ta_nrg, nu_nrg, np.zeros_like(prg_f), pi_ai_nrg),
        (obj_clean, th_n0, ta_n0, nu_n0, np.zeros_like(prg_f), np.zeros_like(pai_f)),
    ]:
        better = cand_obj > best_obj + 1e-10
        if np.any(better):
            best_obj[better] = cand_obj[better]
            th_f[better] = c_th[better]
            ta_f[better] = c_ta[better]
            nu_f[better] = c_nu[better]
            prg_f[better] = c_prg[better]
            pai_f[better] = c_pai[better]

    ll_full = compute_regime_logliks_numpy(
        Y, L, H, th_f, ta_f, nu_f, item_p, reg_p, use_entropy=use_entropy
    )
    post_f, person_ll_raw = _estep_numpy(ll_full, prg_f, pai_f, mask)
    total_ll = float(np.sum(person_ll_raw))

    lrt_rg = np.maximum(0.0, 2.0 * (best_obj - obj_nrg))
    lrt_ai = np.maximum(0.0, 2.0 * (best_obj - obj_nai))
    lrt_joint = np.maximum(0.0, 2.0 * (best_obj - obj_clean))

    p_rg = np.where(lrt_rg > 0.0, 0.5 * chi2.sf(lrt_rg, df=1), 1.0)
    p_ai = np.where(lrt_ai > 0.0, 0.5 * chi2.sf(lrt_ai, df=1), 1.0)
    p_joint = np.where(
        lrt_joint > 0.0,
        0.5 * chi2.sf(lrt_joint, df=1) + 0.25 * chi2.sf(lrt_joint, df=2),
        1.0,
    )

    if compute_ci:
        pen_f = (
            -0.5 * (th_f**2) / prior_var_theta
            - 0.5 * prior_prec_tau * (ta_f**2)
            - 0.5 * prior_prec_nu * (nu_f**2)
        )
        pen_nrg = (
            -0.5 * (th_nrg**2) / prior_var_theta
            - 0.5 * prior_prec_tau * (ta_nrg**2)
            - 0.5 * prior_prec_nu * (nu_nrg**2)
        )
        pen_nai = (
            -0.5 * (th_nai**2) / prior_var_theta
            - 0.5 * prior_prec_tau * (ta_nai**2)
            - 0.5 * prior_prec_nu * (nu_nai**2)
        )
        ll_nrg = compute_regime_logliks_numpy(
            Y, L, H, th_nrg, ta_nrg, nu_nrg, item_p, reg_p, use_entropy=use_entropy
        )
        ll_nai = compute_regime_logliks_numpy(
            Y, L, H, th_nai, ta_nai, nu_nai, item_p, reg_p, use_entropy=use_entropy
        )

        pi_rg_lo, pi_rg_hi = _profile_ci_numpy(
            ll_full, ll_nrg, prg_f, pai_f, mask, target_idx=1, grid_size=ci_grid_size,
            pen_full=pen_f, pen_null=pen_nrg,
        )
        pi_ai_lo, pi_ai_hi = _profile_ci_numpy(
            ll_full, ll_nai, pai_f, prg_f, mask, target_idx=2, grid_size=ci_grid_size,
            pen_full=pen_f, pen_null=pen_nai,
        )
    else:
        pi_rg_lo = pi_rg_hi = pi_ai_lo = pi_ai_hi = None

    return (
        th_f, ta_f, nu_f, prg_f, pai_f, post_f, total_ll,
        lrt_rg, p_rg, lrt_ai, p_ai, lrt_joint, p_joint,
        pi_rg_lo, pi_rg_hi, pi_ai_lo, pi_ai_hi,
    )


def _compute_logliks_torch(
    Y_t, L_t, H_t, theta_t, tau_t, nu_t, a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p: RegimeParameters
):
    """Fused PyTorch GPU/CPU trivariate component log-density evaluation."""
    z = a_t.unsqueeze(0) * (theta_t.unsqueeze(1) - b_t.unsqueeze(0))
    s = torch.sigmoid(torch.clamp(z, -35.0, 35.0))
    c_row = c_t.unsqueeze(0)
    p0 = torch.clamp(c_row + (1.0 - c_row) * s, _EPS, 1.0 - _EPS)
    ll_y0 = Y_t * torch.log(p0) + (1.0 - Y_t) * torch.log(1.0 - p0)

    mu_t0 = beta_t.unsqueeze(0) - tau_t.unsqueeze(1)
    st_row = st_t.unsqueeze(0)
    ll_t0 = -0.5 * (_LOG_2PI + 2.0 * torch.log(st_row) + ((L_t - mu_t0) / st_row) ** 2)

    mu_h0 = gamma_t.unsqueeze(0) + nu_t.unsqueeze(1)
    sh_row = sh_t.unsqueeze(0)
    ll_h0 = -0.5 * (_LOG_2PI + 2.0 * torch.log(sh_row) + ((H_t - mu_h0) / sh_row) ** 2)
    ll0 = ll_y0 + ll_t0 + ll_h0

    p1 = torch.clamp(c_row, _EPS, 1.0 - _EPS)
    ll_y1 = Y_t * torch.log(p1) + (1.0 - Y_t) * torch.log(1.0 - p1)
    ll_t1 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(reg_p.sigma_t_rg)
        + ((L_t - reg_p.mu_t_rg) / reg_p.sigma_t_rg) ** 2
    )
    ll_h1 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(reg_p.sigma_h_rg)
        + ((H_t - reg_p.mu_h_rg) / reg_p.sigma_h_rg) ** 2
    )
    ll1 = ll_y1 + ll_t1 + ll_h1

    p2 = float(np.clip(reg_p.p_ai, _EPS, 1.0 - _EPS))
    ll_y2 = Y_t * math.log(p2) + (1.0 - Y_t) * math.log(1.0 - p2)
    mu_t2 = mu_t0 + reg_p.delta_t_ai
    ll_t2 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(reg_p.sigma_t_ai)
        + ((L_t - mu_t2) / reg_p.sigma_t_ai) ** 2
    )
    ll_h2 = -0.5 * (
        _LOG_2PI
        + 2.0 * math.log(reg_p.sigma_h_ai)
        + ((H_t - reg_p.mu_h_ai) / reg_p.sigma_h_ai) ** 2
    )
    ll2 = ll_y2 + ll_t2 + ll_h2

    return torch.stack([ll0, ll1, ll2], dim=-1)


def _estep_torch(ll_t, pi_rg_t, pi_ai_t, mask_t):
    """Fused PyTorch E-step returning posteriors (N, J, 3) and person log-likelihoods (N,)."""
    pi_sol = torch.clamp(1.0 - pi_rg_t - pi_ai_t, 0.0, 1.0)
    w = torch.stack([pi_sol, torch.clamp(pi_rg_t, 0.0, 1.0), torch.clamp(pi_ai_t, 0.0, 1.0)], dim=-1)
    log_w = torch.where(w > 0.0, torch.log(torch.clamp(w, min=1e-300)), torch.full_like(w, -1e9))
    joint = ll_t + log_w.unsqueeze(1)
    lse = torch.logsumexp(joint, dim=-1, keepdim=True)
    post = torch.exp(joint - lse)
    post = torch.where(mask_t.unsqueeze(-1), post, torch.zeros_like(post))
    person_ll = torch.sum(torch.where(mask_t, lse.squeeze(-1), torch.zeros_like(lse.squeeze(-1))), dim=1)
    return post, person_ll


def _update_theta_newton_torch(theta_t, Y_t, r0_t, a_t, b_t, c_t, prior_var: float, n_steps: int = 4):
    """Batched PyTorch safeguarded exact observed Newton-Raphson update."""
    th = theta_t.clone()
    inv_pv = 1.0 / prior_var
    a_row = a_t.unsqueeze(0)
    b_row = b_t.unsqueeze(0)
    c_row = c_t.unsqueeze(0)
    a2_row = a_row * a_row
    for _ in range(n_steps):
        z = a_row * (th.unsqueeze(1) - b_row)
        s = torch.sigmoid(torch.clamp(z, -35.0, 35.0))
        p = torch.clamp(c_row + (1.0 - c_row) * s, _EPS, 1.0 - _EPS)
        grad = torch.sum(r0_t * a_row * (s / p) * (Y_t - p), dim=1) - th * inv_pv
        s1ms = s * (1.0 - s)
        obs_h = torch.sum(r0_t * a2_row * s1ms * (1.0 - (c_row * Y_t) / (p * p)), dim=1) + inv_pv
        fish_h = torch.sum(r0_t * a2_row * (1.0 - c_row) * (s * s1ms) / p, dim=1) + inv_pv
        hess = torch.maximum(obs_h, torch.clamp(0.25 * fish_h, min=inv_pv))
        step = torch.clamp(grad / torch.clamp(hess, min=1e-8), -0.85, 0.85)
        th = torch.clamp(th + step, -4.5, 4.5)
    return th


def _update_items_newton_torch(a_t, b_t, c_t, Y_t, r0_t, theta_t, n_steps: int = 3):
    """Batched PyTorch damped Newton-Raphson update for all item parameters simultaneously."""
    a_cur = a_t.clone()
    b_cur = b_t.clone()
    th_col = theta_t.unsqueeze(1)
    c_row = c_t.unsqueeze(0)
    for _ in range(n_steps):
        z = a_cur.unsqueeze(0) * (th_col - b_cur.unsqueeze(0))
        s = torch.sigmoid(torch.clamp(z, -35.0, 35.0))
        p = torch.clamp(c_row + (1.0 - c_row) * s, _EPS, 1.0 - _EPS)
        common = (1.0 - c_row) * s * (1.0 - s)
        dp_da = common * (th_col - b_cur.unsqueeze(0))
        dp_db = -common * a_cur.unsqueeze(0)
        denom = p * (1.0 - p)
        resid = (Y_t - p) / denom

        g_a = torch.sum(r0_t * resid * dp_da, dim=0) - (torch.log(a_cur) / (0.25 * a_cur))
        g_b = torch.sum(r0_t * resid * dp_db, dim=0) - b_cur / 6.25

        h_aa = torch.sum(r0_t * (dp_da * dp_da) / denom, dim=0) + 1.0 / (0.25 * a_cur * a_cur) + 1e-4
        h_bb = torch.sum(r0_t * (dp_db * dp_db) / denom, dim=0) + 1.0 / 6.25 + 1e-4
        h_ab = torch.sum(r0_t * (dp_da * dp_db) / denom, dim=0)

        det = torch.clamp(h_aa * h_bb - h_ab * h_ab, min=1e-6)
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det

        a_cur = torch.clamp(a_cur + torch.clamp(da, -0.35, 0.35), 0.25, 3.50)
        b_cur = torch.clamp(b_cur + torch.clamp(db, -0.50, 0.50), -4.0, 4.0)
    return a_cur, b_cur


def _optimize_persons_constrained_torch(
    Y_t, L_t, H_t, mask_t, n_obs_i,
    theta_init, tau_init, nu_init, pi_rg_init, pi_ai_init,
    a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p: RegimeParameters,
    allow_rg: bool, allow_ai: bool,
    prior_var_theta: float, prior_prec_tau: float, prior_prec_nu: float,
    n_iter: int = 18,
):
    """GPU-parallel constrained person optimizer for boundary-corrected LRTs."""
    theta_t = theta_init.clone()
    tau_t = tau_init.clone()
    nu_t = nu_init.clone()
    pi_rg_t = pi_rg_init.clone() if allow_rg else torch.zeros_like(pi_rg_init)
    pi_ai_t = pi_ai_init.clone() if allow_ai else torch.zeros_like(pi_ai_init)

    inv_st2 = 1.0 / (st_t.unsqueeze(0) ** 2)
    inv_sh2 = 1.0 / (sh_t.unsqueeze(0) ** 2)
    inv_st_ai2 = 1.0 / (reg_p.sigma_t_ai**2)

    for _ in range(n_iter):
        ll_t = _compute_logliks_torch(
            Y_t, L_t, H_t, theta_t, tau_t, nu_t, a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p
        )
        post_t, _ = _estep_torch(ll_t, pi_rg_t, pi_ai_t, mask_t)
        r0, r1, r2 = post_t[:, :, 0], post_t[:, :, 1], post_t[:, :, 2]

        if allow_rg:
            pi_rg_t = torch.clamp(torch.sum(r1, dim=1) / n_obs_i, 0.0, 0.99)
        if allow_ai:
            pi_ai_t = torch.clamp(torch.sum(r2, dim=1) / n_obs_i, 0.0, 0.99)
        if allow_rg and allow_ai:
            s = pi_rg_t + pi_ai_t
            scale = torch.where(s > 0.995, 0.995 / torch.clamp(s, min=_EPS), torch.ones_like(s))
            pi_rg_t = pi_rg_t * scale
            pi_ai_t = pi_ai_t * scale

        tau_num = torch.sum(
            r0 * inv_st2 * (beta_t.unsqueeze(0) - L_t)
            + r2 * inv_st_ai2 * (beta_t.unsqueeze(0) + reg_p.delta_t_ai - L_t),
            dim=1,
        )
        tau_den = prior_prec_tau + torch.sum(r0 * inv_st2 + r2 * inv_st_ai2, dim=1)
        tau_t = torch.clamp(tau_num / torch.clamp(tau_den, min=1e-8), -3.5, 3.5)

        nu_num = torch.sum(r0 * inv_sh2 * (H_t - gamma_t.unsqueeze(0)), dim=1)
        nu_den = prior_prec_nu + torch.sum(r0 * inv_sh2, dim=1)
        nu_t = torch.clamp(nu_num / torch.clamp(nu_den, min=1e-8), -3.0, 3.0)

        theta_t = _update_theta_newton_torch(
            theta_t, Y_t, r0, a_t, b_t, c_t, prior_var=prior_var_theta, n_steps=3
        )

    ll_t = _compute_logliks_torch(
        Y_t, L_t, H_t, theta_t, tau_t, nu_t, a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p
    )
    post_t, person_ll = _estep_torch(ll_t, pi_rg_t, pi_ai_t, mask_t)
    pen = (
        -0.5 * (theta_t**2) / prior_var_theta
        - 0.5 * prior_prec_tau * (tau_t**2)
        - 0.5 * prior_prec_nu * (nu_t**2)
    )
    return theta_t, tau_t, nu_t, pi_rg_t, pi_ai_t, post_t, person_ll + pen, ll_t, pen


def _profile_ci_torch(
    ll_full_t, ll_null_t, pi_target_t, pi_other_t, mask_t,
    target_idx: int, grid_size: int, pen_full_t, pen_null_t,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-parallelized 95% profile-likelihood confidence intervals across a grid."""
    N = ll_full_t.shape[0]
    dev = ll_full_t.device
    dt = ll_full_t.dtype
    grid = torch.linspace(0.0, 0.99, grid_size, device=dev, dtype=dt)
    mask_f = mask_t.to(dt)

    def _eval_slice(ll_tensor, pen_t, p_oth_t):
        out = torch.empty((N, grid_size), device=dev, dtype=dt)
        for g_idx in range(grid_size):
            g = grid[g_idx]
            oth = torch.minimum(p_oth_t, torch.clamp(0.995 - g, min=0.0))
            sol = torch.clamp(1.0 - g - oth, 0.0, 1.0)
            g_vec = torch.full((N,), float(g.item()), device=dev, dtype=dt)
            if target_idx == 1:
                w = torch.stack([sol, g_vec, oth], dim=-1)
            else:
                w = torch.stack([sol, oth, g_vec], dim=-1)
            log_w = torch.where(w > 0.0, torch.log(torch.clamp(w, min=1e-300)), torch.full_like(w, -1e9))
            lse = torch.logsumexp(ll_tensor + log_w.unsqueeze(1), dim=-1)
            out[:, g_idx] = torch.sum(lse * mask_f, dim=1) + pen_t
        return out

    prof = torch.maximum(
        _eval_slice(ll_full_t, pen_full_t, pi_other_t),
        _eval_slice(ll_null_t, pen_null_t, pi_other_t),
    )
    max_prof = torch.max(prof, dim=1, keepdim=True).values
    in_ci = ((2.0 * (max_prof - prof)) <= 3.841458820694124).detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()
    pt_np = pi_target_t.detach().cpu().numpy()

    lo = np.zeros(N, dtype=np.float64)
    hi = np.zeros(N, dtype=np.float64)
    for i in range(N):
        idx = np.flatnonzero(in_ci[i])
        if idx.size == 0:
            lo[i] = float(pt_np[i])
            hi[i] = float(pt_np[i])
        else:
            lo[i] = float(min(grid_np[idx[0]], pt_np[i]))
            hi[i] = float(max(grid_np[idx[-1]], pt_np[i]))
    return lo, hi


def _fit_torch_backend(
    Y: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    c_arr: np.ndarray,
    device_tag: str,
    max_iter: int,
    tol: float,
    fixed_item_params: Optional[ItemParameters],
    fixed_regime_params: Optional[RegimeParameters],
    compute_ci: bool,
    ci_grid_size: int,
    prior_var_theta: float,
    prior_prec_tau: float,
    prior_prec_nu: float,
) -> DTMixIRTResult:
    """PyTorch CUDA/CPU vectorized float64 DT-MixIRT estimation backend."""
    t0 = time.perf_counter()
    torch_dev = torch.device("cuda:0" if device_tag.startswith("cuda") else "cpu")
    dt = torch.float64

    theta0, tau0, nu0, pi_rg0, pi_ai0, item_p, reg_p = _initialize_parameters(Y, L, H, mask, c_arr)
    if fixed_item_params is not None:
        item_p = ItemParameters(
            a=fixed_item_params.a.copy(),
            b=fixed_item_params.b.copy(),
            c=fixed_item_params.c.copy(),
            beta=fixed_item_params.beta.copy(),
            sigma_t=fixed_item_params.sigma_t.copy(),
            gamma=fixed_item_params.gamma.copy(),
            sigma_h=fixed_item_params.sigma_h.copy(),
        )
    if fixed_regime_params is not None:
        reg_p = RegimeParameters(**fixed_regime_params.__dict__)

    with torch.no_grad():
        Y_t = torch.as_tensor(Y, device=torch_dev, dtype=dt)
        L_t = torch.as_tensor(L, device=torch_dev, dtype=dt)
        H_t = torch.as_tensor(H, device=torch_dev, dtype=dt)
        mask_t = torch.as_tensor(mask, device=torch_dev, dtype=torch.bool)
        n_obs_i = torch.clamp(mask_t.to(dt).sum(dim=1), min=1.0)

        theta_t = torch.as_tensor(theta0, device=torch_dev, dtype=dt)
        tau_t = torch.as_tensor(tau0, device=torch_dev, dtype=dt)
        nu_t = torch.as_tensor(nu0, device=torch_dev, dtype=dt)
        pi_rg_t = torch.as_tensor(pi_rg0, device=torch_dev, dtype=dt)
        pi_ai_t = torch.as_tensor(pi_ai0, device=torch_dev, dtype=dt)

        a_t = torch.as_tensor(item_p.a, device=torch_dev, dtype=dt)
        b_t = torch.as_tensor(item_p.b, device=torch_dev, dtype=dt)
        c_t = torch.as_tensor(item_p.c, device=torch_dev, dtype=dt)
        beta_t = torch.as_tensor(item_p.beta, device=torch_dev, dtype=dt)
        st_t = torch.as_tensor(item_p.sigma_t, device=torch_dev, dtype=dt)
        gamma_t = torch.as_tensor(item_p.gamma, device=torch_dev, dtype=dt)
        sh_t = torch.as_tensor(item_p.sigma_h, device=torch_dev, dtype=dt)

        prev_ll = -np.inf
        converged = False
        n_iter = 0

        for it in range(1, max_iter + 1):
            n_iter = it
            ll_t = _compute_logliks_torch(
                Y_t, L_t, H_t, theta_t, tau_t, nu_t, a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p
            )
            post_t, person_ll = _estep_torch(ll_t, pi_rg_t, pi_ai_t, mask_t)
            total_ll = float(person_ll.sum().item())

            r0 = post_t[:, :, 0]
            r1 = post_t[:, :, 1]
            r2 = post_t[:, :, 2]

            # Person M-step
            pi_rg_t = torch.clamp(torch.sum(r1, dim=1) / n_obs_i, 0.0, 0.99)
            pi_ai_t = torch.clamp(torch.sum(r2, dim=1) / n_obs_i, 0.0, 0.99)
            s_pi = pi_rg_t + pi_ai_t
            scale = torch.where(s_pi > 0.995, 0.995 / torch.clamp(s_pi, min=_EPS), torch.ones_like(s_pi))
            pi_rg_t = pi_rg_t * scale
            pi_ai_t = pi_ai_t * scale

            inv_st2 = 1.0 / (st_t.unsqueeze(0) ** 2)
            inv_sh2 = 1.0 / (sh_t.unsqueeze(0) ** 2)
            inv_st_ai2 = 1.0 / (reg_p.sigma_t_ai**2)

            tau_num = torch.sum(
                r0 * inv_st2 * (beta_t.unsqueeze(0) - L_t)
                + r2 * inv_st_ai2 * (beta_t.unsqueeze(0) + reg_p.delta_t_ai - L_t),
                dim=1,
            )
            tau_den = prior_prec_tau + torch.sum(r0 * inv_st2 + r2 * inv_st_ai2, dim=1)
            tau_t = tau_num / torch.clamp(tau_den, min=1e-8)
            if fixed_item_params is None:
                tau_t = tau_t - tau_t.mean()
            tau_t = torch.clamp(tau_t, -3.5, 3.5)

            nu_num = torch.sum(r0 * inv_sh2 * (H_t - gamma_t.unsqueeze(0)), dim=1)
            nu_den = prior_prec_nu + torch.sum(r0 * inv_sh2, dim=1)
            nu_t = nu_num / torch.clamp(nu_den, min=1e-8)
            if fixed_item_params is None:
                nu_t = nu_t - nu_t.mean()
            nu_t = torch.clamp(nu_t, -3.0, 3.0)

            theta_t = _update_theta_newton_torch(
                theta_t, Y_t, r0, a_t, b_t, c_t, prior_var=prior_var_theta, n_steps=4
            )

            # Item M-step
            if fixed_item_params is None:
                theta_sd = float(theta_t.std(unbiased=False).item())
                if theta_sd > 1e-3:
                    theta_mean = float(theta_t.mean().item())
                    theta_t = (theta_t - theta_mean) / theta_sd
                    b_t = (b_t - theta_mean) / theta_sd
                    a_t = torch.clamp(a_t * theta_sd, 0.25, 3.50)

                a_t, b_t = _update_items_newton_torch(a_t, b_t, c_t, Y_t, r0, theta_t, n_steps=3)

                w_beta0 = r0 * inv_st2
                w_beta2 = r2 * inv_st_ai2
                beta_num = torch.sum(
                    w_beta0 * (L_t + tau_t.unsqueeze(1))
                    + w_beta2 * (L_t + tau_t.unsqueeze(1) - reg_p.delta_t_ai),
                    dim=0,
                )
                beta_den = torch.sum(w_beta0 + w_beta2, dim=0)
                beta_t = torch.where(beta_den > 1.0, beta_num / torch.clamp(beta_den, min=1e-8), beta_t)

                r0_j = torch.sum(r0, dim=0)
                res_t0 = L_t - (beta_t.unsqueeze(0) - tau_t.unsqueeze(1))
                st2 = (torch.sum(r0 * (res_t0**2), dim=0) + 0.16) / (r0_j + 1.0)
                st_t = torch.clamp(torch.sqrt(st2), 0.18, 1.50)

                gamma_num = torch.sum(r0 * (H_t - nu_t.unsqueeze(1)), dim=0)
                gamma_t = torch.where(r0_j > 1.0, gamma_num / torch.clamp(r0_j, min=1e-8), gamma_t)
                res_h0 = H_t - (gamma_t.unsqueeze(0) + nu_t.unsqueeze(1))
                sh2 = (torch.sum(r0 * (res_h0**2), dim=0) + 0.16) / (r0_j + 1.0)
                sh_t = torch.clamp(torch.sqrt(sh2), 0.18, 1.50)

            # Global regime M-step
            if fixed_regime_params is None:
                med_beta = float(torch.median(beta_t).item())
                med_gamma = float(torch.median(gamma_t).item())

                w1_sum = float(r1.sum().item())
                if w1_sum > 5.0:
                    mu_t_rg = float((r1 * L_t).sum().item() / w1_sum)
                    reg_p.mu_t_rg = min(mu_t_rg, med_beta - 0.55)
                    var_t1 = (float((r1 * ((L_t - reg_p.mu_t_rg) ** 2)).sum().item()) + 0.25) / (w1_sum + 2.0)
                    reg_p.sigma_t_rg = float(np.clip(math.sqrt(var_t1), 0.18, 1.20))

                    mu_h_rg = float((r1 * H_t).sum().item() / w1_sum)
                    reg_p.mu_h_rg = min(mu_h_rg, med_gamma - 0.55)
                    var_h1 = (float((r1 * ((H_t - reg_p.mu_h_rg) ** 2)).sum().item()) + 0.25) / (w1_sum + 2.0)
                    reg_p.sigma_h_rg = float(np.clip(math.sqrt(var_h1), 0.18, 1.20))

                w2_sum = float(r2.sum().item())
                if w2_sum > 5.0:
                    p_ai = (float((r2 * Y_t).sum().item()) + 4.5) / (w2_sum + 5.0)
                    reg_p.p_ai = float(np.clip(p_ai, 0.70, 0.99))

                    rt_shift = L_t - (beta_t.unsqueeze(0) - tau_t.unsqueeze(1))
                    delta_t_ai = (float((r2 * rt_shift).sum().item()) + 1.5) / (w2_sum + 2.0)
                    reg_p.delta_t_ai = float(np.clip(delta_t_ai, 0.30, 2.50))
                    var_t2 = (float((r2 * ((rt_shift - reg_p.delta_t_ai) ** 2)).sum().item()) + 0.25) / (w2_sum + 2.0)
                    reg_p.sigma_t_ai = float(np.clip(math.sqrt(var_t2), 0.18, 1.20))

                    mu_h_ai = float((r2 * H_t).sum().item() / w2_sum)
                    reg_p.mu_h_ai = min(mu_h_ai, med_gamma - 0.70)
                    var_h2 = (float((r2 * ((H_t - reg_p.mu_h_ai) ** 2)).sum().item()) + 0.25) / (w2_sum + 2.0)
                    reg_p.sigma_h_ai = float(np.clip(math.sqrt(var_h2), 0.18, 1.20))

            if it > 2 and abs(total_ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
                converged = True
                break
            prev_ll = total_ll

        # Proposition 3: GPU-parallel boundary-corrected LRT & 95% profile CIs
        th_n0, ta_n0, nu_n0, _, _, _, obj_clean, _, _ = _optimize_persons_constrained_torch(
            Y_t, L_t, H_t, mask_t, n_obs_i, theta_t, tau_t, nu_t, pi_rg_t, pi_ai_t,
            a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p,
            allow_rg=False, allow_ai=False,
            prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        )
        th_nrg, ta_nrg, nu_nrg, _, pai_nrg, _, obj_nrg, ll_nrg, pen_nrg = _optimize_persons_constrained_torch(
            Y_t, L_t, H_t, mask_t, n_obs_i, theta_t, tau_t, nu_t, pi_rg_t, pi_ai_t,
            a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p,
            allow_rg=False, allow_ai=True,
            prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        )
        th_nai, ta_nai, nu_nai, prg_nai, _, _, obj_nai, ll_nai, pen_nai = _optimize_persons_constrained_torch(
            Y_t, L_t, H_t, mask_t, n_obs_i, theta_t, tau_t, nu_t, pi_rg_t, pi_ai_t,
            a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p,
            allow_rg=True, allow_ai=False,
            prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        )
        th_f, ta_f, nu_f, prg_f, pai_f, _, obj_f, _, _ = _optimize_persons_constrained_torch(
            Y_t, L_t, H_t, mask_t, n_obs_i, theta_t, tau_t, nu_t, pi_rg_t, pi_ai_t,
            a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p,
            allow_rg=True, allow_ai=True,
            prior_var_theta=prior_var_theta, prior_prec_tau=prior_prec_tau, prior_prec_nu=prior_prec_nu,
        )

        best_obj = obj_f.clone()
        zeros_n = torch.zeros_like(prg_f)
        for cand_obj, c_th, c_ta, c_nu, c_prg, c_pai in [
            (obj_nai, th_nai, ta_nai, nu_nai, prg_nai, zeros_n),
            (obj_nrg, th_nrg, ta_nrg, nu_nrg, zeros_n, pai_nrg),
            (obj_clean, th_n0, ta_n0, nu_n0, zeros_n, zeros_n),
        ]:
            better = cand_obj > (best_obj + 1e-10)
            best_obj = torch.where(better, cand_obj, best_obj)
            th_f = torch.where(better, c_th, th_f)
            ta_f = torch.where(better, c_ta, ta_f)
            nu_f = torch.where(better, c_nu, nu_f)
            prg_f = torch.where(better, c_prg, prg_f)
            pai_f = torch.where(better, c_pai, pai_f)

        ll_full_t = _compute_logliks_torch(
            Y_t, L_t, H_t, th_f, ta_f, nu_f, a_t, b_t, c_t, beta_t, st_t, gamma_t, sh_t, reg_p
        )
        post_f_t, person_ll_raw = _estep_torch(ll_full_t, prg_f, pai_f, mask_t)
        total_ll = float(person_ll_raw.sum().item())

        lrt_rg_np = torch.clamp(2.0 * (best_obj - obj_nrg), min=0.0).detach().cpu().numpy()
        lrt_ai_np = torch.clamp(2.0 * (best_obj - obj_nai), min=0.0).detach().cpu().numpy()
        lrt_joint_np = torch.clamp(2.0 * (best_obj - obj_clean), min=0.0).detach().cpu().numpy()

        p_rg = np.where(lrt_rg_np > 0.0, 0.5 * chi2.sf(lrt_rg_np, df=1), 1.0)
        p_ai = np.where(lrt_ai_np > 0.0, 0.5 * chi2.sf(lrt_ai_np, df=1), 1.0)
        p_joint = np.where(
            lrt_joint_np > 0.0,
            0.5 * chi2.sf(lrt_joint_np, df=1) + 0.25 * chi2.sf(lrt_joint_np, df=2),
            1.0,
        )

        if compute_ci:
            pen_f = (
                -0.5 * (th_f**2) / prior_var_theta
                - 0.5 * prior_prec_tau * (ta_f**2)
                - 0.5 * prior_prec_nu * (nu_f**2)
            )
            pi_rg_lo, pi_rg_hi = _profile_ci_torch(
                ll_full_t, ll_nrg, prg_f, pai_f, mask_t,
                target_idx=1, grid_size=ci_grid_size, pen_full_t=pen_f, pen_null_t=pen_nrg,
            )
            pi_ai_lo, pi_ai_hi = _profile_ci_torch(
                ll_full_t, ll_nai, pai_f, prg_f, mask_t,
                target_idx=2, grid_size=ci_grid_size, pen_full_t=pen_f, pen_null_t=pen_nai,
            )
        else:
            pi_rg_lo = pi_rg_hi = pi_ai_lo = pi_ai_hi = None

        if device_tag.startswith("cuda"):
            torch.cuda.synchronize()

        item_out = ItemParameters(
            a=a_t.detach().cpu().numpy(),
            b=b_t.detach().cpu().numpy(),
            c=c_t.detach().cpu().numpy(),
            beta=beta_t.detach().cpu().numpy(),
            sigma_t=st_t.detach().cpu().numpy(),
            gamma=gamma_t.detach().cpu().numpy(),
            sigma_h=sh_t.detach().cpu().numpy(),
        )
        theta_out = th_f.detach().cpu().numpy()
        tau_out = ta_f.detach().cpu().numpy()
        nu_out = nu_f.detach().cpu().numpy()
        pi_rg_out = prg_f.detach().cpu().numpy()
        pi_ai_out = pai_f.detach().cpu().numpy()
        pi_sol_out = np.clip(1.0 - pi_rg_out - pi_ai_out, 0.0, 1.0)
        post_out = np.where(mask[:, :, None], post_f_t.detach().cpu().numpy(), np.nan)

    fit_seconds = time.perf_counter() - t0
    return DTMixIRTResult(
        theta=theta_out,
        tau=tau_out,
        nu=nu_out,
        pi_rg=pi_rg_out,
        pi_ai=pi_ai_out,
        pi_sol=pi_sol_out,
        item_params=item_out,
        regime_params=reg_p,
        posteriors=post_out,
        lrt_rg=lrt_rg_np,
        p_rg=p_rg,
        lrt_ai=lrt_ai_np,
        p_ai=p_ai,
        lrt_joint=lrt_joint_np,
        p_joint=p_joint,
        pi_rg_ci_low=pi_rg_lo,
        pi_rg_ci_high=pi_rg_hi,
        pi_ai_ci_low=pi_ai_lo,
        pi_ai_ci_high=pi_ai_hi,
        log_likelihood=total_ll,
        n_iter=n_iter,
        converged=converged,
        device_used=device_tag,
        fit_seconds=fit_seconds,
    )


def fit_dtmix_irt(
    Y: np.ndarray,
    log_rt: np.ndarray,
    entropy: np.ndarray,
    guessing: Union[float, np.ndarray] = 0.25,
    max_iter: int = 45,
    tol: float = 1e-5,
    device: str = "auto",
    compute_ci: bool = True,
    ci_grid_size: int = 100,
    fixed_item_params: Optional[ItemParameters] = None,
    fixed_regime_params: Optional[RegimeParameters] = None,
    prior_var_theta: float = 4.0,
    prior_prec_tau: float = 0.25,
    prior_prec_nu: float = 0.25,
    use_entropy: bool = True,
) -> DTMixIRTResult:
    """Fit the trivariate Dual-Tail Process-Data Mixture IRT (``DT-MixIRT``) model.

    Jointly estimates examinee latent ability (``theta``), working speed (``tau``),
    editing engagement (``nu``), left-tail rapid-guessing rate (``pi_rg``), and
    right-tail AI-assisted copying rate (``pi_ai``), alongside item parameters,
    boundary-corrected likelihood-ratio tests, 95% profile-likelihood confidence
    intervals, and item-level forensic regime posteriors.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix of shape ``(N, J)`` (``NaN`` allowed for missing cells).
    log_rt : np.ndarray
        Log response time matrix of shape ``(N, J)``.
    entropy : np.ndarray
        Keystroke/action-stream edit entropy matrix of shape ``(N, J)``.
    guessing : float or np.ndarray, default=0.25
        Known physical chance asymptote(s) ``c_j`` (e.g., ``0.25`` for four-choice items).
    max_iter : int, default=45
        Maximum number of alternating EM-Newton iterations.
    tol : float, default=1e-5
        Relative log-likelihood convergence tolerance.
    device : str, default="auto"
        Compute backend: ``"auto"`` (prefers CUDA GPU if available), ``"cuda:0"``,
        ``"cpu-torch"``, or ``"numpy"``.
    compute_ci : bool, default=True
        Whether to compute 95% profile-likelihood confidence intervals for
        ``pi_rg`` and ``pi_ai``.
    ci_grid_size : int, default=100
        Number of grid points on ``[0, 0.99]`` for profile-likelihood CI inversion.
    fixed_item_params : ItemParameters, optional
        Pre-calibrated item bank parameters for fixed-item forensic scoring.
    fixed_regime_params : RegimeParameters, optional
        Pre-calibrated global regime parameters for fixed-regime forensic scoring.
    prior_var_theta : float, default=4.0
        Gaussian prior variance on ``theta_i`` during damped Newton updates.
    prior_prec_tau : float, default=0.25
        Gaussian prior precision on ``tau_i``.
    prior_prec_nu : float, default=0.25
        Gaussian prior precision on ``nu_i``.
    use_entropy : bool, default=True
        If False, ablates the edit-entropy channel and fits a pure bivariate
        ``(Y, log_rt)`` three-regime mixture IRT model.

    Returns
    -------
    DTMixIRTResult
        Complete fitted model with de-biased abilities, dual-tail mixture weights,
        boundary-corrected LRTs, 95% profile CIs, and item-level forensic posteriors.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be strictly positive.")
    if ci_grid_size < 10:
        raise ValueError("ci_grid_size must be >= 10.")

    Y_c, L_c, H_c, mask, c_arr = _validate_inputs(Y, log_rt, entropy, guessing)
    dev_tag = _resolve_device(device)

    if dev_tag == "cpu-numpy" or not use_entropy:
        return _fit_numpy_backend(
            Y_c,
            L_c,
            H_c,
            mask,
            c_arr,
            max_iter=max_iter,
            tol=tol,
            fixed_item_params=fixed_item_params,
            fixed_regime_params=fixed_regime_params,
            compute_ci=compute_ci,
            ci_grid_size=ci_grid_size,
            prior_var_theta=prior_var_theta,
            prior_prec_tau=prior_prec_tau,
            prior_prec_nu=prior_prec_nu,
            use_entropy=use_entropy,
        )

    return _fit_torch_backend(
        Y_c,
        L_c,
        H_c,
        mask,
        c_arr,
        device_tag=dev_tag,
        max_iter=max_iter,
        tol=tol,
        fixed_item_params=fixed_item_params,
        fixed_regime_params=fixed_regime_params,
        compute_ci=compute_ci,
        ci_grid_size=ci_grid_size,
        prior_var_theta=prior_var_theta,
        prior_prec_tau=prior_prec_tau,
        prior_prec_nu=prior_prec_nu,
    )

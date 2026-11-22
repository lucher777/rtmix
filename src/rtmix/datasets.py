"""Data generation and bundled benchmark loader for ``rtmix``."""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict, Optional

import numpy as np

from .dtmix import _sigmoid_np
from .types import ItemParameters, RegimeParameters


def simulate_dual_tail_exam(
    n_examinees: int = 600,
    n_items: int = 60,
    prop_slow_careful: float = 0.20,
    prop_rapid_guesser: float = 0.15,
    prop_ai_copier: float = 0.15,
    prop_dual_tail: float = 0.05,
    pi_rg_range: tuple[float, float] = (0.20, 0.55),
    pi_ai_range: tuple[float, float] = (0.30, 0.75),
    delta_t_ai: float = 0.60,
    slow_careful_rt_sd: float = 0.50,
    guessing: float = 0.25,
    theta_pool: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate a trivariate digital-exam dataset with five examinee archetypes.

    Generates binary accuracy (``Y``), log response time (``log_rt``), and
    keystroke/action edit entropy (``entropy``) across:
    - Group 0: Typical clean solution examinees
    - Group 1: Slow careful examinees (strongly negative speed ``tau``, high edit
      entropy, strictly IRT-monotone solution accuracy; zero RG and zero AI)
    - Group 2: Left-tail rapid guessers (``pi_rg > 0``)
    - Group 3: Right-tail AI-assisted copiers (``pi_ai > 0``, elevated prompting
      pause latency + collapsed edit entropy + high hard-tail accuracy)
    - Group 4: Dual-tail examinees exhibiting both RG and AI copying on different items

    Parameters
    ----------
    n_examinees : int, default=600
        Total number of examinees ``N``.
    n_items : int, default=60
        Total number of test items ``J``.
    prop_slow_careful : float, default=0.20
        Proportion of slow careful genuine examinees.
    prop_rapid_guesser : float, default=0.15
        Proportion of left-tail rapid-guessing examinees.
    prop_ai_copier : float, default=0.15
        Proportion of right-tail AI-assisted copying examinees.
    prop_dual_tail : float, default=0.05
        Proportion of dual-tail examinees exhibiting both regimes.
    pi_rg_range : tuple of float, default=(0.20, 0.55)
        Uniform range for examinee rapid-guessing proportions among RG groups.
    pi_ai_range : tuple of float, default=(0.30, 0.75)
        Uniform range for examinee AI-assisted proportions among AI groups.
    delta_t_ai : float, default=0.60
        Mean prompting/copy-paste log-RT shift above solution baseline under Regime 2.
    slow_careful_rt_sd : float, default=0.50
        Within-person log-RT standard deviation for slow careful examinees,
        capturing realistic cognitive hesitation across items.
    guessing : float, default=0.25
        Known physical chance asymptote ``c_j``.
    theta_pool : np.ndarray, optional
        Empirical latent ability pool (e.g., standardized PISA plausible values)
        sampled with replacement when provided.
    seed : int, default=42
        Random number generator seed.

    Returns
    -------
    dict
        Dictionary containing observed matrices ``Y``, ``log_rt``, ``entropy``,
        ground-truth latent arrays, ``group_label``, ``regime_true``,
        ``item_params``, and ``regime_params``.
    """
    total_prop = prop_slow_careful + prop_rapid_guesser + prop_ai_copier + prop_dual_tail
    if total_prop > 1.0 + 1e-12:
        raise ValueError("Sum of non-typical group proportions must not exceed 1.0.")

    rng = np.random.default_rng(seed)

    # Item bank parameters spanning realistic easy-to-hard range
    a = rng.lognormal(mean=0.15, sigma=0.25, size=n_items).clip(0.55, 2.40)
    b = np.linspace(-2.2, 2.4, n_items) + rng.normal(0.0, 0.15, size=n_items)
    c = np.full(n_items, float(guessing), dtype=np.float64)
    # Harder items tend to require slightly more time and higher drafting/action entropy
    beta = 3.40 + 0.22 * b + rng.normal(0.0, 0.20, size=n_items)
    sigma_t = rng.uniform(0.32, 0.48, size=n_items)
    gamma = 2.60 + 0.18 * b + rng.normal(0.0, 0.18, size=n_items)
    sigma_h = rng.uniform(0.35, 0.50, size=n_items)

    item_params = ItemParameters(
        a=a, b=b, c=c, beta=beta, sigma_t=sigma_t, gamma=gamma, sigma_h=sigma_h
    )
    regime_params = RegimeParameters(
        mu_t_rg=1.25,
        sigma_t_rg=0.35,
        mu_h_rg=0.25,
        sigma_h_rg=0.30,
        p_ai=0.92,
        delta_t_ai=float(delta_t_ai),
        sigma_t_ai=0.40,
        mu_h_ai=0.35,
        sigma_h_ai=0.32,
    )

    # Examinee true latent traits
    if theta_pool is not None:
        pool = np.asarray(theta_pool, dtype=np.float64).ravel()
        pool = (pool - pool.mean()) / max(float(pool.std()), 1e-6)
        idx = rng.integers(0, pool.size, size=n_examinees)
        theta_true = pool[idx] + rng.normal(0.0, 0.05, size=n_examinees)
    else:
        theta_true = rng.normal(0.0, 1.0, size=n_examinees)
    theta_true = (theta_true - theta_true.mean()) / max(float(theta_true.std()), 1e-6)

    tau_true = rng.normal(0.0, 0.35, size=n_examinees)
    nu_true = rng.normal(0.0, 0.30, size=n_examinees)

    # Partition examinees into 5 archetypes
    n_sc = int(round(n_examinees * prop_slow_careful))
    n_rg = int(round(n_examinees * prop_rapid_guesser))
    n_ai = int(round(n_examinees * prop_ai_copier))
    n_dt = int(round(n_examinees * prop_dual_tail))
    n_clean = max(0, n_examinees - (n_sc + n_rg + n_ai + n_dt))

    group_label = np.concatenate([
        np.zeros(n_clean, dtype=np.int64),
        np.ones(n_sc, dtype=np.int64),
        np.full(n_rg, 2, dtype=np.int64),
        np.full(n_ai, 3, dtype=np.int64),
        np.full(n_dt, 4, dtype=np.int64),
    ])
    rng.shuffle(group_label)

    pi_rg_true = np.zeros(n_examinees, dtype=np.float64)
    pi_ai_true = np.zeros(n_examinees, dtype=np.float64)

    # Slow careful examinees: strongly negative speed (tau << 0) mimicking AI pause RT,
    # slightly elevated drafting entropy (nu >= 0), genuine IRT accuracy
    sc_mask = group_label == 1
    tau_true[sc_mask] = rng.uniform(-1.35, -0.75, size=int(sc_mask.sum()))
    nu_true[sc_mask] = rng.uniform(0.15, 0.65, size=int(sc_mask.sum()))

    rg_mask = group_label == 2
    pi_rg_true[rg_mask] = rng.uniform(pi_rg_range[0], pi_rg_range[1], size=int(rg_mask.sum()))

    ai_mask = group_label == 3
    pi_ai_true[ai_mask] = rng.uniform(pi_ai_range[0], pi_ai_range[1], size=int(ai_mask.sum()))

    dt_mask = group_label == 4
    pi_rg_true[dt_mask] = rng.uniform(0.18, 0.35, size=int(dt_mask.sum()))
    pi_ai_true[dt_mask] = rng.uniform(0.18, 0.35, size=int(dt_mask.sum()))

    # Re-center speed and entropy engagement globally while preserving slow-careful contrast
    tau_true = tau_true - np.mean(tau_true)
    nu_true = nu_true - np.mean(nu_true)

    # Draw item-level latent regimes Z_ij in {0, 1, 2}
    u = rng.random(size=(n_examinees, n_items))
    regime_true = np.zeros((n_examinees, n_items), dtype=np.int8)
    regime_true[u < pi_rg_true[:, None]] = 1
    regime_true[(u >= pi_rg_true[:, None]) & (u < (pi_rg_true + pi_ai_true)[:, None])] = 2

    # Generate trivariate outcomes
    p0 = c[None, :] + (1.0 - c[None, :]) * _sigmoid_np(a[None, :] * (theta_true[:, None] - b[None, :]))
    Y0 = (rng.random(size=(n_examinees, n_items)) < p0).astype(np.float64)
    sigma_t_mat = np.broadcast_to(sigma_t[None, :], (n_examinees, n_items)).copy()
    sigma_t_mat[sc_mask] = np.maximum(sigma_t_mat[sc_mask], float(slow_careful_rt_sd))
    L0 = rng.normal(beta[None, :] - tau_true[:, None], sigma_t_mat)
    H0 = rng.normal(gamma[None, :] + nu_true[:, None], sigma_h[None, :])

    Y1 = (rng.random(size=(n_examinees, n_items)) < c[None, :]).astype(np.float64)
    L1 = rng.normal(regime_params.mu_t_rg, regime_params.sigma_t_rg, size=(n_examinees, n_items))
    H1 = rng.normal(regime_params.mu_h_rg, regime_params.sigma_h_rg, size=(n_examinees, n_items))

    Y2 = (rng.random(size=(n_examinees, n_items)) < regime_params.p_ai).astype(np.float64)
    L2 = rng.normal(
        beta[None, :] - tau_true[:, None] + regime_params.delta_t_ai,
        regime_params.sigma_t_ai,
    )
    H2 = rng.normal(regime_params.mu_h_ai, regime_params.sigma_h_ai, size=(n_examinees, n_items))

    Y = np.where(regime_true == 0, Y0, np.where(regime_true == 1, Y1, Y2))
    log_rt = np.where(regime_true == 0, L0, np.where(regime_true == 1, L1, L2))
    entropy = np.where(regime_true == 0, H0, np.where(regime_true == 1, H1, H2))

    return {
        "Y": Y,
        "log_rt": log_rt,
        "entropy": entropy,
        "theta_true": theta_true,
        "tau_true": tau_true,
        "nu_true": nu_true,
        "pi_rg_true": pi_rg_true,
        "pi_ai_true": pi_ai_true,
        "regime_true": regime_true,
        "group_label": group_label,
        "item_params": item_params,
        "regime_params": regime_params,
    }


def load_benchmark_dataset() -> Dict[str, np.ndarray]:
    """Load the bundled PISA-calibrated dual-tail process-data benchmark slice.

    The bundled archive contains 480 examinees across 50 items whose latent
    proficiency distribution is calibrated to empirical standardized student
    achievement quantiles, paired with synthetic trivariate process streams
    (accuracy, log response time, and edit entropy) representing typical clean,
    slow careful, rapid-guessing, AI-assisted copying, and dual-tail regimes.

    Returns
    -------
    dict
        Dictionary of NumPy arrays including ``Y``, ``log_rt``, ``entropy``,
        ``theta_true``, ``tau_true``, ``nu_true``, ``pi_rg_true``, ``pi_ai_true``,
        ``group_label``, and ``regime_true``.
    """
    data_ref = resources.files("rtmix.data").joinpath("pisa_calibrated_benchmark.npz")
    with resources.as_file(data_ref) as path:
        with np.load(path, allow_pickle=False) as arc:
            return {k: arc[k].copy() for k in arc.files}

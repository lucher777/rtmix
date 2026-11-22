"""Typed result containers for the ``rtmix`` package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RegimeParameters:
    """Global latent parameters for the left-tail and right-tail mixture regimes.

    Attributes
    ----------
    mu_t_rg : float
        Mean log response time under Regime 1 (rapid guessing).
    sigma_t_rg : float
        Standard deviation of log response time under Regime 1.
    mu_h_rg : float
        Mean edit/action entropy under Regime 1.
    sigma_h_rg : float
        Standard deviation of edit/action entropy under Regime 1.
    p_ai : float
        Item-level elevated accuracy floor under Regime 2 (AI-assisted copying).
    delta_t_ai : float
        Positive prompting/copy-paste log-RT shift above solution baseline under Regime 2.
    sigma_t_ai : float
        Standard deviation of log response time under Regime 2.
    mu_h_ai : float
        Collapsed mean edit/action entropy under Regime 2.
    sigma_h_ai : float
        Standard deviation of edit/action entropy under Regime 2.
    """

    mu_t_rg: float
    sigma_t_rg: float
    mu_h_rg: float
    sigma_h_rg: float
    p_ai: float
    delta_t_ai: float
    sigma_t_ai: float
    mu_h_ai: float
    sigma_h_ai: float


@dataclass
class ItemParameters:
    """Calibrated item-level parameters for the three-regime DT-MixIRT model.

    Attributes
    ----------
    a : np.ndarray
        Item discrimination parameters, shape ``(J,)``.
    b : np.ndarray
        Item difficulty parameters, shape ``(J,)``.
    c : np.ndarray
        Known or fixed lower chance asymptotes, shape ``(J,)``.
    beta : np.ndarray
        Item time intensity parameters (log-RT scale), shape ``(J,)``.
    sigma_t : np.ndarray
        Item log-RT residual standard deviations under solution behavior, shape ``(J,)``.
    gamma : np.ndarray
        Item baseline action/edit entropy parameters, shape ``(J,)``.
    sigma_h : np.ndarray
        Item entropy residual standard deviations under solution behavior, shape ``(J,)``.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    beta: np.ndarray
    sigma_t: np.ndarray
    gamma: np.ndarray
    sigma_h: np.ndarray


@dataclass
class DTMixIRTResult:
    """Full output of Dual-Tail Process-Data Mixture IRT (``DT-MixIRT``) estimation.

    Attributes
    ----------
    theta : np.ndarray
        De-biased latent ability estimates, shape ``(N,)``.
    tau : np.ndarray
        Latent working speed estimates (higher = faster, negative = slow careful), shape ``(N,)``.
    nu : np.ndarray
        Latent editing/revision engagement estimates, shape ``(N,)``.
    pi_rg : np.ndarray
        Examinee-level left-tail rapid-guessing mixture weights, shape ``(N,)``.
    pi_ai : np.ndarray
        Examinee-level right-tail AI-assisted copying mixture weights, shape ``(N,)``.
    pi_sol : np.ndarray
        Examinee-level genuine solution-behavior weights ``1 - pi_rg - pi_ai``, shape ``(N,)``.
    item_params : ItemParameters
        Calibrated or fixed item parameters.
    regime_params : RegimeParameters
        Calibrated or fixed global parameters for Regimes 1 and 2.
    posteriors : np.ndarray
        Item-level regime posterior probabilities ``P(Z_ij = k | Y_ij, L_ij, H_ij)``,
        shape ``(N, J, 3)`` with channels ``0=Solution, 1=RG, 2=AI``.
    lrt_rg : np.ndarray
        Boundary-corrected likelihood-ratio statistics for ``H0: pi_rg = 0``, shape ``(N,)``.
    p_rg : np.ndarray
        Boundary-corrected 50:50 chi-square p-values for ``H0: pi_rg = 0``, shape ``(N,)``.
    lrt_ai : np.ndarray
        Boundary-corrected likelihood-ratio statistics for ``H0: pi_ai = 0``, shape ``(N,)``.
    p_ai : np.ndarray
        Boundary-corrected 50:50 chi-square p-values for ``H0: pi_ai = 0``, shape ``(N,)``.
    lrt_joint : np.ndarray
        Omnibus likelihood-ratio statistics for ``H0: pi_rg = pi_ai = 0``, shape ``(N,)``.
    p_joint : np.ndarray
        Conservative mixture chi-square p-values for ``H0: pi_rg = pi_ai = 0``, shape ``(N,)``.
    pi_rg_ci_low : Optional[np.ndarray]
        Lower 95% profile-likelihood confidence bounds for ``pi_rg``, shape ``(N,)``.
    pi_rg_ci_high : Optional[np.ndarray]
        Upper 95% profile-likelihood confidence bounds for ``pi_rg``, shape ``(N,)``.
    pi_ai_ci_low : Optional[np.ndarray]
        Lower 95% profile-likelihood confidence bounds for ``pi_ai``, shape ``(N,)``.
    pi_ai_ci_high : Optional[np.ndarray]
        Upper 95% profile-likelihood confidence bounds for ``pi_ai``, shape ``(N,)``.
    log_likelihood : float
        Total observed marginal log-likelihood at convergence.
    n_iter : int
        Number of EM-Newton iterations executed.
    converged : bool
        Whether relative log-likelihood change fell below tolerance.
    device_used : str
        Compute backend identifier (e.g., ``"cuda:0"``, ``"cpu-torch"``, or ``"cpu-numpy"``).
    fit_seconds : float
        Total wall-clock runtime in seconds.
    """

    theta: np.ndarray
    tau: np.ndarray
    nu: np.ndarray
    pi_rg: np.ndarray
    pi_ai: np.ndarray
    pi_sol: np.ndarray
    item_params: ItemParameters
    regime_params: RegimeParameters
    posteriors: np.ndarray
    lrt_rg: np.ndarray
    p_rg: np.ndarray
    lrt_ai: np.ndarray
    p_ai: np.ndarray
    lrt_joint: np.ndarray
    p_joint: np.ndarray
    pi_rg_ci_low: Optional[np.ndarray]
    pi_rg_ci_high: Optional[np.ndarray]
    pi_ai_ci_low: Optional[np.ndarray]
    pi_ai_ci_high: Optional[np.ndarray]
    log_likelihood: float
    n_iter: int
    converged: bool
    device_used: str
    fit_seconds: float


@dataclass
class BaselineResult:
    """Container for baseline IRT, speed-accuracy, and threshold models.

    Attributes
    ----------
    method : str
        Name of the baseline estimator.
    theta : np.ndarray
        Estimated examinee ability scores, shape ``(N,)``.
    rg_score : np.ndarray
        Examinee-level rapid-guessing anomaly score, shape ``(N,)``.
    ai_score : np.ndarray
        Examinee-level right-tail / AI-cheating anomaly score, shape ``(N,)``.
    fit_seconds : float
        Wall-clock runtime in seconds.
    extras : Dict[str, np.ndarray]
        Additional method-specific diagnostic arrays.
    """

    method: str
    theta: np.ndarray
    rg_score: np.ndarray
    ai_score: np.ndarray
    fit_seconds: float
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ForensicAuditResult:
    """Examinee-level and item-level forensic flagging summary.

    Attributes
    ----------
    flags : np.ndarray
        Integer regime flags per examinee: ``0=Clean``, ``1=RapidGuessing``,
        ``2=AIAssisted``, ``3=DualTail``.
    flag_labels : np.ndarray
        Human-readable string labels corresponding to ``flags``.
    flagged_rg : np.ndarray
        Boolean mask of examinees flagged for significant rapid guessing.
    flagged_ai : np.ndarray
        Boolean mask of examinees flagged for significant AI-assisted copying.
    hard_tail_obs_acc : np.ndarray
        Observed accuracy on the hardest 25% of items per examinee, shape ``(N,)``.
    hard_tail_exp_acc : np.ndarray
        Expected genuine-solution accuracy on the hardest 25% of items, shape ``(N,)``.
    hard_tail_excess : np.ndarray
        Difference ``hard_tail_obs_acc - hard_tail_exp_acc``, shape ``(N,)``.
    easy_tail_obs_acc : np.ndarray
        Observed accuracy on the easiest 25% of items per examinee, shape ``(N,)``.
    easy_tail_exp_acc : np.ndarray
        Expected genuine-solution accuracy on the easiest 25% of items, shape ``(N,)``.
    mean_entropy_gap : np.ndarray
        Mean difference between expected baseline entropy and observed entropy, shape ``(N,)``.
    item_ai_rate : np.ndarray
        Mean posterior probability of Regime 2 per item, shape ``(J,)``.
    item_rg_rate : np.ndarray
        Mean posterior probability of Regime 1 per item, shape ``(J,)``.
    """

    flags: np.ndarray
    flag_labels: np.ndarray
    flagged_rg: np.ndarray
    flagged_ai: np.ndarray
    hard_tail_obs_acc: np.ndarray
    hard_tail_exp_acc: np.ndarray
    hard_tail_excess: np.ndarray
    easy_tail_obs_acc: np.ndarray
    easy_tail_exp_acc: np.ndarray
    mean_entropy_gap: np.ndarray
    item_ai_rate: np.ndarray
    item_rg_rate: np.ndarray


@dataclass
class EquivalenceReport:
    """Numerical parity and speedup verification against scalar SciPy reference.

    Attributes
    ----------
    max_abs_diff_estep : float
        Maximum absolute discrepancy in E-step posterior responsibilities ``r_ijk``.
    max_abs_diff_loglik : float
        Maximum absolute discrepancy in per-examinee marginal log-likelihoods.
    max_abs_diff_theta_mstep : float
        Maximum absolute discrepancy in damped Newton ``theta`` M-step vs. scalar SciPy L-BFGS-B.
    max_abs_diff_tau_nu : float
        Maximum absolute discrepancy in closed-form ``tau`` and ``nu`` M-step updates.
    max_abs_diff_gpu_cpu : Optional[float]
        Maximum absolute discrepancy between PyTorch CUDA and NumPy CPU posteriors (if CUDA available).
    vectorized_seconds : float
        Wall-clock runtime of the vectorized/GPU engine on the benchmark task.
    scipy_seconds : float
        Wall-clock runtime of the scalar per-examinee SciPy L-BFGS-B reference.
    speedup : float
        Ratio ``scipy_seconds / vectorized_seconds``.
    device_used : str
        Device used for the fast engine.
    all_passed : bool
        True if all parity checks are strictly below ``tol`` (default ``1e-8``).
    """

    max_abs_diff_estep: float
    max_abs_diff_loglik: float
    max_abs_diff_theta_mstep: float
    max_abs_diff_tau_nu: float
    max_abs_diff_gpu_cpu: Optional[float]
    vectorized_seconds: float
    scipy_seconds: float
    speedup: float
    device_used: str
    all_passed: bool

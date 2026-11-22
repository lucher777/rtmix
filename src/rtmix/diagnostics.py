"""Forensic diagnostics, person-fit anomaly flagging, and recovery metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from .dtmix import _EPS, _sigmoid_np, _validate_inputs
from .types import DTMixIRTResult, ForensicAuditResult

_FLAG_NAMES = {
    0: "Clean",
    1: "RapidGuessing",
    2: "AIAssisted",
    3: "DualTail",
}


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact rank-based Area Under the ROC Curve (AUROC).

    Handles ties via mid-ranks without external ML library dependencies.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground-truth indicator array (``0`` or ``1``).
    scores : np.ndarray
        Continuous anomaly or classification scores.

    Returns
    -------
    float
        Exact AUROC in ``[0, 1]``.
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("labels and scores must have identical length.")
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative classes are required to compute AUROC.")

    # Mid-rank Mann-Whitney U formula
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty_like(s_sorted, dtype=np.float64)
    n = s_sorted.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    inv_ranks = np.empty_like(ranks)
    inv_ranks[order] = ranks
    u_stat = float(np.sum(inv_ranks[pos]) - n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def evaluate_ability_recovery(
    theta_true: np.ndarray,
    theta_est: np.ndarray,
    subset_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate latent ability recovery, bias, and rank-order fidelity.

    Aligns ``theta_est`` only by checking sign if needed? No: preserves exact
    metric scale and reports both raw RMSE/MAE/bias and scale-invariant rank
    correlations (Pearson r, Spearman rho, Kendall tau).

    Parameters
    ----------
    theta_true : np.ndarray
        Ground-truth latent abilities, shape ``(N,)``.
    theta_est : np.ndarray
        Estimated latent abilities, shape ``(N,)``.
    subset_mask : np.ndarray, optional
        Boolean mask restricting evaluation to a subgroup (e.g., AI copiers).

    Returns
    -------
    dict
        Dictionary with keys ``rmse``, ``mae``, ``bias``, ``pearson_r``,
        ``spearman_rho``, and ``kendall_tau``.
    """
    t_true = np.asarray(theta_true, dtype=np.float64).ravel()
    t_est = np.asarray(theta_est, dtype=np.float64).ravel()
    if t_true.shape != t_est.shape:
        raise ValueError("theta_true and theta_est must have identical shape.")
    if subset_mask is not None:
        m = np.asarray(subset_mask, dtype=bool).ravel()
        t_true = t_true[m]
        t_est = t_est[m]
    if t_true.size < 2:
        raise ValueError("At least 2 examinees are required to compute recovery metrics.")

    diff = t_est - t_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r_val = float(pearsonr(t_true, t_est)[0])
    rho_val = float(spearmanr(t_true, t_est)[0])
    tau_val = float(kendalltau(t_true, t_est)[0])
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson_r": r_val,
        "spearman_rho": rho_val,
        "kendall_tau": tau_val,
    }


def audit_dual_tail_forensics(
    Y: np.ndarray,
    log_rt: np.ndarray,
    entropy: np.ndarray,
    result: DTMixIRTResult,
    alpha: float = 0.01,
    min_pi: float = 0.08,
    tail_quantile: float = 0.25,
) -> ForensicAuditResult:
    """Produce examinee-level and item-level dual-tail forensic anomaly audits.

    Flags examinees using both boundary-corrected statistical significance
    (``p < alpha``) and practical prevalence (``pi >= min_pi``), while computing
    interpretable hard-tail accuracy surges, easy-tail accuracy collapses, and
    edit-entropy collapse gaps.

    Parameters
    ----------
    Y : np.ndarray
        Binary response matrix, shape ``(N, J)``.
    log_rt : np.ndarray
        Log response time matrix, shape ``(N, J)``.
    entropy : np.ndarray
        Edit/action entropy matrix, shape ``(N, J)``.
    result : DTMixIRTResult
        Fitted ``DT-MixIRT`` model result.
    alpha : float, default=0.01
        Significance threshold applied to boundary-corrected LRT p-values.
    min_pi : float, default=0.08
        Minimum mixture proportion required for practical flagging.
    tail_quantile : float, default=0.25
        Fraction of hardest and easiest items used for tail contrast summaries.

    Returns
    -------
    ForensicAuditResult
        Structured examinee and item forensic audit report.
    """
    Y_c, L_c, H_c, mask, _ = _validate_inputs(Y, log_rt, entropy, result.item_params.c)
    ip = result.item_params

    flagged_rg = (result.p_rg < alpha) & (result.pi_rg >= min_pi)
    flagged_ai = (result.p_ai < alpha) & (result.pi_ai >= min_pi)

    flags = np.zeros(Y_c.shape[0], dtype=np.int64)
    flags[flagged_rg & (~flagged_ai)] = 1
    flags[(~flagged_rg) & flagged_ai] = 2
    flags[flagged_rg & flagged_ai] = 3
    flag_labels = np.array([_FLAG_NAMES[int(f)] for f in flags], dtype=object)

    # Expected genuine-solution probabilities under calibrated Regime 0
    z = ip.a[None, :] * (result.theta[:, None] - ip.b[None, :])
    p0 = np.clip(ip.c[None, :] + (1.0 - ip.c[None, :]) * _sigmoid_np(z), _EPS, 1.0 - _EPS)

    hard_cut = float(np.quantile(ip.b, 1.0 - tail_quantile))
    easy_cut = float(np.quantile(ip.b, tail_quantile))
    hard_items = ip.b >= hard_cut
    easy_items = ip.b <= easy_cut

    mask_f = mask.astype(np.float64)
    h_den = np.maximum(mask_f[:, hard_items].sum(axis=1), 1.0)
    e_den = np.maximum(mask_f[:, easy_items].sum(axis=1), 1.0)

    hard_obs = np.sum(mask_f[:, hard_items] * Y_c[:, hard_items], axis=1) / h_den
    hard_exp = np.sum(mask_f[:, hard_items] * p0[:, hard_items], axis=1) / h_den
    hard_excess = hard_obs - hard_exp

    easy_obs = np.sum(mask_f[:, easy_items] * Y_c[:, easy_items], axis=1) / e_den
    easy_exp = np.sum(mask_f[:, easy_items] * p0[:, easy_items], axis=1) / e_den

    exp_h = ip.gamma[None, :] + result.nu[:, None]
    mean_entropy_gap = np.sum(mask_f * (exp_h - H_c), axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)

    post = np.where(np.isfinite(result.posteriors), result.posteriors, 0.0)
    obs_j = np.maximum(mask_f.sum(axis=0), 1.0)
    item_rg_rate = np.sum(post[:, :, 1] * mask_f, axis=0) / obs_j
    item_ai_rate = np.sum(post[:, :, 2] * mask_f, axis=0) / obs_j

    return ForensicAuditResult(
        flags=flags,
        flag_labels=flag_labels,
        flagged_rg=flagged_rg,
        flagged_ai=flagged_ai,
        hard_tail_obs_acc=hard_obs,
        hard_tail_exp_acc=hard_exp,
        hard_tail_excess=hard_excess,
        easy_tail_obs_acc=easy_obs,
        easy_tail_exp_acc=easy_exp,
        mean_entropy_gap=mean_entropy_gap,
        item_ai_rate=item_ai_rate,
        item_rg_rate=item_rg_rate,
    )

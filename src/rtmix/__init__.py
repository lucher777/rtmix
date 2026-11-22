"""``rtmix``: Dual-Tail Process-Data Mixture IRT for Digital Exam Forensics.

Public API exports trivariate three-regime estimation (``fit_dtmix_irt``),
item-level regime posteriors (``compute_item_regime_posteriors``), classical
baselines, dual-tail forensic diagnostics, benchmark data loaders, and
cross-implementation numerical parity verification.
"""

from __future__ import annotations

from .baselines import (
    fit_bivariate_mixture_ablation,
    fit_classical_irt,
    fit_vdl_speed_accuracy,
    fit_wise_nt_baseline,
)
from .datasets import load_benchmark_dataset, simulate_dual_tail_exam
from .diagnostics import audit_dual_tail_forensics, compute_auroc, evaluate_ability_recovery
from .dtmix import (
    compute_item_regime_posteriors,
    compute_regime_logliks_numpy,
    fit_dtmix_irt,
)
from .equiv import verify_numerical_equivalence
from .types import (
    BaselineResult,
    DTMixIRTResult,
    EquivalenceReport,
    ForensicAuditResult,
    ItemParameters,
    RegimeParameters,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaselineResult",
    "DTMixIRTResult",
    "EquivalenceReport",
    "ForensicAuditResult",
    "ItemParameters",
    "RegimeParameters",
    "audit_dual_tail_forensics",
    "compute_auroc",
    "compute_item_regime_posteriors",
    "compute_regime_logliks_numpy",
    "evaluate_ability_recovery",
    "fit_bivariate_mixture_ablation",
    "fit_classical_irt",
    "fit_dtmix_irt",
    "fit_vdl_speed_accuracy",
    "fit_wise_nt_baseline",
    "load_benchmark_dataset",
    "simulate_dual_tail_exam",
    "verify_numerical_equivalence",
]

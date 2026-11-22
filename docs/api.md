# API Reference

## Core Estimation (`rtmix.dtmix`)

- `fit_dtmix_irt(Y, log_rt, entropy, guessing=0.25, max_iter=45, tol=1e-5, device="auto", compute_ci=True, ci_grid_size=100, fixed_item_params=None, fixed_regime_params=None, use_entropy=True) -> DTMixIRTResult`
- `compute_item_regime_posteriors(Y, log_rt, entropy, result) -> np.ndarray`
- `compute_regime_logliks_numpy(Y, log_rt, entropy, theta, tau, nu, item_params, regime_params, use_entropy=True) -> np.ndarray`

## Forensic Diagnostics (`rtmix.diagnostics`)

- `audit_dual_tail_forensics(Y, log_rt, entropy, result, alpha=0.01, min_pi=0.08, tail_quantile=0.25) -> ForensicAuditResult`
- `evaluate_ability_recovery(theta_true, theta_est, subset_mask=None) -> dict`
- `compute_auroc(labels, scores) -> float`

## Classical Baselines (`rtmix.baselines`)

- `fit_classical_irt(Y, guessing=0.25, max_iter=35) -> BaselineResult`
- `fit_vdl_speed_accuracy(Y, log_rt, guessing=0.25, max_iter=35) -> BaselineResult`
- `fit_wise_nt_baseline(Y, log_rt, threshold_ratio=0.10, guessing=0.25) -> BaselineResult`
- `fit_bivariate_mixture_ablation(Y, log_rt, guessing=0.25, max_iter=35) -> BaselineResult`

## Datasets & Verification (`rtmix.datasets`, `rtmix.equiv`)

- `simulate_dual_tail_exam(...) -> dict`
- `load_benchmark_dataset() -> dict`
- `verify_numerical_equivalence(n_examinees=200, n_items=50, seed=42, tol=1e-8) -> EquivalenceReport`

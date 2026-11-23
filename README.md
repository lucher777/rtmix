# rtmix

**Separating rapid guessing, careful slowness and AI-assisted responding with process data**

`rtmix` (v0.1.0) is a Python package for computer-based tests that record process data. It jointly models item accuracy ($Y_{ij}$), log response time ($\log T_{ij}$) and an action-stream measure such as keystroke edit entropy ($H_{ij}$), and fits a three-regime mixture IRT model:

1. **Solution behavior** (Regime 0, weight $1 - \pi_i^{\mathrm{RG}} - \pi_i^{\mathrm{AI}}$): 3PL accuracy and lognormal response times, including slow, careful examinees;
2. **Rapid guessing** (Regime 1, weight $\pi_i^{\mathrm{RG}}$): short response times, low action entropy and chance-level accuracy $c_j$;
3. **AI-assisted responding** (Regime 2, weight $\pi_i^{\mathrm{AI}}$): a pause shift ($\Delta_T^{\mathrm{AI}} > 0$), collapsed edit entropy and raised accuracy $p_{\mathrm{AI}}$ on hard items.

## Why `rtmix`?

Response-time screens target rapid guessing in the fast tail (Wise & Kong, 2005; Wise & DeMars, 2006). An examinee who consults an external AI tool instead pauses and then answers hard items correctly, and speed-accuracy models (van der Linden, 2007; Wang & Xu, 2015) cannot tell this pattern from honest, careful work because both give long response times and good accuracy. `rtmix` adds the action-stream measure to separate the two:

- **Three-regime mixture.** Accuracy, response time and edit entropy enter one likelihood, so AI-assisted responding and slow careful work are distinguished by entropy rather than by speed alone.
- **Ability estimates weighted by solution behavior.** Newton ability updates are weighted by the solution-behavior posteriors $r_{ij0}$, which reduces the downward bias from rapid guessing and the upward bias from AI-assisted responses.
- **Person-level inference.** Boundary-corrected likelihood-ratio tests of a zero regime proportion (Self & Liang, 1987), 95% profile-likelihood confidence intervals (`pi_rg_ci_low/high`, `pi_ai_ci_low/high`) and item-level regime posteriors (`compute_item_regime_posteriors()`). Runs on NumPy (float64) or, optionally, PyTorch CUDA.
- **Baselines included.** A 3PL model, the NT10 rule, a speed-accuracy screen and a bivariate (accuracy, response time) mixture.

**Scope.** The bundled benchmark data are simulated from the model, so the reported separation is parameter recovery under the model, not field accuracy. A flag calls for human review, not a finding of misconduct. CI tests Python 3.10-3.12 on Linux.

## Installation

```bash
pip install "git+https://github.com/lucher777/rtmix.git@v0.1.0"
# Optional PyTorch CUDA acceleration
pip install "rtmix[gpu] @ git+https://github.com/lucher777/rtmix.git@v0.1.0"
# Development install (tests)
git clone https://github.com/lucher777/rtmix.git && cd rtmix && pip install -e ".[dev]"
```

## Quick Start

```python
from rtmix import (
    audit_dual_tail_forensics,
    fit_dtmix_irt,
    load_benchmark_dataset,
    verify_numerical_equivalence,
)

data = load_benchmark_dataset()
res = fit_dtmix_irt(
    data["Y"],
    data["log_rt"],
    data["entropy"],
    guessing=0.25,
    device="auto",
    compute_ci=True,
)
audit = audit_dual_tail_forensics(data["Y"], data["log_rt"], data["entropy"], res)
print(f"Device: {res.device_used} | Fit: {res.fit_seconds:.2f}s | Flagged AI: {audit.flagged_ai.sum()}")
assert verify_numerical_equivalence().all_passed
```

## Command-Line Interface

```bash
rtmix demo
rtmix audit
rtmix equiv
rtmix fit --input exam_process.npz --output dtmix_out.npz
```

## License

MIT License. Copyright (c) 2026 Kang Liu (ORCID: 0009-0004-3214-505X).

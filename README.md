# rtmix

**Dual-Tail Process-Data Mixture IRT (`DT-MixIRT`) for Digital Exam Forensics**

`rtmix` (v0.1.0) is an open-source, GPU-accelerated Python package that jointly models trivariate digital assessment process data—binary response accuracy ($Y_{ij}$), log response time ($\log T_{ij}$), and keystroke/action-stream edit entropy ($H_{ij}$)—to orthogonally disentangle:

1. **Genuine Solution Behavior** (Regime 0, weight $1 - \pi_i^{\mathrm{RG}} - \pi_i^{\mathrm{AI}}$), including **slow careful examinees** with strongly negative latent speed $\tau_i \ll 0$;
2. **Left-Tail Rapid Guessing** (Regime 1, weight $\pi_i^{\mathrm{RG}}$): compressed response times, minimal action entropy, and near-chance accuracy $c_j$;
3. **Right-Tail AI-Assisted Copying / Off-Platform Prompting** (Regime 2, weight $\pi_i^{\mathrm{AI}}$): elevated prompting/copy-paste latency shift ($\Delta_T^{\mathrm{AI}} > 0$), **collapsed edit/revision entropy**, and abnormally elevated accuracy $p_{\mathrm{AI}}$ even on difficult items.

## Why `rtmix`?

Conventional accuracy-only IRT and speed-accuracy models (e.g., van der Linden, 2007; Wise & DeMars, 2006) only address left-tail rapid guessing and fatally confound **slow careful examinees** with **AI-assisted copiers** because both occupy the slow right tail of response times. `rtmix` resolves this identifiability barrier via three core capabilities:

- **Trivariate Three-Regime Orthogonal Identifiability**: Combines $(Y_{ij}, \log T_{ij}, H_{ij})$ to separate AI-assisted copying from slow careful work without false accusations against honest slow examinees.
- **Dual-Tail Latent Ability De-Biasing**: Weights person ability Newton-Raphson updates strictly by genuine-solution posteriors $r_{ij0}$, simultaneously eliminating downward rapid-guessing bias and upward AI-copying inflation.
- **Boundary-Corrected Inference & GPU Acceleration**: Executes via PyTorch CUDA (`cuda:0`) or float64 NumPy, providing Self-Liang (1987) 50:50 $\chi^2$ boundary-corrected likelihood-ratio tests, 95% profile-likelihood confidence intervals (`pi_rg_ci_low/high`, `pi_ai_ci_low/high`), and item-level forensic posteriors (`compute_item_regime_posteriors()`).

## Installation

```bash
pip install rtmix
# Optional PyTorch CUDA acceleration
pip install "rtmix[gpu]"
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

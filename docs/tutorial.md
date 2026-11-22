# Tutorial: Auditing Dual-Tail Digital Exam Process Data

## 1. Load or Prepare Trivariate Matrices

`rtmix` accepts three aligned 2D arrays of shape `(N, J)`:
- `Y`: binary correctness (`0`/`1`, `NaN` permitted for missing responses)
- `log_rt`: natural logarithm of item response times (seconds)
- `entropy`: keystroke or action-stream edit entropy per item response

```python
from rtmix import load_benchmark_dataset, fit_dtmix_irt, audit_dual_tail_forensics

data = load_benchmark_dataset()
res = fit_dtmix_irt(
    data["Y"],
    data["log_rt"],
    data["entropy"],
    guessing=0.25,
    device="auto",
    compute_ci=True,
)
```

## 2. Inspect Examinee Flags, Confidence Intervals, and Item Posteriors

```python
audit = audit_dual_tail_forensics(data["Y"], data["log_rt"], data["entropy"], res, alpha=0.01, min_pi=0.08)
print("Flagged AI-assisted:", audit.flagged_ai.sum())
print("Flagged Rapid Guessing:", audit.flagged_rg.sum())
print("Posterior simplex max error:", abs(res.posteriors.sum(axis=-1) - 1.0).max())
```

## 3. Compare Against Classical Baselines

```python
from rtmix import fit_classical_irt, fit_vdl_speed_accuracy, fit_wise_nt_baseline, evaluate_ability_recovery

irt = fit_classical_irt(data["Y"])
vdl = fit_vdl_speed_accuracy(data["Y"], data["log_rt"])
nt10 = fit_wise_nt_baseline(data["Y"], data["log_rt"])

print("DT-MixIRT ability recovery:", evaluate_ability_recovery(data["theta_true"], res.theta))
print("Classical 3PL recovery:", evaluate_ability_recovery(data["theta_true"], irt.theta))
```

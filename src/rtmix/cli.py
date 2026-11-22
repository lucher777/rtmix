"""Command-line interface for ``rtmix`` (``demo``, ``fit``, ``audit``, ``equiv``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .baselines import fit_classical_irt, fit_vdl_speed_accuracy
from .datasets import load_benchmark_dataset
from .diagnostics import audit_dual_tail_forensics, compute_auroc, evaluate_ability_recovery
from .dtmix import fit_dtmix_irt
from .equiv import verify_numerical_equivalence


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run an end-to-end demonstration on the bundled benchmark dataset."""
    data = load_benchmark_dataset()
    Y = data["Y"]
    log_rt = data["log_rt"]
    entropy = data["entropy"]
    group = data["group_label"]
    theta_true = data["theta_true"]

    res = fit_dtmix_irt(
        Y,
        log_rt,
        entropy,
        guessing=0.25,
        device=args.device,
        compute_ci=not args.no_ci,
    )
    audit = audit_dual_tail_forensics(Y, log_rt, entropy, res)
    base_irt = fit_classical_irt(Y, guessing=0.25)
    base_vdl = fit_vdl_speed_accuracy(Y, log_rt, guessing=0.25)

    # AI vs Slow Careful discrimination
    sc_ai = (group == 1) | (group == 3)
    y_ai = (group[sc_ai] == 3).astype(int)
    auroc_dt = compute_auroc(y_ai, res.pi_ai[sc_ai])
    auroc_vdl = compute_auroc(y_ai, base_vdl.ai_score[sc_ai])

    rec_dt = evaluate_ability_recovery(theta_true, res.theta)
    rec_irt = evaluate_ability_recovery(theta_true, base_irt.theta)

    summary = {
        "device_used": res.device_used,
        "fit_seconds": round(res.fit_seconds, 4),
        "n_examinees": int(Y.shape[0]),
        "n_items": int(Y.shape[1]),
        "auroc_ai_vs_slow_careful_dtmix": round(auroc_dt, 4),
        "auroc_ai_vs_slow_careful_vdl": round(auroc_vdl, 4),
        "theta_rmse_dtmix": round(rec_dt["rmse"], 4),
        "theta_rmse_classical_irt": round(rec_irt["rmse"], 4),
        "flagged_rg": int(audit.flagged_rg.sum()),
        "flagged_ai": int(audit.flagged_ai.sum()),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    """Fit DT-MixIRT to an input ``.npz`` archive and save JSON/NPZ outputs."""
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input archive not found: {in_path}")
    with np.load(in_path, allow_pickle=False) as arc:
        Y = arc["Y"]
        log_rt = arc["log_rt"]
        entropy = arc["entropy"]

    res = fit_dtmix_irt(
        Y,
        log_rt,
        entropy,
        guessing=args.guessing,
        max_iter=args.max_iter,
        device=args.device,
        compute_ci=not args.no_ci,
    )
    summary = {
        "device_used": res.device_used,
        "fit_seconds": round(res.fit_seconds, 4),
        "n_iter": res.n_iter,
        "converged": res.converged,
        "log_likelihood": round(res.log_likelihood, 4),
        "mean_pi_rg": round(float(res.pi_rg.mean()), 4),
        "mean_pi_ai": round(float(res.pi_ai.mean()), 4),
        "p_ai_est": round(res.regime_params.p_ai, 4),
        "delta_t_ai_est": round(res.regime_params.delta_t_ai, 4),
    }
    if args.output:
        out_path = Path(args.output)
        np.savez_compressed(
            out_path,
            theta=res.theta,
            tau=res.tau,
            nu=res.nu,
            pi_rg=res.pi_rg,
            pi_ai=res.pi_ai,
            lrt_rg=res.lrt_rg,
            p_rg=res.p_rg,
            lrt_ai=res.lrt_ai,
            p_ai=res.p_ai,
            posteriors=res.posteriors,
        )
        summary["saved_to"] = str(out_path)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Run dual-tail forensic auditing on an input ``.npz`` archive or bundled benchmark."""
    if args.input:
        with np.load(Path(args.input), allow_pickle=False) as arc:
            Y, log_rt, entropy = arc["Y"], arc["log_rt"], arc["entropy"]
    else:
        data = load_benchmark_dataset()
        Y, log_rt, entropy = data["Y"], data["log_rt"], data["entropy"]

    res = fit_dtmix_irt(Y, log_rt, entropy, guessing=args.guessing, device=args.device, compute_ci=True)
    audit = audit_dual_tail_forensics(
        Y, log_rt, entropy, res, alpha=args.alpha, min_pi=args.min_pi
    )
    counts = {
        "Clean": int(np.sum(audit.flags == 0)),
        "RapidGuessing": int(np.sum(audit.flags == 1)),
        "AIAssisted": int(np.sum(audit.flags == 2)),
        "DualTail": int(np.sum(audit.flags == 3)),
    }
    report = {
        "device_used": res.device_used,
        "fit_seconds": round(res.fit_seconds, 4),
        "flag_counts": counts,
        "max_pi_ai": round(float(res.pi_ai.max()), 4),
        "max_pi_rg": round(float(res.pi_rg.max()), 4),
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_equiv(args: argparse.Namespace) -> int:
    """Verify strict numerical parity (< 1e-8) and speedup against SciPy L-BFGS-B."""
    rep = verify_numerical_equivalence(
        n_examinees=args.n_examinees,
        n_items=args.n_items,
        seed=args.seed,
        tol=args.tol,
    )
    payload = {
        "all_passed": rep.all_passed,
        "device_used": rep.device_used,
        "max_abs_diff_estep": rep.max_abs_diff_estep,
        "max_abs_diff_loglik": rep.max_abs_diff_loglik,
        "max_abs_diff_theta_mstep": rep.max_abs_diff_theta_mstep,
        "max_abs_diff_tau_nu": rep.max_abs_diff_tau_nu,
        "max_abs_diff_gpu_cpu": rep.max_abs_diff_gpu_cpu,
        "vectorized_seconds": round(rep.vectorized_seconds, 6),
        "scipy_seconds": round(rep.scipy_seconds, 6),
        "speedup": round(rep.speedup, 2),
    }
    print(json.dumps(payload, indent=2))
    return 0 if rep.all_passed else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``rtmix`` command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser with subcommands ``demo``, ``fit``, ``audit``,
        and ``equiv``.
    """
    parser = argparse.ArgumentParser(
        prog="rtmix",
        description="Dual-Tail Process-Data Mixture IRT (DT-MixIRT) CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run benchmark demonstration.")
    p_demo.add_argument("--device", default="auto", help="Compute backend (auto, cuda:0, cpu-torch, numpy).")
    p_demo.add_argument("--no-ci", action="store_true", help="Skip 95% profile-likelihood CIs.")
    p_demo.set_defaults(func=_cmd_demo)

    p_fit = sub.add_parser("fit", help="Fit DT-MixIRT to an .npz trivariate archive.")
    p_fit.add_argument("--input", required=True, help="Path to .npz file containing Y, log_rt, entropy.")
    p_fit.add_argument("--output", default=None, help="Optional .npz output file path.")
    p_fit.add_argument("--guessing", type=float, default=0.25, help="Chance asymptote.")
    p_fit.add_argument("--max-iter", type=int, default=45, help="Maximum EM-Newton iterations.")
    p_fit.add_argument("--device", default="auto", help="Compute backend.")
    p_fit.add_argument("--no-ci", action="store_true", help="Skip 95% profile CIs.")
    p_fit.set_defaults(func=_cmd_fit)

    p_audit = sub.add_parser("audit", help="Run dual-tail forensic examinee flagging.")
    p_audit.add_argument("--input", default=None, help="Optional .npz archive (defaults to bundled benchmark).")
    p_audit.add_argument("--guessing", type=float, default=0.25, help="Chance asymptote.")
    p_audit.add_argument("--alpha", type=float, default=0.01, help="LRT p-value threshold.")
    p_audit.add_argument("--min-pi", type=float, default=0.08, help="Minimum regime weight threshold.")
    p_audit.add_argument("--device", default="auto", help="Compute backend.")
    p_audit.set_defaults(func=_cmd_audit)

    p_equiv = sub.add_parser("equiv", help="Verify numerical parity against scalar SciPy reference.")
    p_equiv.add_argument("--n-examinees", type=int, default=200, help="Examinees in parity check.")
    p_equiv.add_argument("--n-items", type=int, default=50, help="Items in parity check.")
    p_equiv.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_equiv.add_argument("--tol", type=float, default=1e-8, help="Parity tolerance.")
    p_equiv.set_defaults(func=_cmd_equiv)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``rtmix`` command-line tool.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit status code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

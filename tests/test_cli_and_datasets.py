"""Tests for bundled datasets, simulator edge cases, and CLI subcommands."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rtmix import load_benchmark_dataset, simulate_dual_tail_exam
from rtmix.cli import build_parser, main


def test_bundled_benchmark_dataset_integrity():
    """Bundled PISA-calibrated benchmark dataset loads with expected keys and shapes."""
    d = load_benchmark_dataset()
    required = {
        "Y", "log_rt", "entropy", "theta_true", "tau_true", "nu_true",
        "pi_rg_true", "pi_ai_true", "group_label", "regime_true",
    }
    assert required.issubset(d.keys())
    assert d["Y"].shape == (480, 50)
    assert d["log_rt"].shape == (480, 50)
    assert d["entropy"].shape == (480, 50)
    assert set(np.unique(d["group_label"]).tolist()) == {0, 1, 2, 3, 4}
    assert set(np.unique(d["regime_true"]).tolist()) == {0, 1, 2}


def test_simulator_theta_pool_and_validation():
    """Simulator supports empirical theta_pool sampling and rejects invalid group sums."""
    pool = np.linspace(-2.0, 2.0, 100)
    sim = simulate_dual_tail_exam(n_examinees=100, n_items=25, theta_pool=pool, seed=9)
    assert sim["Y"].shape == (100, 25)
    assert abs(float(sim["theta_true"].mean())) < 1e-6

    with pytest.raises(ValueError):
        simulate_dual_tail_exam(
            prop_slow_careful=0.4, prop_rapid_guesser=0.4, prop_ai_copier=0.4
        )


def test_cli_demo_smoke(capsys):
    """rtmix demo prints valid JSON summary with high AI vs Slow-Careful AUROC."""
    rc = main(["demo", "--device", "numpy", "--no-ci"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["auroc_ai_vs_slow_careful_dtmix"] > 0.99
    assert out["theta_rmse_dtmix"] < out["theta_rmse_classical_irt"]


def test_cli_audit_smoke(capsys):
    """rtmix audit prints structured JSON regime flag counts."""
    rc = main(["audit", "--device", "numpy"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert sum(out["flag_counts"].values()) == 480
    assert out["flag_counts"]["AIAssisted"] > 50


def test_cli_fit_roundtrip(tmp_path, capsys):
    """rtmix fit reads an .npz archive and saves fitted posteriors and traits."""
    sim = simulate_dual_tail_exam(n_examinees=60, n_items=20, seed=11)
    in_file = tmp_path / "exam_in.npz"
    out_file = tmp_path / "exam_out.npz"
    np.savez_compressed(in_file, Y=sim["Y"], log_rt=sim["log_rt"], entropy=sim["entropy"])

    rc = main([
        "fit", "--input", str(in_file), "--output", str(out_file),
        "--device", "numpy", "--max-iter", "15", "--no-ci",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_iter"] >= 1
    assert out_file.exists()
    with np.load(out_file) as arc:
        assert arc["theta"].shape == (60,)
        assert arc["posteriors"].shape == (60, 20, 3)


def test_cli_equiv_smoke(capsys):
    """rtmix equiv passes strict parity check and exits 0."""
    rc = main(["equiv", "--n-examinees", "80", "--n-items", "25"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_passed"] is True
    assert out["max_abs_diff_theta_mstep"] < 1e-8


def test_cli_fit_missing_file_raises():
    """rtmix fit raises FileNotFoundError for missing input path."""
    with pytest.raises(FileNotFoundError):
        main(["fit", "--input", "nonexistent_exam_file_xyz.npz"])


def test_cli_parser_subcommands():
    """CLI parser registers all four subcommands."""
    parser = build_parser()
    args = parser.parse_args(["demo"])
    assert args.command == "demo"

# Contributing to rtmix

Contributions, bug reports, and methodological extensions are welcome.

## Development Workflow

1. Fork the repository and create a feature branch.
2. Ensure all public classes and functions include NumPy-style docstrings.
3. Run the full verification suite before opening a pull request:

```bash
pytest
```

4. Confirm strict numerical parity (`< 1e-8`) using:

```bash
rtmix equiv
```

## Code Style

- Target Python 3.10+.
- Maintain vectorized NumPy and PyTorch `float64` parity across CPU and CUDA backends.
- Avoid introducing heavy runtime dependencies beyond `numpy` and `scipy` (keep `torch` optional).

# Development tests

Install development tools and run the lightweight suite:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q -ra -k 'not test_optional_real_processing_regeneration'
python -m build
```

The inference regressions use PyTorch and Numba. The CPU CI job installs
those dependencies and runs the suite without GPU checkpoint weights.
Tests cover HMM coding constraints and numeric stability, persistent window
workers, memory backoff, multi-node coordination, and GFF3/GTF assembly.

The optional processing-fixture regeneration test requires a separately
configured processing checkout. Ordinary fixture replay runs in CI.

# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Before Opening a PR

Run locally:

```bash
ruff check src tests
pytest
pip-audit -r requirements.txt
bandit -r src -q
```

## Coding Guidelines

- Keep modules focused and single-responsibility.
- Add tests for behavior changes and bug fixes.
- Avoid introducing network calls in tests.
- Prefer deterministic behavior for reproducibility.

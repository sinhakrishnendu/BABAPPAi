# Contributing to BABAPPAi

Thank you for contributing to BABAPPAi.

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e .[test]
python -m pip install ruff build twine
```

## Local Checks Before Opening a PR

```bash
ruff check .
pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
python -m babappai.cli --help
python -m babappai.cli version
```

## Contribution Expectations

- Keep pull requests focused and reviewable.
- Add or update tests for behavior changes.
- Keep scientific interpretation language conservative and explicit.
- Do not commit generated benchmark trees under `results/`; use external archives for large artifacts.

## Branch and PR Guidance

- Branch from `main`.
- Use clear commit messages describing scientific or engineering impact.
- In the PR description include:
  - what changed
  - why it changed
  - validation commands and outcomes
  - any reproducibility/archive links for large outputs

## Reporting Bugs

Open a GitHub issue with:

- reproducible steps
- expected vs observed behavior
- platform and Python version
- command used and relevant log excerpt

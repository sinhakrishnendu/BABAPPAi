# Release Checklist

## Pre-Release

1. Confirm target version in `babappai/metadata.py` and `CHANGELOG.md`.
2. Confirm `CITATION.cff` version matches release version.
3. Run local checks:
   - `ruff check .`
   - `pytest -q`
   - `python -m build --sdist --wheel`
   - `python -m twine check dist/*`
4. Smoke-install from wheel:
   - `python -m pip install dist/*.whl`
   - `python -m babappai.cli --help`
   - `python -m babappai.cli version`

## GitHub Actions / PyPI

1. Ensure GitHub Actions `CI` workflow is green.
2. Ensure `publish` workflow environment `pypi` is configured for trusted publishing.
3. Create annotated tag/release for the target version.
4. Publish GitHub Release.
5. Verify `Publish` workflow completed successfully.

## Post-Release Verification

1. Verify install from PyPI in a clean environment:
   - `pip install babappai==<version>`
2. Verify CLI:
   - `babappai --help`
   - `babappai version`
3. Confirm PyPI metadata rendering and links.
4. Confirm citation metadata (`CITATION.cff`) is present and correct.

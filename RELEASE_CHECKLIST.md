# Release Checklist

## Pre-Release

1. Confirm target version in `pyproject.toml` and `CHANGELOG.md`.
2. Confirm `CITATION.cff` version matches release version and cites BABAPPAΩ model DOI.
3. Confirm calibration asset version/provenance:
   - `babappai/data/ceii_calibration_v2.json` exists
   - `calibration_version` and threshold bands are intentional for this release
   - `D_obs` definition in asset matches locked method
4. Run local checks:
   - `ruff check .`
   - `pytest -q`
   - `python -m build --sdist --wheel`
   - `python -m twine check dist/*`
5. Smoke-install from wheel:
   - `python -m pip install dist/*.whl`
   - `python -m babappai.cli --help`
   - `python -m babappai.cli version`
   - `python -m babappai.cli run --help` (confirm cEII CLI options are present)

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
   - `babappai run --alignment <...> --tree <...> --outdir <...>` includes `ceii_*` fields
3. Confirm PyPI metadata rendering and links.
4. Confirm citation metadata (`CITATION.cff`) is present and correct.

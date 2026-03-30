"""Canonical metadata for BABAPPAi and the frozen BABAPPAΩ model artifact.

The package version source of truth is ``[project].version`` in ``pyproject.toml``.
"""

from __future__ import annotations

import re
from importlib import metadata as importlib_metadata
from pathlib import Path

PACKAGE_NAME = "babappai"
SOFTWARE_NAME = "BABAPPAi"
SOFTWARE_DOI = "10.5281/zenodo.18520163"

MODEL_LINEAGE_NAME = "BABAPPAΩ"
MODEL_NAME = "BABAPPAΩ canonical frozen model"
MODEL_TAG = "frozen_babappaomega_model"
MODEL_ROLE = "canonical_inference_backbone"
MODEL_FILE_NAME = "babappaomega.pt"
MODEL_DOI = "10.5281/zenodo.18195869"
MODEL_URL = "https://zenodo.org/records/18195869/files/babappaomega.pt?download=1"
MODEL_SHA256 = "657a662563af31304abcb208fc903d2770a9184632a9bab2095db4c538fed8eb"
MODEL_COMPATIBILITY_NOTE = (
    "BABAPPAi is the operational software framework around the canonical frozen "
    "BABAPPAΩ model artifact. Model weights are fixed; BABAPPAi updates affect "
    "calibration, applicability, reporting, and packaging layers without "
    "changing the underlying neural inference backbone."
)

# Backward-compatible aliases retained for older imports.
LEGACY_CODEBASE_NAME = MODEL_LINEAGE_NAME
LEGACY_MODEL_NAME = MODEL_NAME
LEGACY_MODEL_TAG = "legacy_frozen"


def _read_pyproject_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        raise RuntimeError("Could not find [project].version in pyproject.toml")
    return match.group(1)


def resolve_software_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject_path.exists():
        return _read_pyproject_version()

    try:
        return importlib_metadata.version(PACKAGE_NAME)
    except importlib_metadata.PackageNotFoundError:
        raise RuntimeError(
            "Could not resolve software version from pyproject.toml or installed package metadata"
        )


SOFTWARE_VERSION = resolve_software_version()

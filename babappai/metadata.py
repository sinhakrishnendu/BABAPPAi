"""Canonical metadata for BABAPPAi and legacy frozen assets.

The package version source of truth is ``[project].version`` in ``pyproject.toml``.
"""

from __future__ import annotations

import re
from importlib import metadata as importlib_metadata
from pathlib import Path

PACKAGE_NAME = "babappai"
SOFTWARE_NAME = "BABAPPAi"
SOFTWARE_DOI = "10.5281/zenodo.18520163"

LEGACY_CODEBASE_NAME = "BABAPPAΩ"
LEGACY_MODEL_NAME = "BABAPPAΩ frozen model"
MODEL_TAG = "legacy_frozen"
MODEL_FILE_NAME = "babappaomega.pt"
MODEL_DOI = "10.5281/zenodo.18195869"
MODEL_URL = "https://zenodo.org/records/18195869/files/babappaomega.pt?download=1"
MODEL_SHA256 = "657a662563af31304abcb208fc903d2770a9184632a9bab2095db4c538fed8eb"
MODEL_COMPATIBILITY_NOTE = (
    "BABAPPAi is the renamed continuation of the BABAPPAΩ codebase. "
    "The cached frozen model is a legacy BABAPPAΩ asset used for "
    "backward-compatible inference until BABAPPAi-specific weights are released."
)


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

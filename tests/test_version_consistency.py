from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import babappai.cli as cli
from babappai import __version__


def _read_pyproject_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        raise AssertionError("Could not find static [project].version in pyproject.toml")
    return match.group(1)


def test_pyproject_version_matches_runtime_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert _read_pyproject_version(pyproject) == __version__


def test_installed_metadata_version_matches_runtime_version():
    cmd = [
        sys.executable,
        "-c",
        "from importlib import metadata; print(metadata.version('babappai'))",
    ]
    with TemporaryDirectory() as temp_dir:
        output = subprocess.check_output(cmd, cwd=temp_dir, text=True).strip()
    if output != __version__ and os.environ.get("CI", "").lower() != "true":
        pytest.skip(
            "Installed distribution version does not match this checkout. "
            "Run after `pip install -e .` to enforce locally."
        )
    assert output == __version__


def test_cli_version_matches_runtime_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "model_status", lambda: {"cached": True})
    code = cli.main(["version"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"software_version={__version__}" in out

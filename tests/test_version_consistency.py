from __future__ import annotations

import importlib
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import babappai.cli as cli
from babappai import __version__


def _get_section(text: str, section_name: str) -> str:
    pattern = rf"(?ms)^\[{re.escape(section_name)}\]\n(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    if not match:
        raise AssertionError(f"Missing [{section_name}] in pyproject.toml")
    return match.group(1)


def _resolve_pyproject_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    project_section = _get_section(text, "project")
    static_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_section)
    if static_match:
        return static_match.group(1)

    dynamic_section = _get_section(text, "tool.setuptools.dynamic")
    dynamic_match = re.search(
        r'(?m)^version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"\s*\}\s*$',
        dynamic_section,
    )
    if not dynamic_match:
        raise AssertionError("Could not resolve dynamic version attr from pyproject.toml")

    attr_path = dynamic_match.group(1)
    module_name, attr_name = attr_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return str(getattr(module, attr_name))


def test_pyproject_version_matches_runtime_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert _resolve_pyproject_version(pyproject) == __version__


def test_installed_metadata_version_matches_runtime_version():
    cmd = [
        sys.executable,
        "-c",
        "from importlib import metadata; print(metadata.version('babappai'))",
    ]
    with TemporaryDirectory() as temp_dir:
        output = subprocess.check_output(cmd, cwd=temp_dir, text=True).strip()
    assert output == __version__


def test_cli_version_matches_runtime_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "model_status", lambda: {"cached": True})
    code = cli.main(["version"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"software_version={__version__}" in out

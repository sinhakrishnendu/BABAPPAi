"""Adapter layer for external neutral calibration generator scripts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from babappai.calibration import _get_reference_path


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _discover_reference_file(output_dir: Path, model_tag: str) -> Optional[Path]:
    candidates = sorted(output_dir.glob("neutral_reference*.json"))
    if candidates:
        return candidates[0]

    package_reference = _get_reference_path(model_tag)
    if package_reference.exists():
        return package_reference
    return None


def run_neutral_generator(
    *,
    generator_path: str,
    output_dir: str,
    model_tag: str,
    seed: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Execute an external neutral-generator script with robust fallback behavior."""

    script = Path(generator_path).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Neutral generator script not found: {script}")

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    extra = list(extra_args or [])
    command_with_args = [sys.executable, str(script), "--outdir", str(outdir)]
    if seed is not None:
        command_with_args.extend(["--seed", str(seed)])
    command_with_args.extend(extra)

    attempted_commands: list[list[str]] = [command_with_args]
    first = _run_command(command_with_args, cwd=outdir)

    if first.returncode != 0:
        fallback = [sys.executable, str(script), *extra]
        attempted_commands.append(fallback)
        second = _run_command(fallback, cwd=outdir)
        result = second
    else:
        result = first

    if result.returncode != 0:
        raise RuntimeError(
            "Neutral generator failed. "
            f"Attempted commands: {attempted_commands}. "
            f"stderr: {result.stderr.strip()}"
        )

    discovered = _discover_reference_file(outdir, model_tag=model_tag)
    if discovered is None:
        raise RuntimeError(
            "Neutral generator completed but no neutral reference JSON was found."
        )

    copied_reference = outdir / discovered.name
    if discovered.resolve() != copied_reference.resolve():
        shutil.copy2(discovered, copied_reference)

    metadata = {
        "generator_path": str(script),
        "attempted_commands": attempted_commands,
        "seed": seed,
        "output_dir": str(outdir),
        "reference_file": str(copied_reference),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (outdir / "neutral_generator_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


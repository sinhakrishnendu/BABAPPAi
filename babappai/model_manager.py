"""Frozen model download, cache, and verification for BABAPPAi."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

from platformdirs import user_cache_dir

from babappai.metadata import (
    LEGACY_MODEL_TAG,
    MODEL_COMPATIBILITY_NOTE,
    MODEL_DOI,
    MODEL_FILE_NAME,
    MODEL_LINEAGE_NAME,
    MODEL_NAME,
    MODEL_ROLE,
    MODEL_SHA256,
    MODEL_TAG,
    MODEL_URL,
    PACKAGE_NAME,
)


class ModelError(RuntimeError):
    """Raised when model cache/download/verification operations fail."""


def get_cache_dir() -> Path:
    override = os.getenv("BABAPPAI_CACHE_DIR")
    candidates = []

    if override:
        candidates.append(Path(override))

    candidates.append(Path(user_cache_dir(PACKAGE_NAME)))
    candidates.append(Path(tempfile.gettempdir()) / f"{PACKAGE_NAME}-cache")

    last_error: Optional[Exception] = None
    for cache in candidates:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            return cache
        except OSError as exc:
            last_error = exc

    raise ModelError(
        "Unable to create model cache directory. "
        "Set BABAPPAI_CACHE_DIR to a writable directory."
    ) from last_error


def model_cache_path() -> Path:
    return get_cache_dir() / MODEL_FILE_NAME


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def model_status() -> Dict[str, object]:
    path = model_cache_path()
    exists = path.exists()
    actual_sha256: Optional[str] = None
    verified = False

    if exists:
        actual_sha256 = sha256sum(path)
        verified = actual_sha256 == MODEL_SHA256

    return {
        "model_tag": MODEL_TAG,
        "model_name": MODEL_NAME,
        "model_lineage": MODEL_LINEAGE_NAME,
        "model_role": MODEL_ROLE,
        "file_name": MODEL_FILE_NAME,
        "doi": MODEL_DOI,
        "url": MODEL_URL,
        "expected_sha256": MODEL_SHA256,
        "cached": exists,
        "cached_path": str(path),
        "actual_sha256": actual_sha256,
        "verified": verified,
        "compatibility_note": MODEL_COMPATIBILITY_NOTE,
    }


def verify_cached_model() -> Dict[str, object]:
    status = model_status()
    if not status["cached"]:
        raise ModelError(
            "Frozen model is not cached. Run 'babappai model fetch' first."
        )
    if not status["verified"]:
        raise ModelError(
            "Cached model checksum mismatch. Re-run 'babappai model fetch' "
            "to download a verified copy."
        )
    return status


def _download_to_tempfile(url: str) -> Path:
    with urlopen(url, timeout=120) as response:
        with NamedTemporaryFile(delete=False, suffix=".download") as tmp:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                tmp.write(chunk)
            return Path(tmp.name)


def fetch_model(*, force: bool = False, offline: bool = False) -> Dict[str, object]:
    existing = model_status()
    path = Path(existing["cached_path"])

    if existing["cached"] and existing["verified"] and not force:
        return {**existing, "downloaded": False}

    if offline:
        raise ModelError(
            "Offline mode is enabled and no verified cached model is available."
        )

    if path.exists() and not path.is_file():
        raise ModelError(f"Model cache path is not a file: {path}")

    tmp_path: Optional[Path] = None
    try:
        tmp_path = _download_to_tempfile(MODEL_URL)
        actual_sha = sha256sum(tmp_path)
        if actual_sha != MODEL_SHA256:
            raise ModelError(
                "Model download failed checksum verification "
                f"(expected {MODEL_SHA256}, got {actual_sha})."
            )

        os.replace(tmp_path, path)
        refreshed = model_status()
        return {**refreshed, "downloaded": True}
    except URLError as exc:
        raise ModelError(
            "Failed to download frozen model from Zenodo. "
            "Check your network or run again without --offline."
        ) from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_model(model_tag: str = MODEL_TAG, *, offline: bool = False) -> Path:
    accepted_tags = {MODEL_TAG, "canonical_frozen_model", "frozen", LEGACY_MODEL_TAG}
    if model_tag not in accepted_tags:
        raise ValueError(f"Unknown model tag: {model_tag}")

    status = fetch_model(offline=offline)
    if not status["verified"]:
        raise ModelError("Model checksum verification failed.")

    return Path(status["cached_path"])

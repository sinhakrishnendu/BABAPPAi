"""Backward-compatible import surface for model management."""

from babappai.model_manager import (  # noqa: F401
    ModelError,
    ensure_model,
    fetch_model,
    get_cache_dir,
    model_cache_path,
    model_status,
    sha256sum,
    verify_cached_model,
)


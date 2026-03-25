import hashlib
from pathlib import Path

import pytest

import babappai.model_manager as mm


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mm, "get_cache_dir", lambda: cache)
    return cache


def test_fetch_first_download(monkeypatch, tmp_path, isolated_cache):
    payload = b"frozen-model-content"
    expected = _sha(payload)
    monkeypatch.setattr(mm, "MODEL_SHA256", expected)

    source = tmp_path / "source.pt"
    source.write_bytes(payload)
    monkeypatch.setattr(mm, "_download_to_tempfile", lambda url: source)

    status = mm.fetch_model()
    assert status["downloaded"] is True
    assert status["verified"] is True


def test_cache_reuse_without_download(monkeypatch, isolated_cache):
    payload = b"cached-model"
    monkeypatch.setattr(mm, "MODEL_SHA256", _sha(payload))
    (isolated_cache / mm.MODEL_FILE_NAME).write_bytes(payload)

    monkeypatch.setattr(
        mm,
        "_download_to_tempfile",
        lambda url: (_ for _ in ()).throw(RuntimeError("should-not-download")),
    )

    status = mm.fetch_model()
    assert status["downloaded"] is False
    assert status["verified"] is True


def test_checksum_mismatch_failure(monkeypatch, tmp_path, isolated_cache):
    monkeypatch.setattr(mm, "MODEL_SHA256", _sha(b"expected"))
    source = tmp_path / "bad.pt"
    source.write_bytes(b"unexpected")
    monkeypatch.setattr(mm, "_download_to_tempfile", lambda url: source)

    with pytest.raises(mm.ModelError):
        mm.fetch_model()


def test_offline_failure_when_missing(isolated_cache):
    with pytest.raises(mm.ModelError, match="Offline mode"):
        mm.fetch_model(offline=True)


def test_status_contains_compatibility_note(monkeypatch, isolated_cache):
    payload = b"ok"
    monkeypatch.setattr(mm, "MODEL_SHA256", _sha(payload))
    (isolated_cache / mm.MODEL_FILE_NAME).write_bytes(payload)

    status = mm.model_status()
    assert "compatibility_note" in status
    assert "BABAPPAi" in status["compatibility_note"]

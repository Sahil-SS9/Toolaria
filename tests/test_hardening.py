"""Phase 0 hardening tests (T0.1-T0.5).

TDD discipline: every test was written before the implementation, witnessed
RED, then driven GREEN by the smallest production change. Each cluster
groups the assertions required by the plan's testing matrix for one task.
"""
import json
import logging
import os
from pathlib import Path

import pytest

import blobstore as _blobstore_mod
from blobstore import BlobStore


# ═══ T0.1 — Explicit permissions (0700 dirs / 0600 files) + chmod migration ═══


@pytest.fixture
def fresh_cfg(toolaria, tmp_path):
    """A minimal cfg pointing at a fresh tmp_path store. Not registered."""
    return {
        **toolaria._cfg,
        "store_path": str(tmp_path / "store_t01"),
        "ttl_hours": 1,
    }


def test_blob_file_is_0600_under_umask0(fresh_cfg, tmp_path):
    """put() must chmod the blob file to 0600 even when umask=0.

    write_bytes() creates files honouring umask, so without an explicit
    chmod this would land at 0o666 under umask=0 — RED witness for T0.1.
    """
    old = os.umask(0)
    try:
        bs = BlobStore(fresh_cfg)
        bid = bs.put("DATA" * 200, "web_search", session_id="sess-x")
        mode = (bs.blob_dir / bid).stat().st_mode & 0o777
        assert mode == 0o600, f"blob file mode {oct(mode)}, expected 0o600"
    finally:
        os.umask(old)


def test_idx_file_is_0600_under_umask0(fresh_cfg):
    """Session index writes must be 0600 (mkstemp default; verified by
    explicit chmod after os.replace so the guarantee survives a future
    refactor that switches write paths)."""
    old = os.umask(0)
    try:
        bs = BlobStore(fresh_cfg)
        bs.put("hello", "web_search", session_id="sess-y")
        idx_path = bs.meta_dir / f"{BlobStore._safe_sid('sess-y')}.json"
        mode = idx_path.stat().st_mode & 0o777
        assert mode == 0o600, f"index file mode {oct(mode)}, expected 0o600"
    finally:
        os.umask(old)


def test_sidecar_file_is_0600_under_umask0(fresh_cfg):
    """Sidecar writes (outline/chunks/vectors) must be 0600."""
    old = os.umask(0)
    try:
        bs = BlobStore(fresh_cfg)
        bs.write_sidecar("abcdef012345", "outline", {"x": 1})
        path = bs.sidecar_dir / "abcdef012345.outline.json"
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"sidecar file mode {oct(mode)}, expected 0o600"
    finally:
        os.umask(old)


def test_dirs_are_0700_under_umask0(fresh_cfg):
    """Store root and sub-dirs (blobs/sessions/sidecars) must all be 0700
    even when umask=0 (mkdir honours umask; without explicit chmod the dirs
    would land at 0o777)."""
    old = os.umask(0)
    try:
        bs = BlobStore(fresh_cfg)
        bp = Path(fresh_cfg["store_path"]).expanduser().resolve()
        for d in (bp, bs.blob_dir, bs.meta_dir, bs.sidecar_dir):
            mode = d.stat().st_mode & 0o777
            assert mode == 0o700, f"dir {d} mode {oct(mode)}, expected 0o700"
    finally:
        os.umask(old)


def test_migration_tightens_existing_permissive_tree(toolaria, tmp_path):
    """An older store with deliberately-permissive perms gets tightened on
    init (0755 dirs -> 0700, 0644 files -> 0600)."""
    sp = tmp_path / "store_legacy"
    (sp / "blobs").mkdir(parents=True)
    (sp / "sessions").mkdir()
    (sp / "sidecars").mkdir()
    # Older install left the tree at 0755/0644.
    os.chmod(sp / "blobs", 0o755)
    os.chmod(sp / "sessions", 0o755)
    os.chmod(sp / "sidecars", 0o755)
    (sp / "blobs" / "deadbeef0000").write_bytes(b"x" * 100)
    os.chmod(sp / "blobs" / "deadbeef0000", 0o644)
    idx = sp / "sessions" / "legacy.json"
    idx.write_text('{"blobs": {}}')
    os.chmod(idx, 0o644)

    cfg = {**toolaria._cfg, "store_path": str(sp)}
    BlobStore(cfg)

    for d in (sp / "blobs", sp / "sessions", sp / "sidecars"):
        mode = d.stat().st_mode & 0o777
        assert mode == 0o700, (
            f"migrated dir {d} mode {oct(mode)}, expected 0o700")
    for f in (sp / "blobs" / "deadbeef0000", idx):
        mode = f.stat().st_mode & 0o777
        assert mode == 0o600, (
            f"migrated file {f} mode {oct(mode)}, expected 0o600")


def test_migration_chmod_failure_does_not_crash_init(
    toolaria, tmp_path, monkeypatch, caplog
):
    """A chmod failure (e.g. read-only mount) logs a warning but init still
    succeeds. Without the safe wrapper, init would raise and the store
    would never come up."""
    sp = tmp_path / "store_chmod_fail"
    sp.mkdir()
    (sp / "blobs").mkdir()
    os.chmod(sp / "blobs", 0o755)

    cfg = {**toolaria._cfg, "store_path": str(sp)}

    real_chmod = os.chmod

    def boom(p, mode, *a, **kw):
        if str(p).startswith(str(sp)):
            raise OSError("simulated chmod failure (read-only fs)")
        return real_chmod(p, mode, *a, **kw)

    monkeypatch.setattr(os, "chmod", boom)

    with caplog.at_level(logging.WARNING, logger="blobstore"):
        bs = BlobStore(cfg)
    assert bs is not None
    assert any("chmod" in r.getMessage().lower() for r in caplog.records), (
        f"expected a chmod warning, got: {[r.getMessage() for r in caplog.records]}"
    )

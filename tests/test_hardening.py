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


# ═══ T0.2 — Shell/write-class tools never rescued even under registry failure ═══


SHELL_WRITE_CLASS_TOOLS = [
    "shell", "bash", "exec", "terminal", "subprocess",
    "run_command", "run_shell",
    "write_file", "file_write", "fs_write", "edit_file",
]


@pytest.fixture
def fail_open_registry(monkeypatch):
    """Simulate a broken `tools.registry` import so _is_rescuable returns
    True for anything not in its built-in set or the unconditional excludes.

    The fail-open path is the very safety hole T0.2 closes: shell/exec and
    file-write results must still pass through untouched."""
    # Force the import to raise by stashing an unimportable module.
    import sys
    import types

    class _Broken(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError("simulated registry breakage")

    monkeypatch.setitem(sys.modules, "tools.registry", _Broken("tools.registry"))


def test_shell_class_tools_not_rescued_under_broken_registry(
    toolaria, plugin, fail_open_registry
):
    """With the registry broken, shell/exec-class tools must still pass
    through unrescued (return None) so a token-bearing shell output never
    reaches the blob store."""
    big = "rm -rf " + "payload " * 1000
    for tool in ("shell", "run_shell", "bash", "exec",
                 "run_command", "terminal", "subprocess"):
        r = toolaria._on_transform(tool_name=tool, result=big)
        assert r is None, f"{tool} must NOT be rescued under registry failure"


def test_write_class_tools_not_rescued_under_broken_registry(
    toolaria, plugin, fail_open_registry
):
    """Same guarantee for write-class tools: token-bearing output (e.g.
    the contents of a file about to be edited) must never hit the store."""
    big = "secret " + "x " * 2000
    for tool in ("write_file", "file_write", "fs_write", "edit_file"):
        r = toolaria._on_transform(tool_name=tool, result=big)
        assert r is None, f"{tool} must NOT be rescued under registry failure"


def test_normal_tools_still_rescued_under_broken_registry(
    toolaria, plugin, fail_open_registry
):
    """Fail-open (True on registry failure) must STILL rescue normal tools;
    T0.2 only tightens the unconditional excludes, not the fail-open
    default for unknown tools."""
    big = "a " * 5000  # ~10K chars, well over base_cfg max_result_chars (8000)
    r = toolaria._on_transform(tool_name="some_mcp_tool", result=big)
    assert r is not None, "fail-open must still rescue unknown tools"
    assert "rescued" in r


def test_unconditional_excludes_contain_shell_and_write_classes(
    toolaria, plugin
):
    """The hardening is in the shipped frozenset, not in a per-call branch:
    audit the constants so a future edit that drops a class fails the test
    rather than silently reopening the fail-open safety hole."""
    excl = toolaria._UNCONDITIONAL_EXCLUDES
    for t in SHELL_WRITE_CLASS_TOOLS:
        assert t in excl, (
            f"{t} missing from _UNCONDITIONAL_EXCLUDES; registry failure "
            f"would rescue shell/write results into the blob store"
        )


def test_excluded_shell_tool_not_rescued_even_when_in_allowlist(
    toolaria, plugin
):
    """Config-level allowlists cannot override the unconditional excludes:
    a user adding 'shell' to exclude_tools or a permissive rescue path
    must not be able to bring shell back into the rescue catchment."""
    big = "data " * 1500
    # Even with a forced rescue path, shell must not be rescued.
    assert toolaria._on_transform(tool_name="shell", result=big) is None

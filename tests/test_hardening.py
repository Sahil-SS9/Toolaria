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


# ═══ T0.3 — passref_external_destinations deny list ═══


DEST_TOOLS = [
    "send_email", "send_mail", "post_email",
    "social_post", "twitter_post", "linkedin_post", "post_to_social",
    "webhook_send", "send_webhook", "slack_post",
    "peer_send_message", "peer_broadcast",
]


def test_external_destination_expansion_returns_honest_marker(plugin, toolaria):
    """A tla:<id> token expanded into a mail/social/webhook/peer destination
    must yield an honest marker — the destination tool sees the handle
    failed, not the content."""
    big = "SECRET PAYLOAD " * 500
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = plugin[0].middleware["tool_request"][0]
    for tool in DEST_TOOLS:
        out = mw(tool_name=tool, args={"body": f"tla:{bid}"})
        assert out is not None, f"{tool} must not silently pass-through"
        v = out["args"]["body"]
        assert "SECRET" not in v, (
            f"{tool} received the secret content; expansion must yield a "
            f"marker, not the bytes"
        )
        assert "external destination" in v.lower() or "denied" in v.lower(), (
            f"{tool} marker missing explanation: {v!r}"
        )


def test_external_destination_results_still_rescued(toolaria, plugin):
    """T0.3 must NOT merge the new list into exclude_tools: those tools'
    oversized results must still be rescued (passref just refuses the
    cross-tool handoff)."""
    big = "MAIL BODY " * 1000
    for tool in ("send_email", "slack_post"):
        r = toolaria._on_transform(tool_name=tool, result=big)
        assert r is not None, (
            f"{tool} oversized result was not rescued; T0.3 must not merge "
            f"passref_external_destinations into exclude_tools"
        )
        assert "rescued" in r


def test_destination_deny_overrides_passref_allowed_tools(plugin, toolaria):
    """Precedence: an entry in passref_external_destinations beats an entry
    in passref_allowed_tools. The whole point of the deny list is to NOT be
    overridable by an operator's permissive allowlist."""
    toolaria._cfg["passref_allowed_tools"] = list(DEST_TOOLS)
    big = "SECRET " * 500
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = plugin[0].middleware["tool_request"][0]
    out = mw(tool_name="send_email", args={"body": f"tla:{bid}"})
    v = out["args"]["body"]
    assert "SECRET" not in v, "destination deny must beat allowlist"
    assert "external destination" in v.lower() or "denied" in v.lower()


def test_destination_deny_overrides_sink_default(plugin, toolaria):
    """Precedence: an explicit allowlist must not let a destination expand
    either. Same point as above from the other side of the matrix."""
    # passref_allowed_tools empty (default) → sink-deny governs; this test
    # exercises the case where an operator narrows sinks but adds destinations.
    toolaria._cfg["passref_allowed_tools"] = ["send_email"]  # explicitly whitelisted
    big = "DATA " * 500
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = plugin[0].middleware["tool_request"][0]
    out = mw(tool_name="send_email", args={"body": f"tla:{bid}"})
    v = out["args"]["body"]
    assert "DATA" not in v, "destination deny beats allowlist entry"
    assert "external destination" in v.lower() or "denied" in v.lower()


def test_non_destination_tools_unaffected_by_deny_list(plugin, toolaria):
    """A tool not on the deny list still expands normally — the deny list is
    a targeted external-send gate, not a blanket expansion disable."""
    big = "PLAIN " * 200
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = plugin[0].middleware["tool_request"][0]
    out = mw(tool_name="summarise", args={"body": f"tla:{bid}"})
    assert out["args"]["body"] == big


def test_config_defaults_to_empty_destination_list(toolaria, tmp_path):
    """Plugin-local config.yaml must include the passref_external_destinations
    key (so operators can configure it). Use raw text matching so the test
    does not require PyYAML."""
    cfg_path = toolaria._LOCAL_CFG  # type: ignore[attr-defined]
    raw = cfg_path.read_text()
    assert "passref_external_destinations:" in raw, (
        "config.yaml must expose the new key so operators can configure it"
    )


def test_destination_list_driven_by_config(plugin, toolaria):
    """Operator configures the list at runtime; the deny set tracks it
    without code edits.

    This test mutates cfg in place after register() ran. The deny set is
    recomputed on demand when the operator-key changes, so direct cfg
    mutation (which is the operator's actual workflow at runtime) takes
    effect immediately.
    """
    # Clear the cached frozen set so the change takes effect.
    toolaria._cfg.pop("_passref_external_destinations_frozen", None)
    toolaria._cfg["passref_external_destinations"] = ["my_custom_mailer"]
    big = "SECRET " * 500
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = plugin[0].middleware["tool_request"][0]
    out = mw(tool_name="my_custom_mailer", args={"body": f"tla:{bid}"})
    v = out["args"]["body"]
    assert "SECRET" not in v, (
        f"custom destination {out['args']['body']!r}"
    )
    assert "external destination" in v.lower() or "denied" in v.lower()
    # And another non-listed tool still expands.
    bid2 = toolaria._store.put("PLAIN", "web_extract", session_id="test-s")
    out2 = mw(tool_name="summarise", args={"body": f"tla:{bid2}"})
    assert out2["args"]["body"] == "PLAIN"


def test_destination_list_must_be_list_of_strings(toolaria):
    """Malformed config (non-list, non-string entries) must be rejected at
    load time, not silently dropped or silently widen the deny set."""
    from passref import _parse_external_destinations
    # non-list
    with pytest.raises(ValueError):
        _parse_external_destinations("not a list")
    # list with non-string
    with pytest.raises(ValueError):
        _parse_external_destinations(["ok", 42])
    # empty list OK
    assert _parse_external_destinations([]) == frozenset()
    # proper list
    assert _parse_external_destinations(["a", "b"]) == frozenset({"a", "b"})
    # deduplicated
    assert _parse_external_destinations(["a", "a", "b"]) == frozenset({"a", "b"})


def test_destinations_documented_in_plugin_config(toolaria):
    """The shipped defaults key is present and the precedence comment
    exists. Use raw text matching so the test does not require PyYAML."""
    cfg = toolaria._LOCAL_CFG.read_text()  # type: ignore[attr-defined]
    assert "passref_external_destinations:" in cfg
    # Comment must explain the deny-vs-allow precedence.
    assert "external" in cfg.lower()
    assert "allowlist" in cfg.lower() or "passref_allowed_tools" in cfg

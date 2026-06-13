"""Tests for Toolaria (rescuer) plugin: rescue, fetch, sweep, registration."""
import time
from pathlib import Path

import pytest


# ═══ Rescue path ═══


def test_rescue_json_fixture(plugin, toolaria):
    """279k-char JSON becomes excerpt + handle that fits."""
    fixture = Path(__file__).parent / "fixture_23767.json"
    raw = fixture.read_text()
    assert len(raw) > 200000

    result = toolaria._on_transform(tool_name="web_extract", result=raw)
    assert result is not None
    assert "rescued" in result
    assert "rescuer_fetch" in result
    assert "NOT the full output" in result
    assert len(result) < 12000

    bid = _parse_blob_id(result)
    assert bid and len(bid) == 12


def _parse_blob_id(handle: str) -> str:
    for token in handle.splitlines()[0].split(";"):
        token = token.strip().rstrip("]")
        if token.startswith("blob="):
            return token[len("blob="):]
    return ""


def test_handle_header_fields(plugin, toolaria):
    result = toolaria._on_transform(tool_name="web_search", result="y\n" * 8000)
    header = result.splitlines()[0]
    assert header.startswith("[Toolaria: tool result rescued.")
    assert "tool=web_search" in header
    assert "size=" in header and "lines=" in header and "blob=" in header


def test_small_result_passthrough(plugin, toolaria):
    assert toolaria._on_transform(tool_name="web_search", result="small") is None


def test_rescue_without_store_returns_none(plugin, toolaria):
    """No durable store means no rescue: original result passes through."""
    toolaria._store = None
    r = toolaria._on_transform(tool_name="web_search", result="x" * 9000)
    assert r is None


def test_rescue_put_failure_returns_none_and_logs(plugin, toolaria, caplog):
    """A failing blob write must not destroy the original result."""
    def boom(*a, **kw):
        raise OSError("disk full")
    toolaria._store.put = boom
    with caplog.at_level("WARNING"):
        r = toolaria._on_transform(tool_name="web_search", result="x" * 9000)
    assert r is None
    assert "blob write failed" in caplog.text


def test_rescuer_fetch_never_rescued(plugin, toolaria):
    """rescuer_fetch output is excluded: no spill-fetch circularity."""
    r = toolaria._on_transform(tool_name="rescuer_fetch", result="x" * 50000)
    assert r is None


def test_unlisted_tool_not_rescued(plugin, toolaria):
    for tool in ("delegate_task", "session_search"):
        assert toolaria._on_transform(tool_name=tool, result="x" * 9000) is None


def test_nil_result_passthrough(plugin, toolaria):
    assert toolaria._on_transform(tool_name="web_search", result="") is None
    assert toolaria._on_transform(tool_name="web_search", result=None) is None


# ═══ Fetch modes ═══


def test_fetch_range_echoes_position(plugin, toolaria):
    content = "\n".join(f"line {i}" for i in range(100))
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 10, "count": 5})
    assert "[lines 10..14 of 100]" in r
    assert "line 10" in r and "line 14" in r


def test_fetch_range_clamps_past_end(plugin, toolaria):
    content = "\n".join(f"line {i}" for i in range(10))
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 500, "count": 5})
    assert "clamped" in r
    assert "line 9" in r


def test_fetch_grep_matches_with_line_numbers(plugin, toolaria):
    content = "\n".join(f"row_{i}: val_{i}" for i in range(1000))
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": "row_42:"})
    assert "row_42: val_42" in r
    assert r.splitlines()[0].startswith("42:")


def test_grep_json_key_colon_quotes(plugin, toolaria):
    """Punctuation greps fine: the old character allowlist blocked ':'."""
    content = '{"name": "value"}\n{"other": 1}'
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": '"name":'})
    assert '"name"' in r


def test_grep_email_at_sign(plugin, toolaria):
    content = "contact: someone@example.com\nnothing here"
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "grep",
                              "pattern": "someone@example.com"})
    assert "someone@example.com" in r


def test_grep_spaces_allowed(plugin, toolaria):
    content = "line one two three\nline four five six\n"
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": "one two"})
    assert "one two" in r


def test_grep_redos_pattern_bounded(plugin, toolaria):
    """A catastrophic-backtracking pattern must not hang the search.

    (a|a)*c on a long run of 'a' with no 'c' is exponential in vanilla re;
    the regex engine's per-search timeout (or the literal fallback's refusal)
    must keep this fast."""
    bid = toolaria._store.put("a" * 1500, "web_search", session_id="test-s")
    t0 = time.time()
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": "(a|a)*c"})
    assert time.time() - t0 < 3.0
    assert isinstance(r, str)


def test_grep_quantifier_chain_bounded(plugin, toolaria):
    """Polynomial blowup (a*a*a*...) is also bounded."""
    bid = toolaria._store.put("a" * 1500, "web_search", session_id="test-s")
    t0 = time.time()
    toolaria._fetch(args={"id": bid, "mode": "grep",
                          "pattern": "a*a*a*a*a*a*a*a*X"})
    assert time.time() - t0 < 3.0


def test_grep_control_chars_rejected(plugin, toolaria):
    bid = toolaria._store.put("data", "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": "a\x00b"})
    assert "control character" in r


def test_grep_long_single_line_bounded(plugin, toolaria):
    """A single multi-hundred-KB line is sliced, not scanned in full."""
    fixture = Path(__file__).parent / "fixture_23767.json"
    raw = fixture.read_text()
    bid = toolaria._store.put(raw, "web_extract", session_id="test-s")
    t0 = time.time()
    r = toolaria._fetch(args={"id": bid, "mode": "grep", "pattern": "zzz_absent"})
    assert time.time() - t0 < 3.0
    assert "no matches" in r


def test_full_mode_returns_entire_content(plugin, toolaria):
    content = "x" * 10000  # over fetch_max_chars, under full_fetch_max_chars
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "full"})
    assert r == content


def test_full_mode_refused_over_threshold(plugin, toolaria):
    content = "x" * 60000  # over full_fetch_max_chars (50000)
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "full"})
    assert "Refused" in r and "range" in r


def test_full_mode_unrestricted_when_disabled(plugin, toolaria, base_cfg):
    toolaria._store.cfg["refuse_full_fetch"] = False
    content = "x" * 60000
    bid = toolaria._store.put(content, "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "full"})
    assert r == content


def test_stat_with_forwarded_session_id(plugin, toolaria):
    bid = toolaria._store.put("s" * 2500, "mcp_tool", session_id="other-s")
    r = toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="other-s")
    assert "blob:" in r and "mcp_tool" in r


def test_stat_without_session_searches_indexes(plugin, toolaria):
    """stat finds the tool name when legacy dispatchers forward no session id."""
    bid = toolaria._store.put("s" * 2500, "mcp_tool", session_id="some-s")
    r = toolaria._fetch(args={"id": bid, "mode": "stat"})
    assert "mcp_tool" in r


def test_fetch_with_forwarded_session_denies_cross_session_read(plugin, toolaria):
    """A forwarded session_id turns rescuer_fetch into a session-scoped read.

    Blob ids are short capabilities. When the host knows the calling session,
    a guessed id from another session must not reveal bytes or metadata.
    """
    bid = toolaria._store.put("SECRET DATA", "web_search", session_id="owner-s")
    r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0, "count": 1},
                        session_id="other-s")
    assert "not available in this session" in r
    assert "SECRET" not in r


def test_fetch_without_session_keeps_legacy_global_read(plugin, toolaria):
    """Back-compat: older Hermes dispatchers may not pass session_id yet."""
    bid = toolaria._store.put("LEGACY READ", "web_search", session_id="owner-s")
    r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0, "count": 1})
    assert "LEGACY READ" in r


def test_dedup(plugin, toolaria):
    c = "uniq-" + "dat-" * 200
    b1 = toolaria._store.put(c, "web_search", session_id="test-s")
    b2 = toolaria._store.put(c, "web_search", session_id="test-s")
    assert b1 == b2


def test_missing_blob(plugin, toolaria):
    r = toolaria._fetch(args={"id": "000000000000", "mode": "stat"})
    assert "not found" in r


def test_invalid_blob_id(plugin, toolaria):
    """Path traversal guard: non-hex blob_id rejected."""
    r = toolaria._fetch(args={"id": "../../../etc/passwd", "mode": "stat"})
    assert "invalid" in r.lower()


def test_fetch_unknown_mode(plugin, toolaria):
    bid = toolaria._store.put("test content", "web_search", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "unknown"})
    assert "unknown mode" in r


def test_outline_mode_json(plugin, toolaria):
    import json
    rows = [{"id": i, "score": i * 1.5} for i in range(50)]
    bid = toolaria._store.put(json.dumps(rows), "web_extract", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "outline"}, session_id="test-s")
    assert "JSON outline" in r
    assert "score" in r and "min=" in r


def test_search_mode_lexical(plugin, toolaria, monkeypatch):
    """Lexical (BM25) search returns the relevant chunk with line numbers."""
    monkeypatch.setattr(toolaria.blobstore._sem, "embeddings_available",
                        lambda: False)
    lines = (["intro paragraph about nothing"] * 20
             + ["the secret password is hunter2"]
             + ["more filler text here"] * 20)
    bid = toolaria._store.put("\n".join(lines), "web_extract", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "search", "query": "secret password"},
                        session_id="test-s")
    assert "lexical" in r
    assert "hunter2" in r
    assert "lines " in r  # hit carries a line range for range-mode follow-up


def test_search_requires_query(plugin, toolaria, monkeypatch):
    monkeypatch.setattr(toolaria.blobstore._sem, "embeddings_available",
                        lambda: False)
    bid = toolaria._store.put("a\nb\nc", "web_extract", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "search"}, session_id="test-s")
    assert "requires query" in r


def test_search_caches_chunks(plugin, toolaria, monkeypatch):
    monkeypatch.setattr(toolaria.blobstore._sem, "embeddings_available",
                        lambda: False)
    bid = toolaria._store.put("\n".join(f"row {i}" for i in range(50)),
                              "web_extract", session_id="test-s")
    toolaria._fetch(args={"id": bid, "mode": "search", "query": "row 7"},
                    session_id="test-s")
    assert toolaria._store.read_sidecar(bid, "chunks") is not None


def test_search_semantic_with_stub(plugin, toolaria, monkeypatch):
    """With embeddings present, search uses the semantic path and caches vectors."""
    sem = toolaria.blobstore._sem

    def stub_embed(texts, model):
        # apple -> [1,0], banana -> [0,1], else -> [0,0]
        out = []
        for t in texts:
            tl = t.lower()
            out.append([1.0, 0.0] if "apple" in tl else
                       [0.0, 1.0] if "banana" in tl else [0.0, 0.0])
        return out

    monkeypatch.setattr(sem, "embeddings_available", lambda: True)
    monkeypatch.setattr(sem, "embed", stub_embed)

    text = "line about apple pie\n" + ("filler\n" * 10) + "line about banana bread"
    bid = toolaria._store.put(text, "web_extract", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "search", "query": "banana"},
                        session_id="test-s")
    assert "semantic" in r
    assert "banana" in r
    cached = toolaria._store.read_sidecar(bid, "vectors")
    assert cached is not None and cached["model"]


def test_search_truncation_note(plugin, toolaria, monkeypatch):
    """A blob larger than search_max_chunks is flagged as partially indexed."""
    monkeypatch.setattr(toolaria.blobstore._sem, "embeddings_available",
                        lambda: False)
    toolaria._store.cfg["search_max_chunks"] = 3
    big = "\n".join(f"line number {i} of content" for i in range(500))
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    r = toolaria._fetch(args={"id": bid, "mode": "search", "query": "content"},
                        session_id="test-s")
    assert "too large to fully index" in r


def test_outline_built_at_rescue(plugin, toolaria):
    import json
    raw = json.dumps([{"a": i, "label": f"row-{i}"} for i in range(1000)])
    assert len(raw) > 8000  # over the rescue threshold
    handle = toolaria._on_transform(tool_name="web_extract", result=raw)
    assert "outline" in handle
    bid = _parse_blob_id(handle)
    # sidecar exists without a fetch having to build it
    assert toolaria._store.read_sidecar(bid, "outline") is not None


# ═══ Sweep ═══


def test_ttl_sweep(plugin, toolaria):
    """A TTL-expired blob's content is gone (file deleted)."""
    bid = toolaria._store.put("w" * 500, "web_search", session_id="test-s")
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = time.time() - 7200  # 2 hours, ttl is 1
    toolaria._store._save_idx(idx, "test-s")
    toolaria._store.lazy_sweep()
    assert not (toolaria._store.blob_dir / bid).exists()


def test_cross_session_sweep_safety(plugin, toolaria):
    """TTL sweep must not delete blobs referenced by other sessions."""
    bid = toolaria._store.put("shared-blob", "web_search", session_id="session-A")
    bid2 = toolaria._store.put("shared-blob", "web_search", session_id="session-B")
    assert bid == bid2  # dedup

    idx = toolaria._store._load_idx("session-A")
    idx["blobs"][bid]["t"] = time.time() - 7200
    toolaria._store._save_idx(idx, "session-A")

    toolaria._store.lazy_sweep()
    r = toolaria._store.fetch(bid, "stat", session_id="session-A")
    assert "not found" not in r, "Blob deleted while still referenced by session-B"


def test_fetch_refreshes_blob_ttl(plugin, toolaria):
    """Fetching a blob bumps its TTL so active use survives a sweep."""
    bid = toolaria._store.put("w" * 500, "web_search", session_id="test-s")
    # Age it almost to expiry, then fetch (which should refresh t to now).
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = time.time() - 3500  # ttl is 3600
    toolaria._store._save_idx(idx, "test-s")
    toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="test-s")
    toolaria._store.lazy_sweep()
    r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0, "count": 1},
                        session_id="test-s")
    assert "Swept" not in r and "not found" not in r


def test_sidecars_swept_with_blob(plugin, toolaria):
    """Outline sidecars do not outlive their blob."""
    import json, time as _t
    raw = json.dumps([{"a": i} for i in range(1000)])
    bid = toolaria._store.put(raw, "web_extract", session_id="test-s")
    toolaria._store.build_outline(bid, raw)
    assert toolaria._store.read_sidecar(bid, "outline") is not None
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = _t.time() - 7200
    toolaria._store._save_idx(idx, "test-s")
    toolaria._store.lazy_sweep()
    assert toolaria._store.read_sidecar(bid, "outline") is None


def test_sidecar_path_rejects_bad_id(plugin, toolaria):
    assert toolaria._store.sidecar_path("../../etc/passwd", "outline") is None


def test_swept_blob_leaves_tombstone(plugin, toolaria):
    bid = toolaria._store.put("data here", "web_extract", session_id="test-s")
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = time.time() - 7200
    toolaria._store._save_idx(idx, "test-s")
    toolaria._store.lazy_sweep()
    assert not (toolaria._store.blob_dir / bid).exists()
    idx = toolaria._store._load_idx("test-s")
    assert "swept_at" in idx["blobs"][bid]


def test_tombstone_fetch_names_tool_and_advises_rerun(plugin, toolaria):
    bid = toolaria._store.put("data here", "web_extract", session_id="test-s")
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = time.time() - 7200
    toolaria._store._save_idx(idx, "test-s")
    toolaria._store.lazy_sweep()
    r = toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="test-s")
    assert "Swept" in r
    assert "web_extract" in r
    assert "re-run" in r


def test_tombstone_expires_after_tombstone_ttl(plugin, toolaria):
    bid = toolaria._store.put("data", "web_extract", session_id="test-s")
    idx = toolaria._store._load_idx("test-s")
    idx["blobs"][bid] = {"swept_at": time.time() - 800 * 3600,
                         "tool": "web_extract", "size": 4}
    toolaria._store._save_idx(idx, "test-s")
    toolaria._store.lazy_sweep()  # tombstone_ttl is 720h
    idx = toolaria._store._load_idx("test-s")
    assert bid not in idx.get("blobs", {})


def test_tombstone_not_served_cross_session(plugin, toolaria):
    """A tombstone names a tool and size; it must not leak to another session."""
    bid = toolaria._store.put("secret data", "web_extract", session_id="owner")
    idx = toolaria._store._load_idx("owner")
    idx["blobs"][bid]["t"] = time.time() - 7200
    toolaria._store._save_idx(idx, "owner")
    toolaria._store.lazy_sweep()
    # Another session gets a generic session-scoped denial, with no tool name
    # or size from the owner session's tombstone.
    r = toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="intruder")
    assert "web_extract" not in r
    assert "not available in this session" in r


def test_size_sweep_also_tombstones(plugin, toolaria):
    toolaria._store.cfg["max_store_mb"] = 0  # force eviction
    bid = toolaria._store.put("x" * 5000, "web_search", session_id="test-s")
    toolaria._store.lazy_sweep()
    assert not (toolaria._store.blob_dir / bid).exists()
    r = toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="test-s")
    assert "Swept" in r


def test_size_sweep_counts_sidecar_bytes(plugin, toolaria):
    """Blob file bytes fit under the cap, but blob + sidecars exceed it, so
    eviction must still fire. Without counting sidecars the blob would survive."""
    import json
    raw = json.dumps([{"a": i, "label": f"row-{i}"} for i in range(1000)])
    bid = toolaria._store.put(raw, "web_extract", session_id="test-s")
    # Build outline + chunks sidecars; vectors are skipped (embeddings absent).
    toolaria._store.build_outline(bid, raw)
    toolaria._store._chunks(bid, raw)

    blob_bytes = (toolaria._store.blob_dir / bid).stat().st_size
    sidecar_bytes = toolaria._store._sidecar_bytes(bid)
    assert sidecar_bytes > 0
    # Cap sits between blob-only and blob+sidecar totals: blob alone fits.
    cap_bytes = blob_bytes + sidecar_bytes // 2
    toolaria._store.cfg["max_store_mb"] = cap_bytes / (1024 * 1024)

    toolaria._store.lazy_sweep()
    assert not (toolaria._store.blob_dir / bid).exists()
    assert toolaria._store._sidecar_bytes(bid) == 0


def test_session_id_slugged(plugin, toolaria):
    """Hostile session ids cannot escape the sessions directory."""
    toolaria._store.put("data", "web_search", session_id="../../evil")
    files = list(toolaria._store.meta_dir.glob("*.json"))
    assert all(".." not in f.name for f in files)
    parent = toolaria._store.meta_dir.parent.parent
    assert not (parent / "evil.json").exists()


# ═══ Registration ═══


def test_register_creates_tool(plugin):
    fc, _ = plugin
    assert "rescuer_fetch" in fc.tools
    schema = fc.tools["rescuer_fetch"]["schema"]
    assert "parameters" in schema
    assert "id" in schema["parameters"]["required"]


def test_status_command(plugin):
    fc, _ = plugin
    handler = fc.commands.get("rescuer")
    assert handler is not None
    result = handler("")
    assert "blobs:" in result or "Rescuer" in result

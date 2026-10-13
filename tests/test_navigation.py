"""JSON/MCP paging must use one derived view without rewriting raw blobs (#7)."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

import blobstore as bs
from blobstore import BlobStore


MARKER = "UNIQUE_TAIL_MARKER"
MCP_TOOL = "mcp__probe__records"


def _store(tmp_path, **extra):
    cfg = {"store_path": str(tmp_path / "store"), "fetch_max_chars": 4000,
           "full_fetch_max_chars": 12000, "refuse_full_fetch": True,
           "ttl_hours": 72, "max_store_mb": 50,
           "search_chunk_chars": 300, "search_chunk_overlap_lines": 1,
           "search_snippet_chars": 300, "search_top_k": 3,
           "search_max_chunks": 400, "enforcement_enabled": False}
    cfg.update(extra)
    return BlobStore(cfg)


def _json_payload(wrapped, pretty):
    obj = {"records": [{"title": f"alert {i}", "detail": "ordinary text " * 6}
                       for i in range(150)] + [{"title": MARKER, "detail": "end"}]}
    expected = json.dumps(obj, ensure_ascii=False, indent=2)
    inner = expected if pretty else json.dumps(obj, ensure_ascii=False)
    raw = json.dumps({"result": inner}, ensure_ascii=False) if wrapped else inner
    return raw, expected


@pytest.mark.parametrize("wrapped,pretty", [(True, True), (True, False), (False, False)])
def test_json_range_reaches_the_tail_on_the_same_view_as_grep(tmp_path, wrapped, pretty):
    raw, expected = _json_payload(wrapped, pretty)
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL if wrapped else "web_extract", session_id="owner")
    lines = expected.splitlines()
    target = next(i for i, line in enumerate(lines) if MARKER in line)
    page = store.fetch(bid, "range", start=target, count=1, session_id="owner")
    assert f"[lines {target}..{target} of {len(lines)}]" in page
    assert MARKER in page, "range returned the capped prefix instead of the requested tail"
    grep = store.fetch(bid, "grep", pattern=MARKER, session_id="owner")
    assert f"{target}: {lines[target]}" in grep
    chain = store.fetch(bid, "chain", pattern=MARKER, count=1, session_id="owner")
    assert MARKER in chain and "end" in chain
    assert store.fetch(bid, "full", session_id="owner").startswith("Refused:")
    assert (store.blob_dir / bid).read_bytes() == raw.encode()
    assert bid == hashlib.sha256(raw.encode()).hexdigest()[:12]
    assert store._load_idx("owner")["blobs"][bid]["hash"] == hashlib.sha256(raw.encode()).hexdigest()


def test_mcp_text_outline_search_and_range_share_decoded_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: False)
    inner = "\n".join([f"ordinary diagnostic line {i}" for i in range(150)]
                      + [f"ERROR: {MARKER}"])
    raw = json.dumps({"result": inner})
    store = _store(tmp_path, search_snippet_chars=1200)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    outline = store.fetch(bid, "outline", session_id="owner")
    assert "151 lines" in outline and "first L150" in outline
    search = store.fetch(bid, "search", query=MARKER, session_id="owner")
    assert MARKER in search
    match = re.search(r"score [\d.]+  lines (\d+)\.\.(\d+)", search)
    assert match, search
    start, end = map(int, match.groups())
    assert start <= 150 <= end
    followup = store.fetch(bid, "range", start=start, count=end - start + 1,
                           session_id="owner")
    assert MARKER in followup, "search advertised coordinates that do not reach its hit"


def test_rescue_preview_line_total_and_outline_use_the_inner_payload(
        toolaria, fake_ctx_cls, base_cfg, monkeypatch):
    # No host registry in this unit test: explicitly permit the named MCP
    # fixture; the real PluginManager/registry path is tested separately.
    monkeypatch.setattr(toolaria, "_is_rescuable", lambda name: name == MCP_TOOL)
    context = fake_ctx_cls({"toolaria": base_cfg})
    toolaria.register(context)
    inner = "\n".join(["ordinary detail " * 10 for _ in range(160)]
                      + [f"ERROR: {MARKER}"])
    raw = json.dumps({"result": inner})
    handle = context.hooks["transform_tool_result"](
        tool_name=MCP_TOOL, result=raw, session_id="owner")
    assert "lines=161;" in handle, "rescue header counted the one-line wrapper"
    assert MARKER in handle, "preview omitted the decoded tail"
    bid = hashlib.sha256(raw.encode()).hexdigest()[:12]
    outline = context.tools["rescuer_fetch"]["handler"](
        {"id": bid, "mode": "outline"}, session_id="owner")
    assert "161 lines" in outline and "first L160" in outline


@pytest.mark.parametrize("suffix", ["outline", "chunks"])
def test_legacy_navigation_sidecars_are_rebuilt(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: False)
    raw = json.dumps({"result": "ordinary line\nERROR: tail"})
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    if suffix == "outline":
        stale = {"kind": "text", "total_lines": 1,
                 "clusters": [{"signal": "LEGACY_SENTINEL", "count": 1,
                               "first_line": 0, "sample": "old wrapper"}]}
    else:
        stale = {"chunks": [{"start_line": 0, "end_line": 0,
                             "text": "LEGACY_SENTINEL"}], "truncated": False}
    store.write_sidecar(bid, suffix, stale)
    result = store.fetch(bid, suffix if suffix == "outline" else "search",
                         query="tail", session_id="owner")
    assert "LEGACY_SENTINEL" not in result
    rebuilt = store.read_sidecar(bid, suffix)
    assert rebuilt is not None
    assert rebuilt != stale
    assert rebuilt.get("navigation"), "cache does not identify the new navigation representation"


def test_vectors_with_the_same_count_but_old_chunks_are_not_reused(tmp_path, monkeypatch):
    store = _store(tmp_path)
    bid = store.put("first line\nsecond line", "web_extract", session_id="owner")
    chunks = [{"start_line": 0, "end_line": 1, "text": "first line\nsecond line"}]
    store.write_sidecar(bid, "vectors", {"model": "probe", "vectors": [[999.0, 999.0]]})
    calls = []
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: True)

    def embed(texts, model):
        calls.append((texts, model))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(bs._sem, "embed", embed)
    vectors = store._chunk_vectors(bid, chunks, "probe")
    assert calls, "legacy embeddings were reused solely because the chunk counts matched"
    assert vectors == [[1.0, 0.0]]
    assert store._chunk_vectors(bid, chunks, "probe") == vectors
    assert len(calls) == 1, "a valid cache should still be reusable"


def test_chunk_cache_tracks_chunking_parameters(tmp_path):
    store = _store(tmp_path)
    raw = "\n".join(f"diagnostic line {i}" for i in range(60))
    bid = store.put(raw, "web_extract", session_id="owner")
    first, _ = store._chunks(bid, raw)
    store.cfg["search_chunk_chars"] = 100
    second, _ = store._chunks(bid, raw)
    assert len(second) > len(first), "chunks were reused after their parameters changed"


@pytest.mark.parametrize("inner,tool", [
    ("plain\ntext\n", "web_extract"),
    ("<html>\n<h1>Title</h1>\n</html>", "web_extract"),
    (json.dumps({"value": "nested\ntext"}, indent=2), "web_extract"),
    ("{not valid json", MCP_TOOL),
])
def test_existing_multiline_and_invalid_payloads_keep_their_line_coordinates(tmp_path, inner, tool):
    store = _store(tmp_path, full_fetch_max_chars=50000)
    bid = store.put(inner, tool, session_id="owner")
    result = store.fetch(bid, "range", count=20, session_id="owner")
    assert result.split("\n", 1)[1] == "\n".join(inner.splitlines())
    assert store.fetch(bid, "full", session_id="owner") == inner


@pytest.mark.parametrize("tool,extra", [("web_extract", False), (MCP_TOOL, True)])
def test_result_strings_are_not_unwrapped_for_arbitrary_tools_or_extra_fields(tmp_path, tool, extra):
    obj: dict[str, object] = {"result": "first\nsecond"}
    if extra:
        obj["structuredContent"] = {"must_survive": True}
    raw = json.dumps(obj)
    store = _store(tmp_path, full_fetch_max_chars=50000)
    bid = store.put(raw, tool, session_id="owner")
    page = store.fetch(bid, "range", count=20, session_id="owner")
    assert '"result"' in page and r"first\nsecond" in page
    if extra:
        assert '"must_survive"' in page
    assert store.fetch(bid, "full", session_id="owner") == raw


def test_full_and_passref_still_expand_the_original_mcp_wrapper(tmp_path):
    from passref import expand_value
    raw = json.dumps({"result": "first\nsecond"}, ensure_ascii=False)
    store = _store(tmp_path, full_fetch_max_chars=50000)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    assert store.fetch(bid, "full", session_id="owner") == raw
    assert store.blob_text(bid) == raw
    expanded, denied = expand_value({"text": f"tla:{bid}"}, store,
                                    {"passref_enabled": True}, {},
                                    session_id="owner", tool_name="summarise_report")
    assert not denied
    assert expanded["text"] == raw


def test_navigation_does_not_bypass_integrity_or_cross_session_checks(tmp_path):
    raw = json.dumps({"result": "first\nsecond"})
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    assert "not available" in store.fetch(bid, "range", session_id="other")
    (store.blob_dir / bid).write_text(json.dumps({"result": "tampered\nsecond"}))
    assert "integrity check failed" in store.fetch(bid, "range", session_id="owner")


@pytest.mark.parametrize("mode", ["range", "grep", "search", "chain", "outline"])
def test_encrypted_mcp_navigation_remains_masked_and_never_creates_sidecars(
        tmp_path, monkeypatch, mode):
    from cryptography.fernet import Fernet
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: False)
    key_path = tmp_path / "key"
    key_path.write_bytes(Fernet.generate_key())
    key_path.chmod(0o600)
    secret = "sk-ABCDEFGHIJKLMNOP123456789"
    inner = f"ordinary line\nERROR: secret {secret}\nlast line"
    raw = json.dumps({"result": inner})
    store = _store(tmp_path, toolaria_key_file=str(key_path), enforcement_enabled=True,
                   credential_slice_total_max_chars=100000)
    bid = store.put(raw, MCP_TOOL, session_id="owner", label="credential")
    result = store.fetch(bid, mode, count=20, pattern="ERROR", query="ERROR",
                         session_id="owner")
    assert "sk-" not in result, "derived navigation exposed a credential-shaped value"
    assert not list(store.sidecar_dir.glob(f"{bid}.*"))
    assert store.blob_text(bid) == raw


@pytest.mark.parametrize("raw", [
    r'{"key":1,"key":2,"large":1e999,"negative_zero":-0,"precise":0.1234567890123456789}',
    r'{"empty":{},"array":[],"nested":[{"items":[1,2]},{}]}',
    r'{"escaped":"quotes: \" and backslash: \\ and comma, colon: braces {}[]","unicode":"\u2603"}',
    '{"name":"日本語 café","flag":true,"value":null}',
    r'{"result":"duplicate one","result":"duplicate two"}',
    r'{"result":"\ud800"}',
])
def test_pretty_navigation_preserves_every_json_token(tmp_path, raw):
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    view = store.navigation_view(bid, raw, "owner")
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|[^\s]')
    assert tokens.findall(view.text) == tokens.findall(raw)
    assert view.key
    assert store.fetch(bid, "full", session_id="owner") == raw


def test_nested_mcp_result_is_unwrapped_only_once(tmp_path):
    inner = json.dumps({"result": "first\nsecond"})
    raw = json.dumps({"result": inner})
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL, session_id="owner")
    result = store.fetch(bid, "range", count=20, session_id="owner")
    assert '"result"' in result and r"first\nsecond" in result
    outline = store.fetch(bid, "outline", session_id="owner")
    assert "JSON outline" in outline and "result: str" in outline


def test_cache_identity_does_not_mix_mcp_and_non_mcp_owners(tmp_path, monkeypatch):
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: False)
    raw = json.dumps({"result": "ordinary line\nERROR: final line"})
    store = _store(tmp_path)
    bid = store.put(raw, MCP_TOOL, session_id="mcp-owner")
    assert store.put(raw, "web_extract", session_id="web-owner") == bid
    for sid in ["mcp-owner", "web-owner", "mcp-owner"]:
        outline = store.fetch(bid, "outline", session_id=sid)
        search = store.fetch(bid, "search", query="final", session_id=sid)
        if sid == "mcp-owner":
            assert "text outline" in outline and "first L1" in outline
            assert '"result"' not in search
        else:
            assert "JSON outline" in outline and "result: str" in outline
            assert '"result"' in search


def test_encryption_upgrade_during_embedding_cannot_publish_plaintext_vectors(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    key_path = tmp_path / "key"
    key_path.write_bytes(Fernet.generate_key())
    key_path.chmod(0o600)
    store = _store(tmp_path, toolaria_key_file=str(key_path))
    raw = json.dumps({"result": "ordinary\nprivate material"})
    bid = store.put(raw, MCP_TOOL, session_id="public", label="public")
    chunks, _ = store._chunks(bid, raw)
    other = BlobStore(store.cfg)
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: True)

    def upgrade_while_embedding(texts, model):
        # Embed runs outside the store lock. Simulate a second process
        # upgrading the same bytes before the pending cache publication.
        assert other.put(raw, MCP_TOOL, session_id="private", label="credential") == bid
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(bs._sem, "embed", upgrade_while_embedding)
    assert store._chunk_vectors(bid, chunks, "probe")
    assert not list(store.sidecar_dir.glob(f"{bid}.*")), "stale encryption check allowed a plaintext cache write"


def test_deep_json_does_not_expand_quadratically_during_navigation(tmp_path):
    raw = "[" * 128 + "0" + "]" * 128
    store = _store(tmp_path)
    bid = store.put(raw, "web_extract", session_id="owner")
    view = store.navigation_view(bid, raw, "owner")
    assert view.text == raw, "deep nesting amplified an untrusted payload before response caps"


def test_cache_publication_after_blob_removal_is_suppressed(tmp_path):
    store = _store(tmp_path)
    bid = store.put("temporary content", "web_extract", session_id="owner")
    (store.blob_dir / bid).unlink()
    store.write_sidecar(bid, "vectors", {"vectors": [[1.0, 0.0]]})
    assert not list(store.sidecar_dir.glob(f"{bid}.*"))


@pytest.mark.parametrize("mode", ["range", "grep", "chain", "search", "outline"])
def test_long_json_string_cannot_hide_a_secret_past_the_old_mask_window(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(bs._sem, "embeddings_available", lambda: False)
    secret = "sk-ABCDEFGHIJKLMNOP123456789"
    raw = json.dumps({"result": json.dumps({
        "message": "ALERT: " + "x" * 2500 + secret})})
    store = _store(tmp_path, enforcement_enabled=True,
                   credential_slice_total_max_chars=100000,
                   search_snippet_chars=4000, search_chunk_chars=4000)
    bid = store.put(raw, MCP_TOOL, session_id="owner", label="credential")
    result = store.fetch(bid, mode, count=20, pattern="ALERT", query="ALERT",
                         session_id="owner")
    assert "sk-" not in result, "navigation masking missed a secret late in a long value"

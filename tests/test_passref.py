"""Tests for Toolaria v2: pass-by-reference (tla: token expansion)."""


def _mw(plugin, toolaria):
    """The registered tool_request middleware callback."""
    fc, _ = plugin
    return fc.middleware["tool_request"][0]


def test_token_expands_to_full_content(plugin, toolaria):
    big = "FULL CONTENT " * 1000
    bid = toolaria._store.put(big, "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="publish", args={"body": f"tla:{bid}"})
    assert out is not None
    assert out["args"]["body"] == big


def test_roundtrip_tool_to_tool_without_context(plugin, toolaria):
    """The headline guarantee: a rescued result reaches a second tool in full,
    while what the model emitted was only the short token."""
    page = "x" * 50000  # far larger than any context preview
    bid = toolaria._store.put(page, "web_extract", session_id="test-s")
    model_emitted = {"content": f"tla:{bid}", "title": "summary"}
    # model_emitted is what the model wrote: small, no page content in it
    assert len(str(model_emitted)) < 200
    mw = _mw(plugin, toolaria)
    delivered = mw(tool_name="summarise", args=model_emitted)["args"]
    # the downstream tool receives the whole page
    assert delivered["content"] == page
    assert delivered["title"] == "summary"


def test_token_in_nested_args(plugin, toolaria):
    bid = toolaria._store.put("NESTED", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"a": {"b": [f"tla:{bid}"]}})
    assert out["args"]["a"]["b"][0] == "NESTED"


def test_no_token_returns_none(plugin, toolaria):
    mw = _mw(plugin, toolaria)
    assert mw(tool_name="t", args={"x": "plain text"}) is None


def test_missing_blob_token_becomes_marker(plugin, toolaria):
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": "tla:000000000000"})
    assert "unavailable" in out["args"]["x"]


def test_oversized_expansion_truncates_with_marker(plugin, toolaria):
    toolaria._cfg["passref_max_chars"] = 100
    bid = toolaria._store.put("y" * 5000, "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": f"tla:{bid}"})
    assert "truncated" in out["args"]["x"]
    assert len(out["args"]["x"]) < 400


def test_excluded_tool_not_expanded(plugin, toolaria):
    bid = toolaria._store.put("data", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    # rescuer_fetch is in the unconditional excludes
    assert mw(tool_name="rescuer_fetch", args={"x": f"tla:{bid}"}) is None


def test_token_inline_within_text(plugin, toolaria):
    bid = toolaria._store.put("WORLD", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": f"hello tla:{bid} end"})
    assert out["args"]["x"] == "hello WORLD end"


def test_disabled_passref_is_noop(plugin, toolaria):
    toolaria._cfg["passref_enabled"] = False
    bid = toolaria._store.put("data", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    assert mw(tool_name="t", args={"x": f"tla:{bid}"}) is None


def test_middleware_registered(plugin, toolaria):
    fc, _ = plugin
    assert "tool_request" in fc.middleware
    assert len(fc.middleware["tool_request"]) == 1


def test_exec_sink_denied_by_default(plugin, toolaria):
    """Exec/exfil sinks do not get silent expansion under the default policy."""
    bid = toolaria._store.put("rm -rf payload", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    for sink in ("shell", "run_shell", "http_post", "write_file"):
        assert mw(tool_name=sink, args={"x": f"tla:{bid}"}) is None


def test_allowlist_restricts_expansion(plugin, toolaria):
    toolaria._cfg["passref_allowed_tools"] = ["summarise"]
    bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    assert mw(tool_name="summarise", args={"x": f"tla:{bid}"})["args"]["x"] == "DATA"
    # a tool not on the allowlist is skipped even though it is not a sink
    assert mw(tool_name="publish", args={"x": f"tla:{bid}"}) is None


def test_same_session_token_expands(plugin, toolaria):
    """With a forwarded session_id, a blob the session owns expands in full."""
    bid = toolaria._store.put("OWNED", "web_extract", session_id="sess-a")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": f"tla:{bid}"}, session_id="sess-a")
    assert out["args"]["x"] == "OWNED"


def test_cross_session_token_denied(plugin, toolaria):
    """A token for a blob the calling session does not reference is refused,
    even though the bytes exist (another session put them)."""
    bid = toolaria._store.put("SECRET", "web_extract", session_id="owner")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": f"tla:{bid}"}, session_id="intruder")
    assert "not available in this session" in out["args"]["x"]
    assert "SECRET" not in out["args"]["x"]


def test_empty_session_id_still_expands(plugin, toolaria):
    """Back-compat: no forwarded session_id keeps the global read so vanilla
    single-session Hermes still works."""
    bid = toolaria._store.put("GLOBAL", "web_extract", session_id="some-s")
    mw = _mw(plugin, toolaria)
    out = mw(tool_name="t", args={"x": f"tla:{bid}"})
    assert out["args"]["x"] == "GLOBAL"


def test_total_expansion_budget(plugin, toolaria):
    toolaria._cfg["passref_total_max_chars"] = 500
    bid = toolaria._store.put("z" * 800, "web_extract", session_id="test-s")
    mw = _mw(plugin, toolaria)
    # first 800-char expansion exhausts the 500 total budget; the second is blocked
    out = mw(tool_name="t", args={"a": f"tla:{bid}", "b": f"tla:{bid}"})
    assert "z" * 800 in out["args"]["a"]
    assert "budget" in out["args"]["b"]

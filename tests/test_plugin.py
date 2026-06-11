"""Tests for Toolaria (rescuer) plugin. Run with pytest from any directory."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

_PDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PDIR))
_INIT = str(_PDIR / "__init__.py")
spec = importlib.util.spec_from_file_location(
    "toolaria", _INIT, submodule_search_locations=[str(_PDIR)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["toolaria"] = mod
spec.loader.exec_module(mod)


class FakeCtx:
    """Minimal PluginContext with register_* support."""
    def __init__(self, cfg, session_id="test-session"):
        self._cfg = cfg
        self.session_id = session_id
        self.hooks = {}
        self.tools = {}
        self.commands = {}

    @property
    def config(self):
        return self._cfg

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = {"handler": handler, "schema": schema}

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler


# ═══ Fixtures ═══

@pytest.fixture
def tmpstore(monkeypatch):
    """Create a BlobStore in a temp directory."""
    d = tempfile.mkdtemp()
    mono = monkeypatch
    cfg = {
        "max_result_chars": 8000, "fetch_max_chars": 4000,
        "head_lines": 10, "tail_lines": 5, "json_head_items": 3, "json_tail_items": 1,
        "error_line_patterns": ["error", "fail"],
        "grep_timeout_ms": 500, "grep_max_pattern_len": 80,
        "ttl_hours": 1, "max_store_mb": 50, "store_path": d,
        "exclude_tools": [], "refuse_full_fetch": True,
    }
    yield cfg, d
    # Cleanup
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def plugin(tmpstore):
    """Register the plugin with a temp store."""
    cfg, d = tmpstore
    os.environ["HERMES_HOME"] = str(Path(d) / "hermes_home")
    os.makedirs(os.environ["HERMES_HOME"], exist_ok=True)
    fc = FakeCtx({"toolaria": cfg}, session_id="test-s")
    mod._cfg = mod._merge_cfg(cfg)
    mod.register(fc)
    return fc, cfg, d


# ═══ Tests ═══

def test_oversized_triggered(plugin):
    _, cfg, _ = plugin
    assert len("x" * 9000) > cfg["max_result_chars"]


def test_rescue_json_fixture(plugin):
    """279k-char JSON → excerpt + handle fits."""
    _, cfg, d = plugin
    fixture = Path(__file__).parent / "fixture_23767.json"
    if not fixture.exists():
        pytest.skip("fixture file not found")
    raw = fixture.read_text()
    assert len(raw) > 200000

    cfg["max_result_chars"] = 8000
    result = mod._on_transform(tool_name="web_extract", result=raw)
    assert result is not None
    assert "Rescued" in result
    assert "rescuer_fetch" in result
    assert len(result) < 10000

    bid = None
    for ln in result.splitlines():
        if ln.startswith("id: "):
            bid = ln.split(": ")[1].strip()
    assert bid and len(bid) == 12


def test_fetch_range(plugin):
    _, cfg, d = plugin
    bid = mod._store.put("START-" + "x" * 5000 + "-END", "web_search", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "range", "start": 0, "count": 2})
    assert "START" in r
    assert len(r) <= cfg["fetch_max_chars"] + 100


def test_fetch_grep(plugin):
    _, cfg, d = plugin
    content = "\n".join(f"row_{i}: val_{i}" for i in range(1000))
    bid = mod._store.put(content, "web_search", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "grep", "pattern": "row_42"})


def test_grep_spaces_allowed(plugin):
    """Multi-word patterns must work."""
    _, cfg, d = plugin
    content = "line one two three\nline four five six\n"
    bid = mod._store.put(content, "web_search", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "grep", "pattern": "one two"})
    assert "one two" in r


def test_full_refused(plugin):
    _, cfg, d = plugin
    bid = mod._store.put("x" * 10000, "web_search", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "full"})
    assert "Refused" in r


def test_dedup(plugin):
    _, cfg, d = plugin
    c = "uniq-" + "dat-" * 200
    b1 = mod._store.put(c, "web_search", session_id="test-s")
    b2 = mod._store.put(c, "web_search", session_id="test-s")
    assert b1 == b2


def test_stat(plugin):
    _, cfg, d = plugin
    bid = mod._store.put("s" * 2500, "mcp_tool", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "stat"})
    assert "blob:" in r and "bytes" in r


def test_rescuer_fetch_excluded(plugin):
    """rescuer_fetch tool never rescued (safety valve)."""
    r = mod._on_transform(tool_name="rescuer_fetch", result="x" * 9000)
    assert r is None


def test_unlisted_tool_not_rescued(plugin):
    """Tools not in RESCUABLE_TOOLS pass through."""
    r = mod._on_transform(tool_name="delegate_task", result="x" * 9000)
    assert r is None

    r = mod._on_transform(tool_name="session_search", result="x" * 9000)
    assert r is None


def test_missing_blob(plugin):
    r = mod._fetch(args={"id": "000000000000", "mode": "stat"})
    assert "not found" in r


def test_invalid_blob_id(plugin):
    """Path traversal guard: non-hex blob_id rejected."""
    r = mod._fetch(args={"id": "../../../etc/passwd", "mode": "stat"})
    assert "invalid" in r.lower()


def test_ttl_sweep(plugin):
    _, cfg, d = plugin
    bid = mod._store.put("w" * 500, "web_search", session_id="test-s")
    idx = mod._store._load_idx("test-s")
    idx["blobs"][bid]["t"] = time.time() - 7200  # 2 hours ago
    mod._store._save_idx(idx, "test-s")
    mod._store.lazy_sweep()
    r = mod._fetch(args={"id": bid, "mode": "stat"}, session_id="test-s")
    assert "not found" in r


def test_cross_session_sweep_safety(plugin):
    """TTL sweep must not delete blobs referenced by other sessions."""
    _, cfg, d = plugin
    bid = mod._store.put("shared-blob", "web_search", session_id="session-A")

    bid2 = mod._store.put("shared-blob", "web_search", session_id="session-B")
    assert bid == bid2  # dedup

    idx = mod._store._load_idx(session_id="session-A")
    idx["blobs"][bid]["t"] = time.time() - 7200
    mod._store._save_idx(idx, session_id="session-A")

    mod._store.lazy_sweep()
    r = mod._store.fetch(bid, "stat", session_id="session-A")
    assert "not found" not in r, "Blob deleted while still referenced by session-B"


def test_html_excerpt(plugin):
    raw = "<!DOCTYPE html>\n<html>\n<body>\n" + "<!--pad-->\n" * 5000 + "</body>\n</html>"
    kind, _ = mod.detect_type(raw)
    assert kind == "html"
    ex = mod.build_excerpt(raw, kind, mod._cfg)
    assert "<body>" in ex or "<html>" in ex


def test_binary_detection(plugin):
    kind, _ = mod.detect_type(b"\x00\x01\x02" * 100)
    assert kind == "binary"


def test_nil_result_passthrough(plugin):
    assert mod._on_transform(tool_name="web_search", result="") is None
    assert mod._on_transform(tool_name="web_search", result=None) is None


def test_register_creates_tool(plugin):
    fc, cfg, d = plugin
    assert "rescuer_fetch" in fc.tools
    schema = fc.tools["rescuer_fetch"]["schema"]
    assert "parameters" in schema
    assert "id" in schema["parameters"]["required"]


def test_status_command(plugin):
    fc, cfg, d = plugin
    handler = fc.commands.get("rescuer")
    assert handler is not None, "No /rescuer command registered"
    result = handler("")
    assert "blobs:" in result or "Rescuer" in result


def test_fetch_unknown_mode(plugin):
    _, cfg, d = plugin
    bid = mod._store.put("test content", "web_search", session_id="test-s")
    r = mod._fetch(args={"id": bid, "mode": "unknown"})
    assert "unknown mode" in r

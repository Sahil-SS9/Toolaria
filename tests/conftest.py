"""Shared fixtures for Toolaria tests.

The plugin module is loaded once, here, under the name "toolaria". Every
test runs with store_path pinned to a per-test tmp_path and module state
reset, so no test can touch the real ~/.hermes/toolaria.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_PDIR = Path(__file__).resolve().parent.parent
if str(_PDIR) not in sys.path:
    sys.path.insert(0, str(_PDIR))

_spec = importlib.util.spec_from_file_location(
    "toolaria", _PDIR / "__init__.py", submodule_search_locations=[str(_PDIR)],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["toolaria"] = _mod
_spec.loader.exec_module(_mod)


class FakeCtx:
    """Minimal PluginContext with register_* support."""
    def __init__(self, cfg, session_id="test-s"):
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


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Reset module state and keep HERMES_HOME away from the real one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    _mod._store = None
    _mod._cfg = {}
    yield


@pytest.fixture
def toolaria():
    return _mod


@pytest.fixture
def fake_ctx_cls():
    return FakeCtx


@pytest.fixture
def base_cfg(tmp_path):
    return {
        "max_result_chars": 8000, "fetch_max_chars": 4000,
        "full_fetch_max_chars": 50000, "excerpt_max_chars": 8000,
        "head_lines": 10, "tail_lines": 5,
        "json_head_items": 3, "json_tail_items": 1,
        "error_line_patterns": ["error", "fail"],
        "grep_timeout_ms": 500, "grep_max_pattern_len": 80,
        "grep_max_line_len": 2000,
        "ttl_hours": 1, "tombstone_ttl_hours": 720, "max_store_mb": 50,
        "store_path": str(tmp_path / "store"),
        "exclude_tools": [], "refuse_full_fetch": True,
        "search_chunk_chars": 300, "search_chunk_overlap_lines": 1,
        "search_max_chunks": 400, "search_top_k": 3,
        "search_snippet_chars": 300,
        "embedding_model": "all-MiniLM-L6-v2",
    }


@pytest.fixture
def plugin(toolaria, base_cfg, fake_ctx_cls):
    """Register the plugin against a FakeCtx with a tmp store."""
    fc = fake_ctx_cls({"toolaria": base_cfg})
    toolaria.register(fc)
    return fc, base_cfg

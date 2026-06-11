"""Toolaria — Rescue oversized tool results before they flood context.

Stores full results to disk via SHA256-addressed blob store.
Returns excerpt + handle block.  Provides rescuer_fetch tool for retrieval.

V1 catchment: MCP and web tool results only (terminal/file-read outputs are
already truncated by tool_output_limits before any hook fires).
Explicit allow-list enforced — only rescued tools get intercepted.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

try:
    from .blobstore import BlobStore
    from .excerpt import detect_type, build_excerpt
except ImportError:
    from blobstore import BlobStore  # type: ignore[no-redef]
    from excerpt import detect_type, build_excerpt  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_store: BlobStore | None = None
_cfg: dict = {}

# Tools whose results may exceed context — the only tools rescued.
# Terminal output and file reads are truncated by tools/tool_output_limits.py
# before any hook fires.  delegate_task, session_search, etc. are explicitly
# excluded as their results are bounded by design.
#
# Static set covers built-in tools; MCP tools are detected dynamically via
# registry.get_toolset_for_tool() — any tool registered under an 'mcp-*'
# toolset is automatically rescuer-eligible.
_RESCUABLE_TOOLS: set[str] = {
    "web_extract",
    "web_search",
    "browser_navigate",
    "browser_snapshot",
    "browser_console",
    "browser_get_images",
}


# Hardcoded safety net — tools that must never be intercepted, regardless of
# registry availability or config state. Checked unconditionally in _on_transform
# so even a fail-open _is_rescuable() + empty config can't touch these.
_UNCONDITIONAL_EXCLUDES: frozenset[str] = frozenset({
    "rescuer_fetch", "delegate_task", "session_search",
    "cronjob", "skill_view", "skill_manage", "skill_request",
    "kanban_create", "open_kanban", "clarify", "memory",
})


def _is_rescuable(tool_name: str) -> bool:
    """Return True if this tool should be rescued.  MCP tools are identified
    via their dynamic 'mcp-{server}' toolset prefix; built-in web/browser
    tools are matched by static name set.  Fails open (True) if the registry
    import breaks."""
    if tool_name in _RESCUABLE_TOOLS:
        return True
    try:
        from tools.registry import registry
        toolset = registry.get_toolset_for_tool(tool_name)
        if toolset and toolset.startswith("mcp-"):
            return True
    except Exception:
        return True  # fail open — safer to rescue than to flood context
    return False

_BLOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _safe_cfg(ctx) -> dict:
    """Read plugin config from PluginContext or config.yaml fallback."""
    try:
        cfg = ctx.config.get("toolaria")
        if cfg:
            return dict(cfg)
    except Exception:
        pass
    try:
        import yaml
        hp = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        cf = hp / "config.yaml"
        if cf.exists():
            raw = yaml.safe_load(cf.read_text())
            return dict(raw.get("toolaria", {}))
    except Exception:
        pass
    return {}

# Load plugin-local config.yaml as defaults layer (overridden by main config)
_CWD = Path(__file__).resolve().parent
_LOCAL_CFG = _CWD / "config.yaml"


def _load_defaults() -> dict:
    """Load plugin-local config.yaml as defaults layer."""
    try:
        import yaml
        raw = yaml.safe_load(_LOCAL_CFG.read_text())
        return dict(raw.get("toolaria", {}))
    except Exception:
        return {}


def _merge_cfg(user_cfg: dict) -> dict:
    """Merge user config over plugin defaults."""
    defaults = _load_defaults()
    defaults.update(user_cfg)
    return defaults


def _session_id(ctx) -> str:
    try:
        return ctx.session_id or "unknown"
    except Exception:
        return os.environ.get("HERMES_SESSION_ID", "unknown")


def register(ctx) -> None:
    global _store, _cfg
    _cfg = _merge_cfg(_safe_cfg(ctx))
    # Add rescuer_fetch + hardcoded safety excludes to stop fail-open rescuing
    # tools that must never be intercepted (delegate_task, session_search, etc.).
    # These protect against _is_rescuable() returning True when registry import
    # breaks — the exclude check runs after _is_rescuable, so pre-populating
    # guarantee-unconditional default excludes makes the fail-open safe.
    _DEFAULT_EXCLUDES = {
        "rescuer_fetch", "delegate_task", "session_search",
        "cronjob", "skill_view", "skill_manage", "skill_request",
        "kanban_create", "open_kanban", "clarify", "memory",
    }
    _cfg.setdefault("exclude_tools", [])
    for t in _DEFAULT_EXCLUDES:
        if t not in _cfg["exclude_tools"]:
            _cfg["exclude_tools"].append(t)
    try:
        sid = _session_id(ctx)
        _store = BlobStore(_cfg, sid)
        _store.lazy_sweep()
    except Exception as e:
        logger.warning("toolaria: blob init failed: %s", e)
        _store = None

    ctx.register_hook("transform_tool_result", _on_transform)
    ctx.register_hook("on_session_start", _on_start)
    ctx.register_hook("on_session_end", _on_end)

    ctx.register_tool(
        name="rescuer_fetch",
        toolset="rescuer",
        description=(
            "Fetch slices of a rescued oversized tool result. "
            "Modes: range(start,count) | grep(pattern) | stat | full"
        ),
        handler=_fetch,
        schema={
            "name": "rescuer_fetch",
            "description": "Retrieve slices of a rescued tool result blob",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Blob ID from the rescue handle block",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["range", "grep", "stat", "full"],
                        "description": "Retrieval mode (default: stat)",
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start line for range mode (default: 0)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Lines to return in range mode (default: 20)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex for grep mode",
                    },
                },
                "required": ["id"],
            },
        },
    )

    ctx.register_command(
        name="rescuer",
        handler=_status_cmd,
        description="Show Toolaria status: blob count, total size, sessions",
    )


# ── hooks ────────────────────────────────────────────────────────────────


def _on_transform(
    tool_name: str = "",
    result: str = "",
    args: dict | None = None,
    session_id: str = "",
    **kwargs,
):
    """Replace oversized tool results with excerpt + rescue handle."""
    # Only rescue MCP/web tools (dynamic allow-list with toolset check)
    if not _is_rescuable(tool_name):
        return None
    # Hardcoded unconditional excludes — must never be intercepted regardless
    # of registry availability or config state (fail-open safety).
    if tool_name in _UNCONDITIONAL_EXCLUDES:
        return None
    if tool_name in _cfg.get("exclude_tools", []):
        return None
    if not result or not isinstance(result, str):
        return None

    try:
        if len(result) > _cfg.get("max_result_chars", 8000):
            return _rescue(result, tool_name, session_id=session_id)
    except Exception:
        pass
    return None


def _rescue(result: str, tool_name: str, session_id: str = "") -> str:
    blob_id = _store.put(result, tool_name, session_id=session_id) if _store else "NO_STORE"
    kind, meta = detect_type(result)
    excerpt = build_excerpt(result, kind, _cfg)
    handle = (
        "[Rescued: full result preserved]\n"
        f"id: {blob_id}\n"
        f"original: {len(result):,} chars, type: {kind} {meta}\n"
        f"fetch: rescuer_fetch(id=\"{blob_id}\", mode=\"range\", start=0, count=20)\n"
        f"       rescuer_fetch(id=\"{blob_id}\", mode=\"grep\", pattern=\"<term>\")"
    )
    return excerpt + "\n\n" + handle


def _on_start(session_id="", **kwargs):
    if _store:
        try:
            _store.lazy_sweep()
        except Exception:
            pass


def _on_end(**kwargs):
    if _store:
        try:
            _store.lazy_sweep()
        except Exception:
            pass


# ── tool handler ─────────────────────────────────────────────────────────


def _fetch(args: dict | None = None, **kwargs) -> str:
    """Handle rescuer_fetch tool calls — dispatched by plugin tool registry."""
    if args is None:
        args = {}
    if not _store:
        return "Error: rescuer store not initialised"

    bid = args.get("id", "")
    mode = args.get("mode", "stat")
    start = int(args.get("start", 0))
    count = int(args.get("count", 20))
    pattern = args.get("pattern", "")
    cap = _cfg.get("fetch_max_chars", 4000)

    if mode == "full" and _cfg.get("refuse_full_fetch", True):
        return (
            "Refused: original exceeds safe inline size. "
            "Use mode='range' or mode='grep' instead."
        )

    if not _BLOB_ID_RE.match(bid):
        return f"Error: invalid blob id '{bid}' — expected 12 hex chars"

    return _store.fetch(bid, mode, start=start, count=count,
                        pattern=pattern, cap=cap)


# ── slash command ────────────────────────────────────────────────────────


def _status_cmd(raw_args: str = "") -> str:
    """Handle /rescuer slash command."""
    if not _store:
        return "Rescuer store not initialised"
    bp = _store.blob_dir
    mp = _store.meta_dir
    blobs = sorted(bp.glob("*")) if bp.exists() else []
    total = sum(b.stat().st_size for b in blobs if b.is_file())
    sessions = sorted(mp.glob("*.json")) if mp.exists() else []
    return (
        f"Toolaria status:\n"
        f"  blobs: {len(blobs)}\n"
        f"  size: {total:,.0f} bytes ({total/1024/1024:.1f} MB)\n"
        f"  sessions: {len(sessions)}\n"
        f"  store: {bp}\n"
    )

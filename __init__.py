"""Toolaria: rescue oversized tool results before they flood context.

Stores full results to disk via SHA256-addressed blob store.
Returns excerpt + handle block.  Provides rescuer_fetch tool for retrieval.

V1 catchment: MCP and web tool results only (terminal/file-read outputs are
already truncated by tool_output_limits before any hook fires).
Explicit allow-list enforced; only rescued tools get intercepted.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

try:
    from .blobstore import BlobStore, _BLOB_ID_RE
    from .excerpt import detect_type, build_excerpt
    from .passref import make_middleware as _make_passref_mw
except ImportError:
    from blobstore import BlobStore, _BLOB_ID_RE  # type: ignore[no-redef]
    from excerpt import detect_type, build_excerpt  # type: ignore[no-redef]
    from passref import make_middleware as _make_passref_mw  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Toolset under which rescuer_fetch registers, and which is marked ambient so
# the tool is reachable in every session. One constant keeps the registration
# and the ambient marking from drifting apart.
_RESCUER_TOOLSET = "rescuer"

_store: BlobStore | None = None
_cfg: dict = {}

# Tools whose results may exceed context: the only built-ins rescued.
# MCP tools are detected dynamically via the registry toolset prefix.
_RESCUABLE_TOOLS: set[str] = {
    "web_extract",
    "web_search",
    "browser_navigate",
    "browser_snapshot",
    "browser_console",
    "browser_get_images",
}

# Single source of truth for tools that must never be intercepted.
# Enforced unconditionally in _on_transform, so _is_rescuable failing open
# (registry import broken) still cannot touch these.
_UNCONDITIONAL_EXCLUDES: frozenset[str] = frozenset({
    "rescuer_fetch", "delegate_task", "session_search",
    "cronjob", "skill_view", "skill_manage", "skill_request",
    "kanban_create", "open_kanban", "clarify", "memory",
})


def _is_rescuable(tool_name: str) -> bool:
    """True if this tool should be rescued.  MCP tools are identified via
    their 'mcp-{server}' toolset prefix; built-in web/browser tools by the
    static set.  Fails open (True) if the registry import breaks; the
    unconditional excludes in _on_transform bound the blast radius."""
    if tool_name in _RESCUABLE_TOOLS:
        return True
    try:
        from tools.registry import registry
        toolset = registry.get_toolset_for_tool(tool_name)
        if toolset and toolset.startswith("mcp-"):
            return True
    except Exception:
        return True  # fail open: safer to rescue than to flood context
    return False


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


def register(ctx) -> None:
    global _store, _cfg
    _cfg = _merge_cfg(_safe_cfg(ctx))
    # Copy rather than mutate the caller's list in place.
    excludes = list(_cfg.get("exclude_tools", []))
    for t in _UNCONDITIONAL_EXCLUDES:
        if t not in excludes:
            excludes.append(t)
    _cfg["exclude_tools"] = excludes
    try:
        _store = BlobStore(_cfg)
        _store.lazy_sweep()
    except Exception as e:
        logger.warning("toolaria: blob store init failed, rescuing disabled: %s", e)
        _store = None

    ctx.register_hook("transform_tool_result", _on_transform)
    ctx.register_hook("on_session_start", _on_start)
    ctx.register_hook("on_session_end", _on_end)

    # Pass-by-reference: expand tla:<id> tokens in downstream tool args into
    # full blob content before the tool runs, so a rescued result can flow
    # tool to tool without ever re-entering the model's context. Always
    # registered when the host supports middleware; passref_enabled is the
    # single live on/off check, inside the callback.
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware(
            "tool_request",
            _make_passref_mw(lambda: _store, _cfg, _UNCONDITIONAL_EXCLUDES),
        )

    ctx.register_tool(
        name="rescuer_fetch",
        toolset=_RESCUER_TOOLSET,
        description=(
            "Fetch slices of a rescued oversized tool result. Modes: "
            "outline | search(query) | range(start,count) | grep(pattern) | "
            "stat | full"
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
                        "enum": ["outline", "search", "range", "grep",
                                 "stat", "full"],
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
                    "query": {
                        "type": "string",
                        "description": "Natural-language query for search mode",
                    },
                },
                "required": ["id"],
            },
        },
    )

    # Availability symmetry: the rescue hook above fires UNGATED in every
    # session, so its inverse (rescuer_fetch) must be reachable in every
    # session too, or a rescued result becomes an unredeemable handle. Mark
    # the rescuer toolset ambient so the host always surfaces rescuer_fetch
    # regardless of a session's enabled_toolsets scope. Safe to expose
    # broadly because fetch() stays session-scoped at the data layer (a
    # session can only read blobs it rescued). On a host without ambient
    # support, fail LOUD rather than silently emit dead handles in
    # toolset-restricted sessions.
    _mark_rescuer_ambient()

    ctx.register_command(
        name="rescuer",
        handler=_status_cmd,
        description="Show Toolaria status: blob count, total size, sessions",
    )


def _mark_rescuer_ambient() -> None:
    """Register the rescuer toolset as ambient with the host tool registry.

    Idempotent and host-agnostic: a host that predates ambient-toolset support
    simply lacks ``registry.mark_ambient``, in which case we emit a prominent
    warning so the operator enables the ``rescuer`` toolset for cron/profile
    sessions instead of discovering dead handles in the logs days later."""
    try:
        from tools.registry import registry
    except Exception as exc:
        logger.warning(
            "toolaria: tool registry unavailable, cannot guarantee rescuer_fetch "
            "availability (%s); rescue handles may be unredeemable in restricted "
            "sessions", exc,
        )
        return
    mark = getattr(registry, "mark_ambient", None)
    if not callable(mark):
        logger.warning(
            "toolaria: host lacks ambient-toolset support; rescuer_fetch will be "
            "UNAVAILABLE in toolset-restricted sessions (cron/profile) and rescue "
            "handles there cannot be fetched. Add 'rescuer' to those sessions' "
            "enabled_toolsets, or upgrade the host.",
        )
        return
    try:
        mark(_RESCUER_TOOLSET)
        logger.info(
            "toolaria: rescuer toolset marked ambient; rescuer_fetch is reachable "
            "in every session regardless of toolset scope",
        )
    except Exception as exc:
        logger.warning("toolaria: mark_ambient('rescuer') failed: %s", exc)


# ── hooks ────────────────────────────────────────────────────────────────


def _on_transform(
    tool_name: str = "",
    result: str = "",
    args: dict | None = None,
    session_id: str = "",
    **kwargs,
):
    """Replace oversized tool results with excerpt + rescue handle."""
    if not _is_rescuable(tool_name):
        return None
    if tool_name in _cfg.get("exclude_tools", []):
        return None
    if not result or not isinstance(result, str):
        return None
    if _store is None:
        # No durable storage means no handle can be honoured; pass the
        # result through untouched rather than destroy content.
        return None

    try:
        if len(result) > _cfg.get("max_result_chars", 12000):
            return _rescue(result, tool_name, args=args, session_id=session_id)
    except Exception as exc:
        logger.warning("toolaria: rescue failed for %s: %s", tool_name, exc)
    return None


def _rescue(result: str, tool_name: str, args: dict | None = None,
            session_id: str = "") -> str | None:
    """Store the result and build the excerpt + handle block.

    Returns None (leave the original untouched) unless the blob is durably
    on disk; a handle that cannot be fetched is worse than no rescue."""
    try:
        blob_id = _store.put(result, tool_name, session_id=session_id)
    except Exception as exc:
        logger.warning("toolaria: blob write failed for %s: %s", tool_name, exc)
        return None

    kind, meta = detect_type(result)
    excerpt = build_excerpt(result, kind, _cfg)
    # Structural outline is cheap and deterministic; build it now so the
    # model can navigate by structure on its first fetch.
    try:
        _store.build_outline(blob_id, result)
    except Exception as exc:
        logger.debug("toolaria: outline build failed for %s: %s", blob_id, exc)
    n_lines = result.count("\n") + 1
    head_lines = _cfg.get("head_lines", 40)
    tail_lines = _cfg.get("tail_lines", 15)
    # Provenance: source URL + timestamp.
    source = ""
    if args and isinstance(args, dict):
        source = args.get("url", args.get("path", args.get("source", "")))
        source = str(source)[:200] if source else ""
    at_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))
    # Semantic role: classify blob by tool function.
    role = _classify_role(tool_name, kind)
    # Build handle with provenance + role enrichment.
    header_fields = [
        f"tool={tool_name}",
    ]
    if source:
        header_fields.append(f"source={source}")
    header_fields += [
        f"at={at_str}",
        f"role={role}",
        f"size={len(result):,} chars",
        f"lines={n_lines:,}",
        f"type={kind} {meta}",
        f"blob={blob_id}",
    ]
    header = "[Toolaria: tool result rescued. " + "; ".join(header_fields) + "]"
    return (
        f"{header}\n"
        f"Preview (first {head_lines} / last {tail_lines} lines); "
        f"this is a preview, NOT the full output:\n"
        f"{excerpt}\n"
        f"Retrieve more with rescuer_fetch(id=\"{blob_id}\", mode=...):\n"
        f"  outline  structural map (sections / JSON schema / error clusters)\n"
        f"  search   find by meaning, e.g. mode=\"search\", query=\"<question>\"\n"
        f"  grep     regex match, e.g. mode=\"grep\", pattern=\"<term>\"\n"
        f"  range    lines, e.g. mode=\"range\", start=0, count=20\n"
        f"  stat | full\n"
        f"To feed this whole result into another tool WITHOUT reading it, pass "
        f"\"tla:{blob_id}\" as that tool's argument; Toolaria expands it to the "
        f"full content before the tool runs."
    )


def _classify_role(tool_name: str, kind: str) -> str:
    """Classify a blob's semantic role: episodic, semantic, or procedural.

    Heuristic derived from ENGRAM (arXiv 2511.12960) three-type memory model,
    mapped to Toolaria's tool/kind signals."""
    # Episodic: raw tool results — outputs of web search, browser, extraction.
    episodic_tools = {
        "web_search", "web_extract", "browser_navigate", "browser_snapshot",
        "browser_click", "browser_type", "browser_scroll", "browser_press",
        "browser_console", "browser_vision", "web_fetch",
    }
    # Semantic: structured summaries — outlines, search results, excerpts.
    semantic_tools = {"rescuer_fetch"}
    # Procedural: pass-by-reference tokens — tla:<id> expansions.
    procedural_tools = {"passref_expand", "tla_expand"}

    if tool_name in procedural_tools:
        return "procedural"
    if tool_name in semantic_tools:
        return "semantic"
    if tool_name in episodic_tools:
        return "episodic"
    # Fallback: classify by content kind.
    if kind in ("json", "html"):
        return "episodic"
    if kind == "code":
        return "procedural"
    return "episodic"


def _on_start(session_id="", **kwargs):
    if _store:
        try:
            _store.lazy_sweep()
        except Exception as exc:
            logger.debug("toolaria: sweep failed on session start: %s", exc)


def _on_end(**kwargs):
    if _store:
        try:
            _store.lazy_sweep()
        except Exception as exc:
            logger.debug("toolaria: sweep failed on session end: %s", exc)


# ── tool handler ─────────────────────────────────────────────────────────


def _fetch(args: dict | None = None, **kwargs) -> str:
    """Handle rescuer_fetch tool calls, dispatched by plugin tool registry.

    Reads session_id from kwargs when the dispatch layer forwards it; the
    store falls back to an all-session metadata search otherwise."""
    if args is None:
        args = {}
    if not _store:
        return "Error: rescuer store not initialised"

    bid = args.get("id", "")
    mode = args.get("mode", "stat")
    try:
        start = int(args.get("start", 0))
        count = int(args.get("count", 20))
    except (TypeError, ValueError):
        return "Error: start and count must be integers"
    pattern = args.get("pattern", "")
    query = args.get("query", "")
    session_id = kwargs.get("session_id", "")

    if not _BLOB_ID_RE.match(bid):
        return f"Error: invalid blob id '{bid}' (expected 12 hex chars)"

    return _store.fetch(bid, mode, start=start, count=count,
                        pattern=pattern, query=query, session_id=session_id)


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

"""Pass-by-reference expansion for rescued blobs.

A rescued result can be handed to another tool without the model ever reading
it: the model writes the token ``tla:<blob_id>`` as a downstream tool's
argument, and this tool_request middleware swaps the token for the blob's full
content before that tool runs. The large payload flows tool to tool and never
re-enters the context window.

The whole point is to move content the model has not seen, so expansion is
bounded by size (to protect the receiving tool, not the context) and degrades
to an honest marker when a blob is missing or over the cap.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"tla:([0-9a-f]{12})")

# Tools that must never receive a silent expansion by default. Pass-by-
# reference moves content the model has not read, so it also bypasses any
# human or filter that inspects model-emitted args; that is fine for content
# tools but dangerous for exec/exfil sinks. A name-substring match is a safety
# net, not the primary control: set passref_allowed_tools for a strict
# allowlist where it matters.
_SINK_DENY = (
    "shell", "bash", "exec", "terminal", "subprocess", "run_command",
    "run_shell", "write_file", "file_write", "fs_write", "edit_file",
    "http_post", "http_request", "curl", "upload",
)


def _tool_allowed(tool_name: str, cfg: dict, skip_tools: frozenset) -> bool:
    if tool_name in skip_tools:
        return False
    allow = cfg.get("passref_allowed_tools") or []
    if allow:
        return tool_name in allow
    low = tool_name.lower()
    return not any(s in low for s in _SINK_DENY)


def expand_value(value, store, cfg: dict, stats: dict):
    """Recursively expand tla: tokens in a JSON-shaped value.

    *stats* accumulates {"expanded": n, "total": chars} so the caller knows
    whether anything changed."""
    if isinstance(value, str):
        return _expand_string(value, store, cfg, stats)
    if isinstance(value, list):
        return [expand_value(v, store, cfg, stats) for v in value]
    if isinstance(value, dict):
        return {k: expand_value(v, store, cfg, stats) for k, v in value.items()}
    return value


def _expand_string(text: str, store, cfg: dict, stats: dict) -> str:
    if "tla:" not in text:
        return text
    cap = int(cfg.get("passref_max_chars", 500000))
    total_cap = int(cfg.get("passref_total_max_chars", 2000000))

    def _sub(m: re.Match) -> str:
        blob_id = m.group(1)
        if stats.get("total", 0) >= total_cap:
            return f"[Toolaria: total expansion budget {total_cap:,} chars exceeded]"
        content = store.blob_text(blob_id) if store else None
        if content is None:
            stats["missing"] = stats.get("missing", 0) + 1
            return f"[Toolaria: blob {blob_id} unavailable; re-run the source tool]"
        if len(content) > cap:
            content = (content[:cap] +
                       f"\n[Toolaria: truncated, blob is {len(content):,} chars "
                       f"> passref_max_chars {cap:,}]")
        stats["expanded"] = stats.get("expanded", 0) + 1
        stats["total"] = stats.get("total", 0) + len(content)
        return content

    return TOKEN_RE.sub(_sub, text)


def make_middleware(get_store, cfg: dict, skip_tools: frozenset):
    """Build a tool_request middleware callback bound to a store accessor."""

    def _tool_request(tool_name: str = "", args=None, **kwargs):
        if not cfg.get("passref_enabled", True):
            return None
        if not isinstance(args, dict) or not _tool_allowed(tool_name, cfg, skip_tools):
            return None
        store = get_store()
        if store is None:
            return None
        stats: dict = {}
        new_args = expand_value(args, store, cfg, stats)
        if not stats:
            return None
        logger.debug("toolaria: pass-by-reference expanded %s for %s",
                     stats, tool_name)
        return {"args": new_args}

    return _tool_request

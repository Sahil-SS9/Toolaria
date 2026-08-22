"""Pass-by-reference expansion for rescued blobs.

A rescued result can be handed to another tool without the model ever reading
it: the model writes the token ``tla:<blob_id>`` as a downstream tool's
argument, and this tool_request middleware swaps the token for the blob's full
content before that tool runs. The large payload flows tool to tool and never
re-enters the context window.

The whole point is to move content the model has not seen, so expansion is
bounded by size (to protect the receiving tool, not the context) and degrades
to an honest marker when a blob is missing or over the cap.

Phase 0 hardening (T0.3): a separate ``passref_external_destinations`` config
key denies expansion into mail-send / social-post / webhook / peer-messaging
classes by name. This deny list is NEVER merged into ``exclude_tools`` (those
tools' oversized results must still be rescued — passref just refuses the
cross-tool handoff) and overrides ``passref_allowed_tools`` when both name a
tool: an operator's permissive allowlist cannot re-enable an external send.

Phase 1 instrumentation (T1.5): every terminal expansion outcome is
appended to a JSONL audit ledger at ``store_path/ledger/expansions.jsonl``.
A failure to write the audit line is logged at WARNING but NEVER raises
into the expansion path — observability must not break delivery.
"""
from __future__ import annotations

import logging
import re

try:
    from .ledger import log_expansion as _log_expansion
except ImportError:
    from ledger import log_expansion as _log_expansion  # type: ignore[no-redef]

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


def _parse_external_destinations(raw) -> frozenset:
    """Validate and freeze the passref_external_destinations config value.

    Operator config is untrusted. Reject anything that is not a list of
    strings at load time (rather than silently dropping bad entries, which
    would make the deny set narrower than the operator thought). Duplicates
    are folded and the empty list is a valid no-op.
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ValueError(
            "passref_external_destinations must be a list of tool names, "
            f"got {type(raw).__name__}"
        )
    out = set()
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                "passref_external_destinations entries must be non-empty "
                f"strings, got {entry!r}"
            )
        out.add(entry)
    return frozenset(out)


# Exact-match set of tool names that must never receive a tla:<id> expansion.
# Loaded once at config merge time via _parse_external_destinations; the
# hardcoded seed below keeps the safety net working out of the box even
# before an operator customises the config (Phase 0 default behaviour).
DEFAULT_EXTERNAL_DESTINATIONS = frozenset({
    # Mail-send class
    "send_email", "send_mail", "post_email", "compose_email", "mail_send",
    # Social-post class
    "social_post", "twitter_post", "linkedin_post", "post_to_social",
    "post_tweet", "post_update",
    # Webhook / chat class
    "webhook_send", "send_webhook", "slack_post", "discord_send",
    # Peer-messaging class (Hermes peer bus)
    "peer_send_message", "peer_broadcast",
})


# Marker returned (per token) when expansion is refused because the
# destination tool is on the external-destinations deny list. The receiving
# tool sees a refusal rather than the (unexpanded) tla:<id> token, so it
# cannot accidentally proceed with raw token text in place of the payload.
_DEST_DENY_MARKER_PREFIX = "[Toolaria: tla:<id> expansion denied; " \
    "tool is on passref_external_destinations — content not forwarded]"


def build_destination_deny_set(cfg: dict) -> frozenset:
    """Compose the effective deny set from cfg.

    Operator-provided ``passref_external_destinations`` is UNIONed with
    DEFAULT_EXTERNAL_DESTINATIONS unless ``passref_disable_builtin_destinations``
    is true. The result is stored back on cfg as a private key so the
    per-request check is O(1).
    """
    user = _parse_external_destinations(cfg.get("passref_external_destinations"))
    if cfg.get("passref_disable_builtin_destinations", False):
        deny = user
    else:
        deny = user | DEFAULT_EXTERNAL_DESTINATIONS
    cfg["_passref_external_destinations_frozen"] = deny
    return deny


def _external_destination_denied(tool_name: str, cfg: dict) -> bool:
    deny = cfg.get("_passref_external_destinations_frozen")
    if deny is None:
        deny = build_destination_deny_set(cfg)
    return tool_name in deny


def _tool_allowed(tool_name: str, cfg: dict, skip_tools: frozenset) -> bool:
    if tool_name in skip_tools:
        return False
    # External-destination deny is checked first and is authoritative: even
    # a tool explicitly named in passref_allowed_tools cannot expand into an
    # external send. This is the documented precedence.
    if _external_destination_denied(tool_name, cfg):
        return False
    allow = cfg.get("passref_allowed_tools") or []
    if allow:
        return tool_name in allow
    low = tool_name.lower()
    return not any(s in low for s in _SINK_DENY)


def expand_value(value, store, cfg: dict, stats: dict, session_id: str = "",
                 tool_name: str = "") -> tuple:
    """Recursively expand tla: tokens in a JSON-shaped value.

    Returns ``(new_value, dest_denied)`` where ``dest_denied`` is True when
    the destination-deny list prevented expansion. The caller
    (``make_middleware``) uses ``dest_denied`` to swap in an honest marker
    so the destination tool sees the refusal rather than the unexpanded
    token.

    *stats* accumulates ``{"expanded", "total", "denied", "missing",
    "dest_denied"}`` so the caller knows whether anything changed.
    """
    if isinstance(value, str):
        return _expand_string(value, store, cfg, stats, session_id,
                              tool_name=tool_name)
    if isinstance(value, list):
        out = []
        any_denied = False
        for v in value:
            new_v, denied = expand_value(v, store, cfg, stats, session_id,
                                         tool_name=tool_name)
            out.append(new_v)
            any_denied = any_denied or denied
        return out, any_denied
    if isinstance(value, dict):
        out = {}
        any_denied = False
        for k, v in value.items():
            new_v, denied = expand_value(v, store, cfg, stats, session_id,
                                         tool_name=tool_name)
            out[k] = new_v
            any_denied = any_denied or denied
        return out, any_denied
    return value, False


def _expand_string(text: str, store, cfg: dict, stats: dict,
                   session_id: str = "", tool_name: str = "") -> tuple:
    """Expand tla:<id> tokens in *text*.

    Returns ``(new_text, dest_denied)``. When ``tool_name`` is on the
    external-destinations deny list, every token in this string is swapped
    for the destination-deny marker so the destination tool sees a refusal
    (it cannot accidentally proceed with the raw token as a payload).
    """
    if "tla:" not in text:
        return text, False
    denied = _external_destination_denied(tool_name, cfg) if tool_name else False
    if denied:
        stats["dest_denied"] = stats.get("dest_denied", 0) + len(TOKEN_RE.findall(text))

    cap = int(cfg.get("passref_max_chars", 500000))
    total_cap = int(cfg.get("passref_total_max_chars", 2000000))

    def _sub(m: re.Match) -> str:
        if denied:
            # T1.5: every token denied by the destination-deny list emits
            # exactly one ledger line. The call is best-effort and never
            # raises into the expansion path.
            _log_expansion(cfg, sid=session_id, blob_id=m.group(1),
                           dst_tool=tool_name, chars=0, decision="dest_denied")
            return _DEST_DENY_MARKER_PREFIX
        blob_id = m.group(1)
        if stats.get("total", 0) >= total_cap:
            _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                           dst_tool=tool_name, chars=0,
                           decision="budget_capped")
            return f"[Toolaria: total expansion budget {total_cap:,} chars exceeded]"
        # Session scoping: when the host forwards a session_id, a blob the
        # calling session does not reference is refused (it belongs to, or was
        # guessed against, another session). An empty session_id means vanilla
        # Hermes did not forward one, so we keep the global behaviour and let
        # single-session setups work, mirroring fetch's all-session fallback.
        if session_id and store and not store.session_references(blob_id, session_id):
            stats["denied"] = stats.get("denied", 0) + 1
            _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                           dst_tool=tool_name, chars=0,
                           decision="session_denied")
            return f"[Toolaria: blob {blob_id} not available in this session]"
        content = store.blob_text(blob_id) if store else None
        if content is None:
            stats["missing"] = stats.get("missing", 0) + 1
            _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                           dst_tool=tool_name, chars=0, decision="missing")
            return f"[Toolaria: blob {blob_id} unavailable; re-run the source tool]"
        if len(content) > cap:
            content = (content[:cap] +
                       f"\n[Toolaria: truncated, blob is {len(content):,} chars "
                       f"> passref_max_chars {cap:,}]")
        stats["expanded"] = stats.get("expanded", 0) + 1
        stats["total"] = stats.get("total", 0) + len(content)
        _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                       dst_tool=tool_name, chars=len(content),
                       decision="expanded")
        return content

    return TOKEN_RE.sub(_sub, text), denied


def make_middleware(get_store, cfg: dict, skip_tools: frozenset):
    """Build a tool_request middleware callback bound to a store accessor."""

    # Pre-freeze the destination-deny set so the per-request check is O(1).
    build_destination_deny_set(cfg)

    def _tool_request(tool_name: str = "", args=None, **kwargs):
        if not cfg.get("passref_enabled", True):
            return None
        if not isinstance(args, dict):
            return None
        store = get_store()
        if store is None:
            return None
        session_id = kwargs.get("session_id", "")
        stats: dict = {}
        # We expand unconditionally here even when the tool is not allowed:
        # the destination-deny path needs to produce a marker (modified
        # args) so the destination tool sees a refusal rather than the
        # unexpanded tla:<id> token. Tools in skip_tools / sink-deny that
        # have no token still get a fast-path return below.
        new_args, dest_denied = expand_value(args, store, cfg, stats,
                                             session_id, tool_name=tool_name)
        if dest_denied:
            logger.info(
                "toolaria: pass-by-reference destination-deny %s (%s tokens refused)",
                tool_name, stats.get("dest_denied", 0),
            )
            return {"args": new_args}
        if not _tool_allowed(tool_name, cfg, skip_tools):
            return None
        if not stats:
            return None
        logger.debug("toolaria: pass-by-reference expanded %s for %s",
                     stats, tool_name)
        return {"args": new_args}

    return _tool_request

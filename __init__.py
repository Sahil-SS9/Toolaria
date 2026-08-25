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

# Phase 0 hardening (T0.4): set to True by register() when the host exposes
# the tool_request middleware contract. Until set, pass-by-reference is
# "dead" — there is no path that can expand a tla:<id> token. The rescue
# handle must not advertise the tla:<id> instruction while this is False,
# because the instruction would be a dead handle (worse than no rescue).
# Cleared back to False whenever the host signals that middleware
# registration failed (e.g. attribute missing).
_passref_alive: bool = False

# Tools whose results may exceed context: the only built-ins rescued.
# MCP tools are detected dynamically via the registry toolset prefix.
# Phase 0 (T0.3): mail-send / social-post / webhook / peer-messaging tools
# are explicitly rescuable too — passref (downstream handoff) blocks them
# via the external-destination deny list, but the upstream rescue path
# still spills their oversized RESULTS to disk so the model can fetch
# them by handle like any other rescued blob.
_RESCUABLE_TOOLS: set[str] = {
    "web_extract",
    "web_search",
    "browser_navigate",
    "browser_snapshot",
    "browser_console",
    "browser_get_images",
    # External-destination class — rescue upstream, deny passref downstream.
    "send_email", "send_mail", "post_email", "compose_email", "mail_send",
    "social_post", "twitter_post", "linkedin_post", "post_to_social",
    "post_tweet", "post_update",
    "webhook_send", "send_webhook", "slack_post", "discord_send",
    "peer_send_message", "peer_broadcast",
}

# Single source of truth for tools that must never be intercepted.
# Enforced unconditionally in _on_transform, so _is_rescuable failing open
# (registry import broken) still cannot touch these. Hardened in Phase 0
# (T0.2): shell/exec-class and file-write-class tools are added so a broken
# registry cannot rescue token-bearing shell output or the contents of a
# file about to be edited.
_UNCONDITIONAL_EXCLUDES: frozenset[str] = frozenset({
    "rescuer_fetch", "delegate_task", "session_search",
    "cronjob", "skill_view", "skill_manage", "skill_request",
    "kanban_create", "open_kanban", "clarify", "memory",
    # Shell / exec-class sinks — token-bearing output must never hit the store.
    "shell", "bash", "exec", "terminal", "subprocess",
    "run_command", "run_shell",
    # File-write-class sinks — file contents about to be written must not be
    # rescued, both because they may contain secrets and because rescuing
    # them adds no value (the model just wrote them).
    "write_file", "file_write", "fs_write", "edit_file",
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
    """Merge user config over plugin defaults.

    Phase 1 instrumentation defaults (T1.1/T1.2/T1.3/T1.4) are seeded in
    code so the cfg carries them even when PyYAML is unavailable to parse
    config.yaml. The YAML file remains the operator-facing knob; the
    code-level defaults are the hard floor that keeps the suite passing
    on the canonical test runner (``uv run --with pytest --with regex``).

    Phase 3 FIX-4: validate ``sensitivity_tool_labels`` here so a bad
    operator config (typo in a label name, list instead of dict, etc.)
    fails LOUD at register time — the plugin refuses to come up
    rather than silently disabling rescue later."""
    defaults = _load_defaults()
    defaults.update(user_cfg)
    for _k, _v in _PHASE1_DEFAULTS.items():
        defaults.setdefault(_k, _v)
    for _k, _v in _PHASE2_DEFAULTS.items():
        defaults.setdefault(_k, _v)
    # Phase 3 FIX-4: surface operator typos at startup. The validator
    # raises ValueError listing offending keys/values; we let that
    # bubble up to register() and abort plugin load so the operator
    # sees the typo before any rescue runs.
    from labels import _parse_tool_label_map
    _parse_tool_label_map(defaults.get("sensitivity_tool_labels"))
    # T3.1: validate the entity_registry the same way so a broken
    # pattern/regex/sensitivity never silently disables the governor
    # or leaks into a half-broken put().
    try:
        from entities import parse_entity_registry
        parse_entity_registry(defaults.get("entity_registry"))
    except ImportError:
        # entities.py missing on this checkout (e.g. running an older
        # snapshot). Skip — the feature stays inert.
        pass
    return defaults


# Phase 1 instrumentation defaults (T1.1/T1.2/T1.3/T1.4). See
# docs/plans/2026-08-22-governance-expansion.md for rationale.
_PHASE1_DEFAULTS: dict = {
    "sequence_capture": False,      # T1.2: default OFF ⇒ zero sidecar writes
    "args_snapshot_max_chars": 2000,  # T1.3: cap on the per-blob redacted args snapshot
    "verify_integrity": True,        # T1.4: default ON; benchmarks opt out via cfg
}

# Phase 2 data-governance defaults (T2.1/T2.3). These are the hard
# floor that keeps the cfg consistent even when PyYAML is unavailable
# to parse config.yaml. The YAML file remains the operator-facing knob;
# code-level defaults here are the safe-on-every-install defaults.
_PHASE2_DEFAULTS: dict = {
    "sensitivity_tool_labels": {},  # T2.1: built-ins cover the common cases
}


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
        # Phase 3 FIX-1 / FIX-3: backfill labels on every existing
        # index entry before the first sweep. Wrapped best-effort so a
        # single broken entry cannot break register() (the comment
        # on backfill_labels() documents the same posture).
        try:
            _store.backfill_labels()
        except Exception as e:
            logger.warning("toolaria: label backfill failed: %s", e)
        _store.lazy_sweep()
    except Exception as e:
        logger.warning("toolaria: blob store init failed, rescuing disabled: %s", e)
        _store = None

    ctx.register_hook("transform_tool_result", _on_transform)
    ctx.register_hook("on_session_start", _on_start)
    ctx.register_hook("on_session_end", _on_end)

    # Pass-by-reference: expand tla:<id> tokens in downstream tool args into
    # full blob content before the tool runs, so a rescued result can flow
    # tool to tool without ever re-entering the model's context. The
    # ``register_middleware`` contract is host-version dependent; older
    # hosts lack it entirely. Phase 0 (T0.4): on a missing register helper,
    # log a fail-loud WARNING and mark passref dead so subsequent rescue
    # handles do NOT advertise a tla:<id> instruction the host cannot
    # honour. A dead handle is worse than no rescue.
    global _passref_alive
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware(
            "tool_request",
            _make_passref_mw(lambda: _store, _cfg, _UNCONDITIONAL_EXCLUDES),
        )
        _passref_alive = bool(_cfg.get("passref_enabled", True))
    else:
        _passref_alive = False
        logger.warning(
            "toolaria: host lacks register_middleware; pass-by-reference "
            "expansion disabled. Rescue handles will omit the tla:<id> "
            "instruction (a dead handle is worse than no rescue). Upgrade "
            "the host or disable passref (passref_enabled: false) to "
            "silence this warning."
        )

    ctx.register_tool(
        name="rescuer_fetch",
        toolset=_RESCUER_TOOLSET,
        description=(
            "Fetch slices of a rescued oversized tool result. Modes: "
            "outline | search(query) | range(start,count) | grep(pattern) | "
            "chain(pattern, count) | stat | full"
        ),
        handler=_fetch,
        schema={
            "name": "rescuer_fetch",
            "description": (
                "Retrieve slices of a rescued tool result blob. Modes: "
                "outline | search(query) | range(start,count) | "
                "grep(pattern) | chain(pattern,count) | stat | full."
            ),
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
                                 "chain", "stat", "full", "audit"],
                        "description": ("Retrieval mode (default: stat). "
                                        "'audit' returns the read-only "
                                        "expansion ledger summary."),
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start line for range mode (default: 0)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Lines to return for range mode or "
                                       "context width for chain mode "
                                       "(default: 20)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex for grep mode, also used by "
                                       "chain mode for the match step",
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
    ctx.register_command(
        name="toolaria-rotate-key",
        handler=_rotate_key_cmd,
        description=("Rotate the Toolaria encryption key: "
                     "/toolaria-rotate-key <new-key-path>"),
    )


def _rotate_key_cmd(raw_args: str = "") -> str:
    """Operator command: durable, restart-safe key rotation.

    Rotation and the durable config pointer are serialised under a real
    cross-process lock. The replacement config is written at owner-only
    permissions, flushed, fsynced and atomically installed without weakening
    an existing stricter owner-only mode.
    """
    if not _store:
        return "Error: Toolaria store not initialised"
    requested = (raw_args or "").strip()
    if not requested:
        return ("Usage: /toolaria-rotate-key <new-key-path>\n"
                "The new key file must be OUTSIDE the store directory "
                "and will be created at 0600 if missing.")
    old_cfg = _cfg.get("toolaria_key_file")
    if not old_cfg:
        return ("Error: no toolaria_key_file configured in config.yaml; "
                "nothing to rotate. Set it first, then re-run.")

    import json as _json
    import re as _re
    import stat as _stat
    import tempfile as _tempfile

    new_path_p = Path(requested).expanduser().resolve()
    store_path = Path(_store.store_path).expanduser().resolve()
    if new_path_p == store_path or store_path in new_path_p.parents:
        return ("Error: new key file must be outside the Toolaria store "
                f"directory ({store_path})")
    new_path = str(new_path_p)

    hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    cfg_path = hermes_home / "config.yaml"
    if not cfg_path.is_file():
        return (f"Error: durable Hermes config not found at {cfg_path}; "
                "rotation was not started")
    lock_path = cfg_path.with_suffix(cfg_path.suffix + ".toolaria.lock")
    lock_fd = None
    tmp_name = None
    encoded_path = _json.dumps(new_path)
    rotation_completed = False
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(lock_fd, 0o600)
        try:
            import fcntl as _fcntl
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX host
            pass

        # Preflight the exact durable pointer before touching blob bytes.
        text = cfg_path.read_text()
        pattern = _re.compile(
            r"^(?P<prefix>[ \t]+toolaria_key_file[ \t]*:[ \t]*).*$",
            _re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) != 1 or str(old_cfg) not in matches[0].group(0):
            return (
                "Error: config.yaml does not contain exactly one matching "
                "toolaria_key_file entry for the active old key; rotation "
                "was not started"
            )
        # JSON strings are valid YAML scalars and safely preserve '#', ':',
        # quotes, backslashes and whitespace in operator-supplied paths.
        new_text = pattern.sub(
            lambda m: m.group("prefix") + encoded_path, text, count=1)

        count = _store.rotate_key(new_path)
        rotation_completed = True

        original_mode = _stat.S_IMODE(cfg_path.stat().st_mode)
        target_mode = original_mode & 0o600
        if target_mode == 0:
            target_mode = 0o600
        fd, tmp_name = _tempfile.mkstemp(
            dir=cfg_path.parent,
            prefix=f".{cfg_path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), target_mode)
            f.write(new_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, cfg_path)
        tmp_name = None
        os.chmod(cfg_path, target_mode)
        # Persist the directory entry as well as the file contents.
        dir_fd = os.open(str(cfg_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        _cfg["toolaria_key_file"] = new_path
    except Exception as exc:
        if not rotation_completed:
            return (
                f"Error: rotation failed before completion: {exc}\n"
                "No durable config change was made."
            )
        # rotate_key persists the new key before rewriting blobs. If config
        # installation fails after a successful rotation, the new key remains
        # recoverable and the exact durable pointer is reported.
        return (
            "PARTIAL SUCCESS — read carefully:\n"
            f"  New key target: {new_path}\n"
            f"  BUT durable config update FAILED: {exc}\n"
            f"Recovery: set 'toolaria_key_file' in {cfg_path} to "
            f"{encoded_path if 'encoded_path' in locals() else new_path!r}, "
            "then restart the gateway. Do NOT delete either key file."
        )
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        if lock_fd is not None:
            try:
                import fcntl as _fcntl
                _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            os.close(lock_fd)

    return (
        f"Key rotation complete.\n"
        f"  blobs re-encrypted: {count}\n"
        f"  new key file: {new_path} (0600)\n"
        f"  config.yaml updated: toolaria_key_file -> {new_path}\n"
        f"NEXT STEP (required): restart the gateway —\n"
        f"  sudo systemctl restart hermes-gateway.service\n"
        f"The OLD key file ({old_cfg}) can be archived once you have\n"
        f"confirmed decryption works after the restart."
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


def _passref_active() -> bool:
    """True if a tla:<id> instruction in a rescue handle will be honoured.

    Combines the register-time middleware-availability check
    (``_passref_alive``) with the per-request passref_enabled config knob
    so the rescue handle text stays in lock-step with the runtime state
    of the expansion middleware. Both must be true for the instruction
    to be emitted.
    """
    return bool(_passref_alive) and bool(_cfg.get("passref_enabled", True))


def _rescue(result: str, tool_name: str, args: dict | None = None,
            session_id: str = "") -> str | None:
    """Store the result and build the excerpt + handle block.

    Returns None (leave the original untouched) unless the blob is durably
    on disk; a handle that cannot be fetched is worse than no rescue."""
    try:
        # T1.3: pass the caller's args (already validated as a dict by
        # the hook contract) so a redacted snapshot lands in the index
        # entry. Redaction happens inside put() — the rescue path itself
        # never sees the raw keys/values, which keeps the secret-handling
        # boundary in one place.
        blob_id = _store.put(result, tool_name, session_id=session_id, args=args)
    except Exception as exc:
        logger.warning("toolaria: blob write failed for %s: %s", tool_name, exc)
        return None

    kind, meta = detect_type(result)
    excerpt = build_excerpt(result, kind, _cfg)
    # Structural outline is cheap and deterministic; build it now so the
    # model can navigate by structure on its first fetch.
    # Reviewer fix 1 (2026-08-25): sidecar suppression now lives inside
    # BlobStore.build_outline (single chokepoint covering outline,
    # chunks, and vectors paths), so the rescue path just calls it —
    # no duplicated encryption checks here. Stale plaintext sidecars
    # from a prior non-encrypted write of the same content-addressed
    # bid are still removed.
    try:
        if _store._sidecars_forbidden(blob_id):
            logger.info(
                "toolaria: skipping sidecars for encrypted credential "
                "blob %s (reviewer fix 1)", blob_id,
            )
            _store.delete_sidecars(blob_id)
        else:
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
    if _passref_active():
        handle_tail = (
            f"To feed this whole result into another tool WITHOUT reading "
            f"it, pass \"tla:{blob_id}\" as that tool's argument; Toolaria "
            f"expands it to the full content before the tool runs."
        )
    else:
        # Phase 0 (T0.4): pass-by-reference is unavailable (no middleware
        # OR passref_enabled: false). Advertising a tla:<id> instruction
        # would be a dead handle — emit rescuer_fetch instructions only.
        handle_tail = (
            f"Retrieve the full result with rescuer_fetch(id=\"{blob_id}\", "
            f"mode=\"full\") or fetch slices with mode=range/grep/outline."
        )
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
        f"{handle_tail}"
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

    # T2.2: audit mode is a status query, not a blob retrieval. It
    # reads the expansion ledger directly (no blob bytes touched, no
    # fetch_count bumped). The blob id is accepted but unused, so a
    # operator can call audit with any well-formed handle from a recent
    # rescue without re-running the rescue.
    if mode == "audit":
        return _store._audit_summary(count)

    bid_raw = args.get("id", "")
    # T4.1: accept the ``<bid>[@<int>]`` grammar. The store's fetch
    # layer parses it; a malformed ref is rejected there too, so we
    # only check the bid half here to keep the legacy "12 hex chars"
    # error wording for callers that do not pass a ref at all.
    bid_only = bid_raw.split("@", 1)[0] if bid_raw else ""
    if not _BLOB_ID_RE.match(bid_only):
        return f"Error: invalid blob id '{bid_raw}' (expected 12 hex chars)"

    return _store.fetch(bid_raw, mode, start=start, count=count,
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

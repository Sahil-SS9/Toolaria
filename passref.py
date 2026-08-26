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

if __package__:
    from .ledger import log_expansion as _log_expansion
    from .ledger import log_entity_binding
else:
    from ledger import log_expansion as _log_expansion  # type: ignore[no-redef]
    from ledger import log_entity_binding  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# T4.1: ``<bid>[@<int>]`` grammar. The optional version suffix
# addresses a specific revision in the chain. Group 2 is the bare
# 12-hex content id (used for the fetch call), group 3 is the
# requested version.
TOKEN_RE = re.compile(r"tla:([0-9a-f]{12})(@(\d+))?")
# Backwards-compatible shorthand: when callers (or tests) expect the
# bare 12-hex shape, ORed expansion below recovers both.

# HG-004 (hermaguard Phase 3): cap on entity-binding rows written per
# request. Beyond blob_ids × kinds rows, a single overflow summary row
# (entity_bound_overflow) replaces the cross-product so a hostile
# request cannot fan the ledger out unboundedly.
_BINDING_ROWS_PER_REQUEST_CAP = 64

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

# T2.3 — credential-grade enforcement markers. The full marker is
# constructed dynamically (it embeds the blob id), but the prefix and
# suffix are exported so tests can pin against the exact shape. The
# marker appears in TWO places: passref (downstream handoff refusal)
# and rescuer_fetch full mode (upstream full-read refusal).
CREDENTIAL_REFUSE_MARKER_PREFIX = (
    "[Toolaria: credential-labelled blob "
)
CREDENTIAL_REFUSE_MARKER_SUFFIX = (
    " withheld; use range/grep slices or add destination to "
    "credential_destinations]"
)

# T4.1: version-refusal marker returned by passref expansion when a
# ``tla:<bid>@<N>`` token references a version that does not exist in
# the calling session's chain.
VERSION_REFUSE_MARKER_PREFIX = "[Toolaria: version "
VERSION_REFUSE_MARKER_MIDDLE = " not found for blob "
VERSION_REFUSE_MARKER_SUFFIX = "]"


def _parse_credential_destinations(raw) -> frozenset:
    """Validate the operator-controlled allowlist of credential destinations.

    Mirrors ``_parse_external_destinations`` — same contract, same fail-loud
    posture. Empty / None is a valid deny-all (no destinations permitted).
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ValueError(
            "credential_destinations must be a list of tool names, "
            f"got {type(raw).__name__}"
        )
    out = set()
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                "credential_destinations entries must be non-empty "
                f"strings, got {entry!r}"
            )
        out.add(entry)
    return frozenset(out)


def _credential_destinations_allow(cfg: dict) -> frozenset:
    """Resolve and freeze the credential_destinations allowlist.

    Cached under a private key so the per-token hot-path check is O(1).
    Empty default ⇒ all credential expansions are refused.
    """
    allow = cfg.get("_credential_destinations_frozen")
    if allow is None:
        allow = _parse_credential_destinations(
            cfg.get("credential_destinations"))
        cfg["_credential_destinations_frozen"] = allow
    return allow


def _credential_enforcement_active(cfg: dict) -> bool:
    """True iff credential-grade enforcement is enabled.

    Explicit truthy semantics (not bare bool()) so a YAML-quoted
    ``"false"`` actually disables the gate. This function only checks
    ``enforcement_enabled`` — it does NOT check the allowlist: an empty
    allowlist + enforcement on is a deny-all (every credential
    expansion refused), which is the safe default.
    ``_credential_destinations_allow`` is the per-destination check."""
    if __package__:
        from .blobstore import BlobStore
    else:
        from blobstore import BlobStore  # type: ignore[no-redef]
    return BlobStore._truthy(cfg.get("enforcement_enabled", False))


def _find_label(store, blob_id: str, session_id: str) -> str:
    """Phase 3 FIX-1 + FIX-3: content-aware label lookup, fail-closed
    under enforcement_enabled=True.

    The label is a property of the *content* (blob_id = SHA256 prefix).
    We resolve the highest sensitivity label present across every
    session index that holds the blob, not just the calling session's
    entry — so a credential blob that was re-rescued as public in
    another session still resolves to ``credential`` here.

    FAIL-CLOSED (FIX-3): when ``enforcement_enabled`` is True, an
    unresolved label (None / missing everywhere / I/O error) is
    treated as ``credential`` — never ``public``. Under enforcement
    OFF the audit-friendly default of ``public`` is kept so the audit
    script can run cleanly against mixed-version stores.

    The audit script can still flag label-less rows in its summary
    when enforcement is OFF (the fail-open case for mixed stores).
    """
    if store is None:
        return "public"
    enforcement_on = _credential_enforcement_active(
        getattr(store, "cfg", {}) or {})
    try:
        # FIX-1: use the content-aware max-label scan so a credential
        # blob downgraded in another session still resolves correctly.
        max_label = store._max_label_for_blob(blob_id)
    except Exception:
        max_label = None
    if max_label is None:
        # FIX-3: under enforcement ON, fail-closed.
        return "credential" if enforcement_on else "public"
    return max_label


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
        # len() of finditer: TOKEN_RE now has optional @N groups (T4.1),
        # so findall would return tuples — the count is what matters.
        stats["dest_denied"] = stats.get("dest_denied", 0) + sum(
            1 for _ in TOKEN_RE.finditer(text))

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
        version_n = int(m.group(3) or 0)
        # T4.1: resolve ``tla:<bid>@<N>`` to the chain-specific bid in
        # the calling session's index. ``version_n == 0`` is the bare
        # bid form (no @N) — that path is byte-identical to pre-T4.1.
        idx: dict | None = None
        if version_n > 0:
            if session_id and store is not None:
                resolved: str | None = None
                try:
                    idx = store._load_idx(session_id)
                    resolved = store._resolve_version(
                        idx, blob_id, version_n)
                except Exception:
                    resolved = None
                blobs_idx: dict = idx.get("blobs", {}) if idx else {}
                if not resolved or resolved not in blobs_idx:
                    if stats.get("total", 0) >= total_cap:
                        _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                                       dst_tool=tool_name, chars=0,
                                       decision="budget_capped")
                        return (f"[Toolaria: total expansion budget "
                                f"{total_cap:,} chars exceeded]")
                    _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                                   dst_tool=tool_name, chars=0,
                                   decision="missing")
                    stats["missing"] = stats.get("missing", 0) + 1
                    return (f"{VERSION_REFUSE_MARKER_PREFIX}{version_n}"
                            f"{VERSION_REFUSE_MARKER_MIDDLE}{blob_id}"
                            f"{VERSION_REFUSE_MARKER_SUFFIX}")
                blob_id = resolved
            else:
                # Either no session scoped (empty-session fallback)
                # or no store at all: treat unknown @N the same way as
                # the rest of passref treats an unwalkable chain.
                stats["missing"] = stats.get("missing", 0) + 1
                return (f"{VERSION_REFUSE_MARKER_PREFIX}{version_n}"
                        f"{VERSION_REFUSE_MARKER_MIDDLE}{blob_id}"
                        f"{VERSION_REFUSE_MARKER_SUFFIX}")
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
        # T2.3: credential-grade enforcement. With enforcement on, a
        # credential blob only expands into destinations on the
        # allowlist; everyone else sees the deterministic refusal
        # marker. The check happens after session-scope + existence so
        # a denied attempt still costs the same I/O as an allowed one
        # (no side-channel about whether the blob exists for the
        # calling session).
        label = _find_label(store, blob_id, session_id)
        if _credential_enforcement_active(cfg) and label == "credential":
            allow = _credential_destinations_allow(cfg)
            if tool_name not in allow:
                stats["denied"] = stats.get("denied", 0) + 1
                _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                               dst_tool=tool_name, chars=0,
                               decision="credential_denied",
                               label=label)
                return (f"{CREDENTIAL_REFUSE_MARKER_PREFIX}{blob_id}"
                        f"{CREDENTIAL_REFUSE_MARKER_SUFFIX}")
        if len(content) > cap:
            content = (content[:cap] +
                       f"\n[Toolaria: truncated, blob is {len(content):,} chars "
                       f"> passref_max_chars {cap:,}]")
        stats["expanded"] = stats.get("expanded", 0) + 1
        stats["total"] = stats.get("total", 0) + len(content)
        _log_expansion(cfg, sid=session_id, blob_id=blob_id,
                       dst_tool=tool_name, chars=len(content),
                       decision="expanded", label=label)
        return content

    return TOKEN_RE.sub(_sub, text), denied


def _confirmation_required(cfg: dict) -> bool:
    """T3.3: ambiguity confirmation gate active iff explicitly enabled.

    Truthy semantics (not bare bool()) so a YAML-quoted ``"false"``
    actually disables the gate — same posture as
    ``_credential_enforcement_active``.
    """
    if __package__:
        from .blobstore import BlobStore
    else:
        from blobstore import BlobStore  # type: ignore[no-redef]
    return BlobStore._truthy(cfg.get("confirmation_required", False))


# Confirmation marker template: ``"<prefix> N entity kinds (<kinds>); <suffix>"``.
# Exported so tests can pin against the exact shape and value_flow_audit
# can group rows by the same string.
ENTITY_CONFIRMATION_MARKER_PREFIX = (
    "[Toolaria: action spans "
)
ENTITY_CONFIRMATION_MARKER_KINDS = " entity kinds ("
ENTITY_CONFIRMATION_MARKER_SEP = ", "
ENTITY_CONFIRMATION_MARKER_MIDDLE = "); confirm target or widen entity_registry]"
ENTITY_CONFIRMATION_MARKER = (
    ENTITY_CONFIRMATION_MARKER_PREFIX + "{n}" + ENTITY_CONFIRMATION_MARKER_KINDS
    + "{kinds}" + ENTITY_CONFIRMATION_MARKER_MIDDLE
)


def _confirmation_marker(kinds: list[str]) -> str:
    """T3.3: build the deterministic confirmation marker for ``kinds``.

    ``kinds`` is sorted defensively (callers already pass sorted
    distinct kinds via ``distinct_entity_kinds``; sorting again here
    means a hand-built test list still produces the canonical output).
    Output is exact and stable — the audit script can match on the
    literal prefix and the ``<kinds>`` substring to attribute
    ambiguous-gated requests.
    """
    return ENTITY_CONFIRMATION_MARKER.format(
        n=len(kinds), kinds=ENTITY_CONFIRMATION_MARKER_SEP.join(
            sorted(kinds)))


def _blob_ids_in_args(args) -> list[str]:
    """Return the order-stable de-duplicated list of tla:<id> blob ids
    found in nested args.

    Recurses over dict/list/str so the same shape the rest of the
    middleware uses for token expansion is covered.
    """
    found: list[str] = []
    seen: set[str] = set()

    def _walk(v) -> None:
        if isinstance(v, str):
            # TOKEN_RE now carries the optional @N version groups
            # (T4.1); findall returns tuples, so take group 1 — the
            # bare 12-hex content id, which is what binding rows and
            # expansion have always keyed on.
            for m in TOKEN_RE.finditer(v):
                bid = m.group(1)
                if bid not in seen:
                    seen.add(bid)
                    found.append(bid)
        elif isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                _walk(vv)

    _walk(args)
    return found


def _replace_tokens_with_marker(args, marker: str):
    """Return a copy of *args* with every ``tla:<id>`` token replaced
    by *marker*. The shape is preserved (dict/list/str); non-tla
    values pass through unchanged.

    HG-002 (hermaguard Phase 3): the marker must be inserted as a
    literal. A string replacement in ``re.sub`` expands ``\\1`` /
    ``\\g<name>`` as backreferences, so a hostile registry ``kind``
    could crash the middleware or inject matched text. The callable
    form makes the marker always literal regardless of content.
    """
    def _walk(v):
        if isinstance(v, str):
            return TOKEN_RE.sub(lambda _m: marker, v)
        if isinstance(v, dict):
            return {k: _walk(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_walk(vv) for vv in v]
        if isinstance(v, tuple):
            return tuple(_walk(vv) for vv in v)
        return v

    return _walk(args)


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
        # T3.1 / T3.2 / T3.3: entity-binding governor. Default-empty
        # ``entity_registry`` short-circuits the whole block so the
        # path is byte-identical to pre-T3 when the operator has not
        # opted in. The block sits BEFORE expansion so binding rows are
        # recorded in the ledger with the pre-expansion context.
        if __package__:
            from .entities import (
                get_registry,
                extract_entities,
                distinct_entity_kinds,
            )
        else:
            from entities import (  # type: ignore[no-redef]
                get_registry,
                extract_entities,
                distinct_entity_kinds,
            )
        try:
            _entity_reg = get_registry(cfg)
        except Exception as exc:
            # A broken entity_registry would have failed at register
            # time (T3.1 / FIX-4). Reaching here means a config was
            # mutated at runtime; log and stay inert so the rescue
            # path never crashes.
            logger.warning(
                "toolaria: entity_registry resolve failed at request "
                "time: %s; skipping binding scan", exc,
            )
            _entity_reg = []
        if _entity_reg:
            # HG-001/HG-003 (hermaguard Phase 3): the entity scan is
            # best-effort on this path — a pathological registry regex
            # hits its own time/length budget inside extract_entities,
            # and any residual failure (e.g. deep-nested args) logs and
            # degrades to no-bindings instead of crashing the request.
            try:
                matches = extract_entities(args, _entity_reg)
            except Exception as exc:
                logger.warning(
                    "toolaria: entity scan failed for %s: %s; "
                    "skipping binding/gate checks", tool_name, exc,
                )
                matches = []
            kinds = distinct_entity_kinds(matches)
            blob_ids = _blob_ids_in_args(args)
            # T3.2: observe-only binding rows, one per (tool, kind,
            # blob_id) combination. Logged even when the destination
            # is later denied — the binding captures the action,
            # not its allow status.
            # HG-004 (hermaguard Phase 3): the cross-product is capped
            # per request; beyond the cap a single summary row with
            # entity_kinds is written instead of one row per pair.
            if matches and blob_ids:
                pairs = len(blob_ids) * len(kinds)
                if pairs > _BINDING_ROWS_PER_REQUEST_CAP:
                    log_entity_binding(
                        cfg, sid=session_id, tool=tool_name,
                        entity_kinds=kinds,
                        decision="entity_bound_overflow",
                    )
                else:
                    for bid in blob_ids:
                        for kind in kinds:
                            log_entity_binding(
                                cfg, sid=session_id, tool=tool_name,
                                entity_kind=kind, blob_id=bid,
                                decision="entity_bound",
                            )
            # T3.3: ambiguity confirmation gate. Only fires when (a)
            # there is at least one token to expand AND (b) the args
            # span multiple distinct kinds. With confirmation_required
            # OFF (default) the gate is dormant and expansion proceeds
            # unchanged.
            # HG-005 (hermaguard Phase 3): this gate is an audit and
            # friction convention, NOT a security boundary. It keys on
            # arg phrasing, so a caller can trivially stay under it by
            # splitting one kind per request or inlining content. Real
            # blocking would require content-level entity detection on
            # the expanded payload — a deliberate Phase-4+ decision.
            if (blob_ids and len(kinds) > 1
                    and _confirmation_required(cfg)):
                marker = _confirmation_marker(kinds)
                log_entity_binding(
                    cfg, sid=session_id, tool=tool_name,
                    entity_kinds=kinds, decision="ambiguous_gated",
                )
                return {"args": _replace_tokens_with_marker(args, marker)}
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

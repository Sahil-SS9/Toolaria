"""Entity registry for Toolaria's entity-binding governor (Phase 3 T3.1–T3.3).

Deterministic, config-driven entity patterns. An entity is a person,
document, or record that appears in tool arguments; the governor uses
these patterns to:

  - T3.1  tag blobs with the entity kinds their args mention
          (``entity_kinds`` lives on the index entry alongside label
          and survives both sweep paths)
  - T3.2  log action→entity bindings during passref expansion
          (observe-only, no expansion behaviour change)
  - T3.3  gate ambiguous expansions that span multiple distinct entity
          kinds via the confirmation marker

Config (default empty ⇒ the whole feature is inert):

    entity_registry:
      - pattern_type: name          # or: regex
        pattern: "alice"            # literal name (case-insensitive) or a regex
        kind: person                # free-form, drives the <kinds> marker
        sensitivity: personal       # one of labels.VALID_LABELS (default public)

The registry is frozen + validated at register time; a malformed entry
raises ValueError so a broken config fails loud (same posture as the
T2.1 sensitivity_tool_labels validator). When the registry is empty,
``extract_entities`` returns ``[]`` and the per-call hot path is a
single O(1) list check — the feature stays byte-identical inert.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from labels import VALID_LABELS

logger = logging.getLogger(__name__)

# HG-001 (hermaguard Phase 3): bounds for the entity scan hot path.
# Registry regexes are operator-supplied and can backtrack
# catastrophically; the scan therefore never sees unbounded text and
# never runs unbounded in wall-clock time. When the optional `regex`
# engine is available (same dependency the grep path uses), every
# registry regex runs through it with a hard mid-match timeout —
# Python's `re` cannot be interrupted mid-search, so WITHOUT the
# `regex` package a catastrophic pattern blocks once per string before
# the budget aborts. To keep that worst case bounded, stdlib-compiled
# registry regexes are additionally rejected if their pattern contains
# nested quantifiers (the classic ReDoS shape).
_ENTITY_SCAN_MAX_CHARS = 4096
_ENTITY_SCAN_BUDGET_S = 0.25
_ENTITY_REGEX_TIMEOUT_S = 0.05
_ENTITY_SCAN_MAX_DEPTH = 64

try:
    import regex as _regex_engine  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback engine
    _regex_engine = None

# Nested-quantifier heuristic: quantifier directly applied to a group
# that itself ends with a quantifier, e.g. (a+)+, (\\w+\\s?)*.
_NESTED_QUANT_RE = re.compile(
    r"\((?:[^()\\]|\\.)*[+*]\s*\)\s*[+*{]")


def _compile_registry_regex(pattern: str):
    """Compile *pattern*, preferring the timeout-honouring engine.

    Returns the compiled pattern. When the `regex` package is present
    it is used (its ``search``/``finditer`` accept a ``timeout=``
    keyword that interrupts mid-match — Python's ``re`` cannot). Falls
    back to ``re.compile`` otherwise; callers then rely on the length
    + wall-clock budget as best-effort bounds.
    """
    if _regex_engine is not None:
        return _regex_engine.compile(pattern)
    return re.compile(pattern)


# pattern_type is the only discriminated field; everything else is
# plain metadata validated by type/format below.
_ENTITY_PATTERN_TYPES = frozenset({"name", "regex"})

# HG-002 / HG-006 (hermaguard Phase 3): ``kind`` reaches the
# confirmation marker and audit keys, so it is restricted to a safe,
# human-meaningful charset. Lowercase words with . - _ separators.
_KIND_PATTERN = r"[a-z0-9][a-z0-9_.-]{0,63}"
_KIND_RE = re.compile(_KIND_PATTERN)


def parse_entity_registry(raw: Any) -> list[dict]:
    """Validate + compile the operator ``entity_registry`` config.

    Returns a list of compiled entries, each shaped:

        {"pattern_type": "name" | "regex",
         "pattern": str, "kind": str, "sensitivity": str,
         "regex": compiled re.Pattern | None}

    All offending entries are accumulated and raised in a single
    ``ValueError`` so the operator fixes everything in one pass rather
    than N register-reload cycles.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "entity_registry must be a list of entity entries, got "
            f"{type(raw).__name__}"
        )
    out: list[dict] = []
    errors: list[str] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(
                f"entity_registry[{i}] must be a dict, got {entry!r}"
            )
            continue
        ptype = entry.get("pattern_type")
        pattern = entry.get("pattern")
        kind = entry.get("kind")
        sensitivity = entry.get("sensitivity", "public")
        if ptype not in _ENTITY_PATTERN_TYPES:
            errors.append(
                f"entity_registry[{i}].pattern_type must be one of "
                f"{sorted(_ENTITY_PATTERN_TYPES)}, got {ptype!r}"
            )
            continue
        if not isinstance(pattern, str) or not pattern:
            errors.append(
                f"entity_registry[{i}].pattern must be a non-empty string, "
                f"got {pattern!r}"
            )
            continue
        if not isinstance(kind, str) or not kind:
            errors.append(
                f"entity_registry[{i}].kind must be a non-empty string, "
                f"got {kind!r}"
            )
            continue
        # HG-002 / HG-006 (hermaguard Phase 3): ``kind`` flows into the
        # confirmation marker and audit grouping keys, so restrict it to
        # a safe charset. This also blocks whitespace-only kinds.
        if not _KIND_RE.fullmatch(kind):
            errors.append(
                f"entity_registry[{i}].kind must match "
                f"{_KIND_PATTERN} (got {kind!r})"
            )
            continue
        if sensitivity not in VALID_LABELS:
            errors.append(
                f"entity_registry[{i}].sensitivity must be one of "
                f"{sorted(VALID_LABELS)}, got {sensitivity!r}"
            )
            continue
        compiled = None
        if ptype == "regex":
            if _regex_engine is None and _NESTED_QUANT_RE.search(pattern):
                # No timeout-capable engine installed: refuse the
                # classic nested-quantifier ReDoS shape outright.
                errors.append(
                    f"entity_registry[{i}].pattern uses a nested "
                    f"quantifier (ReDoS risk) and the optional 'regex' "
                    f"package is not installed; install 'regex' or "
                    f"simplify the pattern"
                )
                continue
            try:
                compiled = _compile_registry_regex(pattern)
            except TimeoutError:
                errors.append(
                    f"entity_registry[{i}].pattern timed out during "
                    f"validation compile (catastrophic pattern?)"
                )
                continue
            except Exception as exc:
                # re.error and regex.error are both named `error` in
                # their respective modules; catch engine-agnostically so
                # the timeout-honouring engine's syntax failures are
                # reported the same way as stdlib ones.
                if type(exc).__name__ == "error":
                    errors.append(
                        f"entity_registry[{i}].pattern is not a valid "
                        f"regex: {exc}"
                    )
                    continue
                raise
        out.append({
            "pattern_type": ptype,
            "pattern": pattern,
            "kind": kind,
            "sensitivity": sensitivity,
            "regex": compiled,
        })
    if errors:
        raise ValueError(
            "entity_registry has invalid entries: " + "; ".join(errors)
        )
    return out


def get_registry(cfg: dict | None) -> list[dict]:
    """Resolve + freeze the compiled entity registry under *cfg*.

    Cached on cfg under a private key so the per-call hot path is O(1):
    a configured plugin reads ``cfg["_entity_registry_frozen"]`` directly
    and never re-parses.

    Raises ``ValueError`` when the operator config is malformed (the
    caller — typically ``register()`` — decides whether to fail loud or
    continue with a fallback).
    """
    if cfg is None:
        return []
    frozen = cfg.get("_entity_registry_frozen")
    if frozen is not None:
        return frozen
    reg = parse_entity_registry(cfg.get("entity_registry"))
    cfg["_entity_registry_frozen"] = reg
    return reg


# ── Matching ───────────────────────────────────────────────────────────────


def _match_entry(text: str, entry: dict) -> list[str]:
    """Return every matched value (str) for *entry* against *text*.

    A regex entry can match multiple times in one string; we surface all
    of them so the audit's per-kind counts include every instance.
    A name entry is at most one match (literal substring).
    """
    if entry["pattern_type"] == "name":
        if entry["pattern"].lower() in text.lower():
            return [entry["pattern"]]
        return []
    # HG-001 (hermaguard Phase 3): bound the scan window. Registry
    # regexes are operator-supplied and may backtrack catastrophically,
    # so the entity scan never runs them against unbounded text.
    return [m.group(0) for m in
            entry["regex"].finditer(text[:_ENTITY_SCAN_MAX_CHARS])]


def extract_entities(value: Any, registry: list[dict]) -> list[dict]:
    """Return every entity match found in *value* against *registry*.

    *value* may be a ``str`` or a nested dict/list (the tool-args
    shape); the scan recurses so ``{"to": "alice", "meta": {"ref":
    "PR-7"}}`` is fully covered.

    HG-001 (hermaguard Phase 3): each scanned string is truncated to
    ``_ENTITY_SCAN_MAX_CHARS`` and total regex wall-clock time per
    call is bounded by ``_ENTITY_SCAN_BUDGET_S`` — a catastrophic
    registry pattern degrades to a partial (logged) result instead of
    hanging the rescue path or the store's global lock.

    HG-003 (hermaguard Phase 3): recursion depth is capped at
    ``_ENTITY_SCAN_MAX_DEPTH``; deeper nesting simply stops yielding
    new matches instead of raising RecursionError on the hot path.

    Each match is ``{"kind", "sensitivity", "value", "pattern_type"}``.
    Order is deterministic: registry order, then traversal order, with
    matches of the same registry entry emitted left-to-right within
    that string (the underlying ``str.find`` / ``re.search`` order).
    """
    if not registry:
        return []
    deadline = time.monotonic() + _ENTITY_SCAN_BUDGET_S
    try:
        return _extract_entities_inner(value, registry, deadline, 0)
    except _EntityScanBudgetExceeded:
        logger.warning(
            "toolaria: entity scan exceeded %.3fs budget; "
            "returning empty match set for this request",
            _ENTITY_SCAN_BUDGET_S,
        )
        return []


class _EntityScanBudgetExceeded(Exception):
    """Raised internally when the entity scan exceeds its time budget."""


def _extract_entities_inner(value: Any, registry: list[dict],
                            deadline: float, depth: int) -> list[dict]:
    if time.monotonic() > deadline:
        raise _EntityScanBudgetExceeded()
    if depth > _ENTITY_SCAN_MAX_DEPTH:
        return []
    if isinstance(value, str):
        out: list[dict] = []
        for entry in registry:
            if entry["pattern_type"] != "regex":
                # name matching is linear substring search; no budget risk
                for matched in _match_entry(value, entry):
                    out.append({
                        "kind": entry["kind"],
                        "sensitivity": entry["sensitivity"],
                        "value": matched,
                        "pattern_type": entry["pattern_type"],
                    })
                continue
            text = value[:_ENTITY_SCAN_MAX_CHARS]
            try:
                for m in entry["regex"].finditer(
                        text, timeout=_ENTITY_REGEX_TIMEOUT_S):
                    out.append({
                        "kind": entry["kind"],
                        "sensitivity": entry["sensitivity"],
                        "value": m.group(0),
                        "pattern_type": entry["pattern_type"],
                    })
            except TimeoutError:
                # `regex`-engine per-match timeout fired (HG-001):
                # treat this pattern as exhausted for this string and
                # keep scanning the rest of the registry.
                logger.warning(
                    "toolaria: entity registry regex for kind=%r timed "
                    "out mid-scan; pattern skipped for this value",
                    entry["kind"],
                )
            except TypeError:
                # stdlib `re` fallback: no timeout kwarg. The wall-clock
                # budget checked after this call is then best-effort.
                for m in entry["regex"].finditer(text):
                    out.append({
                        "kind": entry["kind"],
                        "sensitivity": entry["sensitivity"],
                        "value": m.group(0),
                        "pattern_type": entry["pattern_type"],
                    })
            if time.monotonic() > deadline:
                raise _EntityScanBudgetExceeded()
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(
                _extract_entities_inner(v, registry, deadline, depth + 1))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(
                _extract_entities_inner(v, registry, deadline, depth + 1))
        return out
    return []


def distinct_entity_kinds(matches: list[dict]) -> list[str]:
    """Return the sorted distinct ``kind`` values across *matches*.

    Used by ``put()`` (T3.1, stored on the index) and the ambiguity
    gate (T3.3, ``len(distinct) > 1`` ⇒ ambiguous). Sorting keeps the
    on-disk representation deterministic for JSONL diffs.
    """
    return sorted({m["kind"] for m in matches})
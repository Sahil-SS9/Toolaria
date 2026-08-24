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

import re
from typing import Any

from labels import VALID_LABELS


# pattern_type is the only discriminated field; everything else is
# plain metadata validated by type/format below.
_ENTITY_PATTERN_TYPES = frozenset({"name", "regex"})


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
        if sensitivity not in VALID_LABELS:
            errors.append(
                f"entity_registry[{i}].sensitivity must be one of "
                f"{sorted(VALID_LABELS)}, got {sensitivity!r}"
            )
            continue
        compiled = None
        if ptype == "regex":
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                errors.append(
                    f"entity_registry[{i}].pattern is not a valid regex: "
                    f"{exc}"
                )
                continue
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
    return [m.group(0) for m in entry["regex"].finditer(text)]


def extract_entities(value: Any, registry: list[dict]) -> list[dict]:
    """Return every entity match found in *value* against *registry*.

    *value* may be a ``str`` or a nested dict/list (the tool-args
    shape); the scan recurses so ``{"to": "alice", "meta": {"ref":
    "PR-7"}}`` is fully covered.

    Each match is ``{"kind", "sensitivity", "value", "pattern_type"}``.
    Order is deterministic: registry order, then traversal order, with
    matches of the same registry entry emitted left-to-right within
    that string (the underlying ``str.find`` / ``re.search`` order).
    """
    if not registry:
        return []
    if isinstance(value, str):
        out: list[dict] = []
        for entry in registry:
            for matched in _match_entry(value, entry):
                out.append({
                    "kind": entry["kind"],
                    "sensitivity": entry["sensitivity"],
                    "value": matched,
                    "pattern_type": entry["pattern_type"],
                })
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(extract_entities(v, registry))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(extract_entities(v, registry))
        return out
    return []


def distinct_entity_kinds(matches: list[dict]) -> list[str]:
    """Return the sorted distinct ``kind`` values across *matches*.

    Used by ``put()`` (T3.1, stored on the index) and the ambiguity
    gate (T3.3, ``len(distinct) > 1`` ⇒ ambiguous). Sorting keeps the
    on-disk representation deterministic for JSONL diffs.
    """
    return sorted({m["kind"] for m in matches})
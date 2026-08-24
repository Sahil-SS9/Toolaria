"""Best-effort JSONL append ledger for Toolaria audit trails.

Used by T1.5 (expansion audit) and reusable for any future per-event
audit trail. The helper is intentionally minimal: append one JSON record
as one line, never raise into the caller. A write failure logs and
returns False so the calling operation (expansion, fetch, etc.) is never
broken by audit-trail I/O.

Security: the ledger inherits the store's perms guarantees (0700 dirs,
0600 files). The store-side redaction in T1.3 is responsible for
scrubbing PII/secrets before records reach this helper.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _safe_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError as exc:
        logger.debug("toolaria: ledger chmod %s 0o%o failed: %s",
                     path, mode, exc)


def ledger_path(cfg: dict, kind: str = "expansions") -> Path:
    """Resolve the canonical ledger file path for *kind* under store_path.

    kinds: expansions (T1.5 audit ledger).
    """
    bp = Path(cfg.get("store_path", "~/.hermes/toolaria")).expanduser().resolve()
    return bp / "ledger" / f"{kind}.jsonl"


def append_line(path, record: dict) -> bool:
    """Append a single JSON line to *path*. Best-effort, never raises.

    Returns True on success, False on any error. A write failure must
    not break the calling operation — expansion, fetch, etc. continue
    with the content they already had.

    HG-004 (hermaguard Phase 3): before appending, the ledger is
    size-checked and rotated at ``_LEDGER_ROTATE_BYTES`` into
    ``<name>.jsonl.1`` (single previous generation). This bounds disk
    growth on hot paths without unbounded read amplification.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Tighten the directory on first touch; idempotent.
        _safe_chmod(p.parent, _DIR_MODE)
        _rotate_if_large(p)
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
        _safe_chmod(p, _FILE_MODE)
        return True
    except Exception as exc:
        logger.warning("toolaria: ledger append %s failed: %s", path, exc)
        return False


# HG-004: rotate a ledger when it exceeds this many bytes. The old
# file moves to <name>.jsonl.1 (one generation kept), so worst case
# on-disk is ~2x the cap.
_LEDGER_ROTATE_BYTES = 16 * 1024 * 1024


def _rotate_if_large(p: Path) -> None:
    """Rotate *p* to ``p.with_suffix(p.suffix + '.1')`` when oversized."""
    try:
        if p.exists() and p.stat().st_size > _LEDGER_ROTATE_BYTES:
            rotated = p.with_suffix(p.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            p.rename(rotated)
            logger.info("toolaria: rotated oversized ledger %s", p.name)
    except OSError as exc:
        logger.debug("toolaria: ledger rotation %s skipped: %s",
                     p.name, exc)


def log_entity_binding(cfg: dict, *, sid: str, tool: str,
                       decision: str, entity_kind: str | None = None,
                       entity_kinds: list[str] | None = None,
                       blob_id: str | None = None) -> bool:
    """Write one T3.2 action→entity binding record (observe-only).

    ``decision`` ∈ ``entity_bound`` (T3.2: a registered entity was
    mentioned alongside a tla:<id> blob reference) | ``ambiguous_gated``
    (T3.3: the ambiguity gate fired for this request).

    Either ``entity_kind`` (single, for ``entity_bound``) or
    ``entity_kinds`` (sorted list, for ``ambiguous_gated``) is set;
    unused fields are omitted so JSONL diffs stay minimal. A write
    failure is logged at WARNING and returns False — the passref path
    must never raise into expansion.
    """
    record: dict = {
        "ts": time.time(),
        "sid": sid,
        "tool": tool,
        "decision": decision,
    }
    if entity_kind is not None:
        record["entity_kind"] = entity_kind
    if entity_kinds:
        record["entity_kinds"] = sorted(entity_kinds)
    if blob_id is not None:
        record["blob_id"] = blob_id
    path = ledger_path(cfg, "entity_bindings")
    if append_line(path, record):
        logger.info(
            "toolaria: entity binding tool=%s decision=%s kind=%s",
            tool, decision, entity_kind,
        )
        return True
    return False


def log_expansion(cfg: dict, *, sid: str, blob_id: str, dst_tool: str,
                  chars: int, decision: str,
                  label: str | None = None) -> bool:
    """Write one T1.5 expansion record and log it at INFO.

    decision ∈ expanded | dest_denied | session_denied | missing | budget_capped
                 | credential_denied (T2.3)

    Returns True if the line was written. The INFO log fires only on
    successful append so a failing audit trail stays quiet rather than
    flooding logs.
    """
    record = {
        "ts": time.time(),
        "sid": sid,
        "blob_id": blob_id,
        "dst_tool": dst_tool,
        "chars": chars,
        "decision": decision,
    }
    # T2.2 / T2.3: per-row label. ``None`` is omitted so pre-T2.2 rows
    # (no label field) still parse cleanly; the audit script falls back
    # to index lookup for label-less rows.
    if label is not None:
        record["label"] = label
    path = ledger_path(cfg, "expansions")
    if append_line(path, record):
        logger.info(
            "toolaria: expansion ledger blob=%s dst=%s chars=%d decision=%s"
            "%s",
            blob_id, dst_tool, chars, decision,
            f" label={label}" if label else "",
        )
        return True
    return False


def log_key_rotation(cfg: dict, *, count: int,
                     old_key_file: str, new_key_file: str) -> bool:
    """T4.3: write one key-rotation ledger row per successful pass.

    The row carries the count of blobs that were re-encrypted, plus
    the source/destination key file paths so an operator can audit
    which key was active at which moment. Lives in
    ``ledger/key_rotations.jsonl`` so it is discoverable alongside the
    expansion and entity-binding ledgers.

    Best-effort, never raises — a failed audit append must not break
    the rotation itself, which has already mutated on-disk bytes by
    this point.
    """
    record = {
        "ts": time.time(),
        "decision": "key_rotated",
        "count": int(count),
        "old_key_file": old_key_file,
        "new_key_file": new_key_file,
    }
    path = ledger_path(cfg, "key_rotations")
    if append_line(path, record):
        logger.info(
            "toolaria: key rotation complete; %d blob(s) re-encrypted "
            "under %s (was %s)", count, new_key_file, old_key_file,
        )
        return True
    return False

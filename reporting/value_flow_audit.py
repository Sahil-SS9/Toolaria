"""Offline value-flow audit over Toolaria's expansion ledger + index labels.

Reads the JSONL expansion ledger (``store_path/ledger/expansions.jsonl``)
and the session indexes (``store_path/sessions/*.json``) to produce a
per-destination-tool breakdown of:

  * which sensitivity labels flowed where
  * how many chars per label per destination
  * how many expansions were denied (and at which destination class)

Designed to mirror ``reporting/reacquisition_report.py`` — purely
read-only, safe to run against a live ``~/.hermes/toolaria`` tree.

Usage:
    python -m reporting.value_flow_audit <store_path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ── IO ────────────────────────────────────────────────────────────────────


def load_sessions(store_path):
    """Yield every blob entry indexed by ``(safe_sid, blob_id)``.

    Returns a plain ``dict`` whose keys are 2-tuples — the type hints
    stay loose because Pyright treats dict-key invariance strictly and
    the audit callers always do a single fetch-or-scan.
    """
    sessions_dir = Path(store_path) / "sessions"
    if not sessions_dir.is_dir():
        return {}
    out: dict = {}
    for ip in sorted(sessions_dir.glob("*.json")):
        try:
            idx = json.loads(ip.read_text())
        except Exception:
            continue
        for bid, meta in idx.get("blobs", {}).items():
            out[(ip.stem, bid)] = meta
    return out


def load_ledger(store_path) -> list[dict]:
    """Yield events from the T1.5 expansion ledger."""
    p = Path(store_path) / "ledger" / "expansions.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ── T3.4: entity bindings ─────────────────────────────────────────────────


def load_entity_bindings(store_path) -> list[dict]:
    """Yield events from the T3.2 entity_bindings.jsonl ledger.

    Best-effort: a missing file or malformed line is skipped so a
    pre-T3 store produces an empty list rather than crashing the
    audit run.
    """
    p = Path(store_path) / "ledger" / "entity_bindings.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_entity_binding_summary(rows: list[dict]) -> dict:
    """Compute the T3.4 entity-binding summary from raw ledger rows.

    Returns
    -------
    dict
        ``per_kind``      — ``{kind: count}`` for ``entity_bound`` rows
        ``ambiguous_total`` — total ``ambiguous_gated`` rows
        ``ambiguous_by_tool`` — ``{tool: count}`` for ambiguous rows
        ``total``         — total ``entity_bound`` rows
    """
    per_kind: dict[str, int] = {}
    ambiguous_total = 0
    ambiguous_by_tool: dict[str, int] = {}
    bound_total = 0
    for r in rows:
        decision = r.get("decision", "")
        if decision == "entity_bound":
            bound_total += 1
            kind = r.get("entity_kind")
            if kind:
                per_kind[kind] = per_kind.get(kind, 0) + 1
        elif decision == "ambiguous_gated":
            ambiguous_total += 1
            tool = r.get("tool") or "(unknown)"
            ambiguous_by_tool[tool] = ambiguous_by_tool.get(tool, 0) + 1
    return {
        "per_kind": per_kind,
        "ambiguous_total": ambiguous_total,
        "ambiguous_by_tool": ambiguous_by_tool,
        "total": bound_total,
    }


# ── T4.4 encryption coverage (Fernet-at-rest counters) ──────────────────


def build_encryption_coverage(entries: dict) -> dict:
    """T4.4: tally credential blobs by at-rest encryption state.

    Counts come from index entries (live + tombstones) so historical
    coverage survives sweep. Every label=='credential' blob is
    classified as either encrypted (entry carries ``enc: True``) or
    plaintext (the inert default). Public / personal / internal blobs
    are tallied under ``public`` for context (they're never
    encrypted at rest under the T4.2 contract, but the audit report
    surfaces the totals so an operator can sanity-check the
    classifier).

    Returns a dict with three integer counters. The renderer omits
    the section when ``encrypted + plaintext_credential + public ==
    0`` so a pre-T4.2 store produces a clean report.
    """
    encrypted = 0
    plaintext_credential = 0
    public = 0
    for meta in entries.values():
        if not isinstance(meta, dict):
            continue
        label = meta.get("label", "public")
        if label == "credential":
            if meta.get("enc") is True:
                encrypted += 1
            else:
                plaintext_credential += 1
        else:
            public += 1
    return {
        "encrypted": encrypted,
        "plaintext_credential": plaintext_credential,
        "public": public,
    }


# ── T4.4 version-chain stats (T4.1) ────────────────────────────────────────


def build_version_chain_stats(entries: dict) -> dict:
    """T4.4: compute max chain depth + superseded count from index
    entries.

    A "chain" is the set of entries sharing a (tool, session) group
    linked by ``supersedes`` / ``superseded_by`` pointers. The
    ``max_depth`` is the longest chain anywhere in the store;
    ``superseded_count`` is the number of entries that are NOT the
    head of their chain (i.e. have a successor).

    Both fields are always present; callers may treat depth=1
    superseded_count=0 as "no chains".

    Parameters
    ----------
    entries : dict[(safe_sid, bid) -> meta]
        The session-index entries dict produced by ``load_sessions``.
        Grouping by (safe_sid, tool) approximates the (tool, session)
        chain boundary since entries from different sessions are
        bucketed under different ``safe_sid`` values.
    """
    # Group entries by (safe_sid, tool). Within each group, walk the
    # supersedes/supereded_by graph to find the head of each chain,
    # then count its length.
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for (safe_sid, bid), meta in entries.items():
        if not isinstance(meta, dict):
            continue
        tool = meta.get("tool", "")
        groups.setdefault((safe_sid, tool), {})[bid] = meta

    max_depth = 1  # a single-entry chain is depth 1
    superseded_count = 0
    for group in groups.values():
        # Build a successor map: bid -> bid whose ``supersedes`` points
        # at the key. Walks forward from a head to its successor even
        # when the predecessor doesn't carry ``superseded_by`` (which
        # is the shape that survives serialization through older
        # indexes and through test fixtures built without the
        # backward-pointer).
        successor: dict[str, str] = {}
        for bid, meta in group.items():
            if not isinstance(meta, dict):
                continue
            prev = meta.get("supersedes")
            if prev and prev in group:
                successor[prev] = bid

        # Heads: entries that do NOT supersede anything else. These are
        # either singletons (no predecessor nor successor) or the
        # oldest link of a chain.
        heads = [bid for bid, meta in group.items()
                 if isinstance(meta, dict) and "supersedes" not in meta]
        seen: set[str] = set()
        for head_bid in heads:
            if head_bid in seen:
                continue
            depth = 0
            cur = head_bid
            for _ in range(1024):  # defensive upper bound on chain length
                if cur in seen:
                    break
                meta = group.get(cur)
                if meta is None:
                    break
                seen.add(cur)
                depth += 1
                nxt = successor.get(cur)
                if not nxt:
                    break
                cur = nxt
            if depth > max_depth:
                max_depth = depth
        # Count superseded: every entry that supersedes a predecessor
        # is a non-head chain link (the head has no ``supersedes``).
        for meta in group.values():
            if isinstance(meta, dict) and "supersedes" in meta:
                superseded_count += 1

    return {
        "max_depth": max_depth,
        "superseded_count": superseded_count,
    }


# ── Stats ─────────────────────────────────────────────────────────────────


def _empty_dst_entry() -> dict:
    return {
        "expanded_total": 0,
        "chars_total": 0,
        "denied": 0,
        "credential": 0,
        "personal": 0,
        "internal": 0,
        "public": 0,
        "credential_chars": 0,
        "personal_chars": 0,
        "internal_chars": 0,
        "public_chars": 0,
    }


def build_report(store_path) -> dict:
    """Compute the value-flow report from on-disk state.

    Each ledger row contributes to ``by_destination[dst_tool]`` under
    its decision category:
      - decision='expanded' per-label counters + char counts
      - decision in (dest_denied, session_denied, budget_capped)
        increments the 'denied' counter
      - decision='missing' is bucketed with 'denied' too — the
        destination never saw the bytes either way
    """
    # Index lookup keyed by (safe_sid, blob_id) so label fallback works
    # even when the ledger row does not carry a label field (rows from
    # before T2.2 will not).
    entries = load_sessions(store_path)
    ledger = load_ledger(store_path)

    by_destination: dict[str, dict] = {}
    labels_observed: set[str] = set()
    expansions_total = 0
    denied_total = 0
    label_missing = 0

    for row in ledger:
        dst = row.get("dst_tool", "") or "(unknown)"
        decision = row.get("decision", "")
        # Best-effort label resolution. Ledger rows from before T2.2 do
        # not carry a label; look it up from the session index, fall back
        # to "unknown".
        label = row.get("label")
        if not label:
            sid = row.get("sid", "")
            # Map sid to on-disk safe_sid; the index file is named by
            # safe_sid, not raw sid. We can't reverse that mapping here
            # without re-running _safe_sid, so first try the raw sid, then
            # scan indexes for a matching blob_id.
            key = (sid, row.get("blob_id", ""))
            meta = entries.get(key)
            if meta:
                label = meta.get("label", "unknown")
            else:
                # Last resort: scan for the blob_id under any session.
                for (s, b), m in entries.items():
                    if b == row.get("blob_id"):
                        label = m.get("label", "unknown")
                        break
                else:
                    label = "unknown"
        if label == "unknown":
            label_missing += 1
        else:
            labels_observed.add(label)

        bucket = by_destination.setdefault(dst, _empty_dst_entry())
        if decision == "expanded":
            bucket["expanded_total"] += 1
            bucket["chars_total"] += int(row.get("chars", 0) or 0)
            if label in ("credential", "personal", "internal", "public"):
                bucket[label] += 1
                bucket[f"{label}_chars"] += int(row.get("chars", 0) or 0)
            expansions_total += 1
        elif decision in ("dest_denied", "session_denied", "missing",
                          "budget_capped"):
            bucket["denied"] += 1
            denied_total += 1

    return {
        "store_path": str(store_path),
        "expansions_total": expansions_total,
        "denied_total": denied_total,
        "labels_observed": sorted(labels_observed),
        "label_missing": label_missing,
        "by_destination": dict(sorted(by_destination.items())),
        "ledger_rows": len(ledger),
        # T3.4: entity-binding summary, read from entity_bindings.jsonl.
        # Empty block (per_kind={}, ambiguous_total=0) when no rows exist
        # so downstream renderers can branch on total == 0 cleanly.
        "entity_bindings": build_entity_binding_summary(
            load_entity_bindings(store_path)),
        # T4.4: encryption-coverage counters derived from index
        # entries. Three integers — encrypted, plaintext_credential,
        # public. The renderer omits the section when ALL three are
        # zero so a pre-T4.2 store produces a clean report.
        "encryption_coverage": build_encryption_coverage(entries),
        # T4.4: version-chain stats (T4.1). max_depth is the longest
        # chain anywhere; superseded_count is the number of non-head
        # entries (every chain link except the head). Always present;
        # a single-entry chain yields max_depth=1 superseded_count=0.
        "version_chain_stats": build_version_chain_stats(entries),
    }


def render(report: dict) -> str:
    """Format the value-flow report as human-readable text."""
    lines: list[str] = []
    sp = report.get("store_path", "?")
    lines.append(f"Toolaria value-flow audit — {sp}")
    lines.append(f"  ledger rows:          {report['ledger_rows']}")
    lines.append(f"  expansions total:     {report['expansions_total']}")
    lines.append(f"  denied total:         {report['denied_total']}")
    if report["labels_observed"]:
        lines.append(f"  labels observed:      "
                     f"{', '.join(report['labels_observed'])}")
    if report["label_missing"]:
        lines.append(
            f"  rows with no label:   {report['label_missing']}  "
            f"(pre-T2.2 rows; look up index by blob_id)"
        )
    if not report["expansions_total"] and not report["denied_total"]:
        lines.append("  no data")
    by_dst = report["by_destination"]
    if by_dst:
        lines.append("  by destination tool:")
        for dst, b in by_dst.items():
            lines.append(
                f"    {dst}: expanded={b['expanded_total']} "
                f"chars={b['chars_total']:,} denied={b['denied']}"
            )
            label_parts = []
            for label in ("credential", "personal", "internal", "public"):
                if b.get(label):
                    label_parts.append(
                        f"{label}={b[label]} ({b.get(f'{label}_chars', 0):,}c)"
                    )
            if label_parts:
                lines.append("      labels: " + ", ".join(label_parts))
    # T3.4: entity-binding summary (per-kind + top ambiguous tools).
    # The block is omitted when the ledger is empty so a pre-T3 store
    # does not gain a noisy header.
    eb = report.get("entity_bindings") or {}
    eb_total = eb.get("total", 0)
    ambig_total = eb.get("ambiguous_total", 0)
    if eb_total or ambig_total:
        lines.append("  entity bindings (T3.4):")
        if eb_total:
            per_kind = eb.get("per_kind", {})
            kind_parts = [
                f"{k}={v}" for k, v in sorted(per_kind.items())
            ]
            lines.append(
                f"    bound total: {eb_total}  "
                + (", ".join(kind_parts) if kind_parts else "(no kinds)")
            )
        if ambig_total:
            lines.append(f"    ambiguous_gated total: {ambig_total}")
            by_tool = sorted(
                eb.get("ambiguous_by_tool", {}).items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
            for tool, count in by_tool:
                lines.append(f"      top ambiguous: {tool} ({count})")
    # T4.4: encryption-coverage section. Omitted when no credential
    # blobs at all (encrypted + plaintext_credential == 0) so a
    # pre-T4.2 / plaintext-only store does not gain a noisy header.
    # The public counter is included when non-zero so an operator
    # can sanity-check the classifier without flipping between two
    # reports.
    enc = report.get("encryption_coverage") or {}
    enc_n = int(enc.get("encrypted", 0))
    plc_n = int(enc.get("plaintext_credential", 0))
    pub_n = int(enc.get("public", 0))
    if enc_n or plc_n:
        lines.append("  encryption coverage (T4.4):")
        lines.append(
            f"    credential blobs at rest: encrypted={enc_n} "
            f"plaintext={plc_n}"
        )
        if pub_n:
            lines.append(f"    non-credential blobs: {pub_n}")
    # T4.4: version-chain stats (T4.1). Omitted when nothing
    # superseded (single-entry chains are not interesting).
    vcs = report.get("version_chain_stats") or {}
    superseded_n = int(vcs.get("superseded_count", 0))
    if superseded_n > 0:
        lines.append("  version chains (T4.1 audit surface):")
        lines.append(
            f"    max chain depth: {vcs.get('max_depth', 1)}  "
            f"superseded versions: {superseded_n}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m reporting.value_flow_audit <store_path>",
            file=sys.stderr,
        )
        return 2
    report = build_report(Path(argv[0]))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

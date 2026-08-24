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

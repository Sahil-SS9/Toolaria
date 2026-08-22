"""Offline reacquisition metrics over Toolaria session indexes.

Reads ``store_path/sessions/*.json`` (and the optional T1.2 sequence
sidecar) to compute:

  * put→first-fetch latency p50 / p95 (seconds)
  * fetches-per-blob histogram
  * turn-aware metrics when the sequence sidecar carries turn counters

Latency is a **wall-clock proxy** unless turn-position data is present
in the sequence sidecar (Phase 1 ships without a host turn counter, so
this is the common case). The report labels the proxy explicitly so
operators are not misled into thinking the numbers reflect agent turns.

Usage:
    python -m reporting.reacquisition_report <store_path>

The script never mutates the store; it is purely read-only and safe to
run against a live ``~/.hermes/toolaria`` tree.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ── IO ────────────────────────────────────────────────────────────────────


def load_sessions(store_path) -> list[dict]:
    """Yield every live blob entry from every session index."""
    sessions_dir = Path(store_path) / "sessions"
    if not sessions_dir.is_dir():
        return []
    out: list[dict] = []
    for ip in sorted(sessions_dir.glob("*.json")):
        try:
            idx = json.loads(ip.read_text())
        except Exception:
            continue
        for bid, meta in idx.get("blobs", {}).items():
            if "swept_at" in meta:
                continue
            out.append({"blob_id": bid, **meta})
    return out


def load_sequences(store_path) -> list[dict]:
    """Yield events from the optional T1.2 sequence sidecar.

    The file is JSONL with ``{ts, sid, blob_id, turn?}``. Missing file
    means the capture flag is off or never enabled — both are valid
    operational states.
    """
    p = Path(store_path) / "sequences" / "events.jsonl"
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


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile. NaN on empty input."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


# ── Report ────────────────────────────────────────────────────────────────


def build_report(store_path) -> dict:
    """Compute the metrics report from on-disk state."""
    entries = load_sessions(store_path)
    sequences = load_sequences(store_path)

    latencies: list[float] = []
    counts: list[int] = []
    for e in entries:
        t = e.get("t")
        first = e.get("first_fetch_ts")
        cnt = e.get("fetch_count", 0) or 0
        counts.append(cnt)
        # Latency only meaningful when first-fetch came after put.
        if t is not None and first is not None and first >= t:
            latencies.append(first - t)

    histogram: dict[int, int] = {}
    for c in counts:
        histogram[c] = histogram.get(c, 0) + 1

    has_turns = any("turn" in ev for ev in sequences)

    return {
        "store_path": str(store_path),
        "blobs_total": len(entries),
        "blobs_with_first_fetch": len(latencies),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "fetch_count_histogram": dict(sorted(histogram.items())),
        "sequences_events": len(sequences),
        "sequences_have_turns": has_turns,
        "mode": "turn-aware" if has_turns else "wall-clock proxy",
    }


def render(report: dict) -> str:
    """Format the report as human-readable text."""
    lines: list[str] = []
    lines.append(f"Toolaria reacquisition report — {report['store_path']}")
    lines.append(f"  blobs (live):                     {report['blobs_total']}")
    lines.append(
        f"  blobs with first-fetch timestamp: {report['blobs_with_first_fetch']}"
    )
    p50 = report["latency_p50_s"]
    p95 = report["latency_p95_s"]
    if math.isnan(p50):
        lines.append("  put→first-fetch latency p50:      (no data)")
        lines.append("  put→first-fetch latency p95:      (no data)")
    else:
        lines.append(f"  put→first-fetch latency p50 (s): {p50:.3f}")
        lines.append(f"  put→first-fetch latency p95 (s): {p95:.3f}")
    lines.append(f"  mode: {report['mode']}")
    if not report["sequences_have_turns"]:
        lines.append(
            "  note: latency is a wall-clock proxy (no turn-position data; "
            "enable T1.2 sequence capture and forward turn counters from "
            "the host for turn-accurate metrics)"
        )
    lines.append(
        f"  sequence events (optional):      {report['sequences_events']}"
    )
    hist = report["fetch_count_histogram"]
    if hist:
        lines.append("  fetches-per-blob histogram:")
        for k in sorted(hist):
            plural = "es" if k != 1 else ""
            lines.append(f"    {k:>3} fetch{plural}: {hist[k]}")
    else:
        lines.append("  fetches-per-blob histogram: (no data)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m reporting.reacquisition_report <store_path>",
            file=sys.stderr,
        )
        return 2
    report = build_report(Path(argv[0]))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

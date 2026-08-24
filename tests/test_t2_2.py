"""Phase 2 data-governance tests — T2.2.

TDD discipline: every test was written before the implementation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from reporting import value_flow_audit as _vfa
from blobstore import BlobStore



class TestValueFlowAuditScript:
    """reporting/value_flow_audit.py — offline reader like reacquisition_report."""

    def _write_index(self, store_path: Path, sid: str, blobs: dict) -> None:
        sessions = store_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        from blobstore import BlobStore
        safe = BlobStore._safe_sid(sid)
        (sessions / f"{safe}.json").write_text(
            json.dumps({"blobs": blobs}, indent=2)
        )

    def _write_ledger(self, store_path: Path, rows: list[dict]) -> None:
        ledger_dir = store_path / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        with open(ledger_dir / "expansions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_script_exists_and_imports(self):
        # Sanity: the module is importable.
        assert hasattr(_vfa, "build_report")
        assert hasattr(_vfa, "render")

    def test_empty_store_returns_empty_report(self, tmp_path):
        sp = tmp_path / "audit_store_empty"
        report = _vfa.build_report(sp)
        assert report["expansions_total"] == 0
        assert report["denied_total"] == 0
        assert report["by_destination"] == {}
        assert report["ledger_rows"] == 0
        # And render() does not crash.
        text = _vfa.render(report)
        assert "no data" in text

    def test_per_destination_tool_breakdown(self, tmp_path):
        """The headline metric: which destination tools consumed what
        labels, and how many chars each."""
        sp = tmp_path / "audit_store"
        # Two sessions with labelled blobs.
        self._write_index(sp, "s1", {
            "aaa111222333": {"t": 1.0, "tool": "send_email", "size": 100,
                             "hash": "h" * 8, "label": "personal"},
            "bbb444555666": {"t": 1.0, "tool": "web_search", "size": 200,
                             "hash": "h" * 8, "label": "public"},
        })
        # Ledger: one expansion into summarise (public), one into
        # publish (personal), and one denied expansion.
        self._write_ledger(sp, [
            {"ts": 2.0, "sid": "s1", "blob_id": "aaa111222333",
             "dst_tool": "publish", "chars": 100, "decision": "expanded",
             "label": "personal"},
            {"ts": 3.0, "sid": "s1", "blob_id": "bbb444555666",
             "dst_tool": "summarise", "chars": 200, "decision": "expanded",
             "label": "public"},
            {"ts": 4.0, "sid": "s1", "blob_id": "aaa111222333",
             "dst_tool": "send_email", "chars": 0,
             "decision": "dest_denied", "label": "personal"},
        ])
        report = _vfa.build_report(sp)
        by_dst = report["by_destination"]
        assert "publish" in by_dst
        assert by_dst["publish"]["personal"] == 1
        assert by_dst["publish"]["personal_chars"] == 100
        assert by_dst["summarise"]["public"] == 1
        # Denials are counted under the destination that was denied.
        assert by_dst["send_email"].get("denied", 0) == 1
        # Expanded vs denied totals match the ledger semantics.
        assert report["expansions_total"] == 2
        assert report["denied_total"] == 1
        assert report["ledger_rows"] == 3
        assert set(report["labels_observed"]) >= {"public", "personal"}

    def test_audit_mode_in_rescuer_fetch_enum(self, plugin):
        fc, _ = plugin
        enum = fc.tools["rescuer_fetch"]["schema"]["parameters"]["properties"]["mode"]["enum"]
        assert "audit" in enum, (
            "T2.2: rescuer_fetch must expose audit mode so models / "
            "operators can pull the last-N expansion summary"
        )

    def test_audit_mode_returns_ledger_summary(self, plugin, toolaria):
        """audit mode reads ledger JSONL and returns a compact summary."""
        # Generate some expansion events so the ledger has data.
        bid = toolaria._store.put("X " * 100, "web_search",
                                   session_id="test-s")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        r = toolaria._fetch(args={"id": bid, "mode": "audit", "count": 5},
                             session_id="test-s")
        assert "audit" in r.lower() or "expansion" in r.lower(), (
            f"audit mode must produce a readable summary; got: {r!r}"
        )
        assert "summarise" in r, (
            "audit summary must surface the destination tool that received "
            "the expansion so operators can see the flow"
        )

    def test_audit_mode_is_read_only(self, plugin, toolaria):
        """audit mode must NOT mutate store state.

        G3 invariant: read-only fetch modes are a contractual property —
        a regression that begins writing on fetch would corrupt sweep
        semantics and silently inflate counters.
        """
        bid = toolaria._store.put("Y " * 50, "web_search",
                                   session_id="test-s")
        before = toolaria._store._load_idx("test-s")["blobs"][bid].copy()
        toolaria._fetch(args={"id": bid, "mode": "audit", "count": 10},
                         session_id="test-s")
        after = toolaria._store._load_idx("test-s")["blobs"][bid]
        # Only fetch_count and first_fetch_ts may change on a normal
        # fetch; audit mode must not bump them either (it's a status
        # query, not a content fetch).
        assert after.get("fetch_count", 0) == before.get("fetch_count", 0), (
            "audit mode must NOT bump fetch_count — it's a read-only "
            "status query, not a true fetch"
        )

    def test_audit_mode_handles_empty_ledger(self, plugin, toolaria):
        """No ledger yet → audit mode returns the empty-report text."""
        bid = toolaria._store.put("data", "web_search", session_id="test-s")
        r = toolaria._fetch(args={"id": bid, "mode": "audit", "count": 10},
                             session_id="test-s")
        assert "no data" in r.lower() or "0 expansion" in r.lower()



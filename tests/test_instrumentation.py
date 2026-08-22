"""Phase 1 instrumentation tests (T1.1 – T1.5).

T1.1 tests live in this file; T1.2–T1.5 tests are appended by their
respective commits. Tests were written before the implementation and
witnessed RED; this file is the executable specification of Phase 1.
"""
import json
import time
from pathlib import Path

import pytest

import blobstore as _blobstore_mod
from blobstore import BlobStore
from reporting import reacquisition_report as _rr


# ═══ T1.1 — first_fetch_ts + fetch_count persisted on fetch ══════════════


class TestFirstFetchAndCountPersistence:
    """put→first-fetch latency (T1.1) is computed from the index."""

    def test_first_fetch_ts_recorded_on_first_fetch(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert "first_fetch_ts" in entry, (
            "first_fetch_ts must be persisted on the first fetch so the "
            "report can compute put→first-fetch latency without waiting "
            "for a sweep"
        )
        assert isinstance(entry["first_fetch_ts"], float)

    def test_first_fetch_ts_set_only_once(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        first = toolaria._store._load_idx("s1")["blobs"][bid]["first_fetch_ts"]
        time.sleep(0.01)
        toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        again = toolaria._store._load_idx("s1")["blobs"][bid]["first_fetch_ts"]
        assert first == again, (
            "first_fetch_ts must be set once on the first observation; "
            "subsequent fetches must not move it"
        )

    def test_fetch_count_increments_per_fetch(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        for i in range(5):
            toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry["fetch_count"] == 5, (
            f"fetch_count must be 5 after 5 fetches, got {entry.get('fetch_count')}"
        )

    def test_fetch_count_survives_restart_without_sweep(self, plugin, toolaria):
        """The count is now persisted on fetch, not just on sweep.

        Closing the store and re-opening it without an intervening sweep
        must not reset the count — the report depends on it being durable."""
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        for _ in range(3):
            toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        persisted = toolaria._store._load_idx("s1")["blobs"][bid]["fetch_count"]
        cfg = toolaria._store.cfg
        fresh = BlobStore(cfg)  # triggers _load_fetch_log
        # Index is the source of truth, read it back from disk.
        idx = fresh._read_idx_file(fresh._idx_path("s1"))
        assert idx["blobs"][bid]["fetch_count"] == persisted
        assert idx["blobs"][bid]["first_fetch_ts"] == \
            toolaria._store._load_idx("s1")["blobs"][bid]["first_fetch_ts"]


class TestReacquisitionReport:
    """reporting/reacquisition_report.py — golden-file math + edges."""

    def _write_index(self, store_path: Path, sid: str, blobs: dict) -> None:
        sessions = store_path / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        # Match the real naming scheme: _safe_sid.
        safe = BlobStore._safe_sid(sid)
        (sessions / f"{safe}.json").write_text(
            json.dumps({"blobs": blobs}, indent=2)
        )

    def test_p50_p95_from_synthetic_timestamps(self, plugin, toolaria, tmp_path):
        """Golden-file math: known put/first times ⇒ known p50/p95."""
        sp = tmp_path / "report_store"
        # Build three blobs with deterministic put→first-fetch gaps.
        gaps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        blobs = {}
        t0 = 1_000_000.0
        for i, gap in enumerate(gaps):
            bid = f"{i:012x}"
            blobs[bid] = {
                "t": t0,
                "tool": "web_search",
                "size": 100,
                "hash": "deadbeef" * 8,
                "first_fetch_ts": t0 + gap,
                "fetch_count": 1,
            }
        self._write_index(sp, "synthetic", blobs)
        report = _rr.build_report(sp)
        # Linear-interpolated percentiles: p50 ≈ 5.5, p95 ≈ 9.55.
        assert abs(report["latency_p50_s"] - 5.5) < 1e-6
        assert abs(report["latency_p95_s"] - 9.55) < 1e-6
        assert report["blobs_with_first_fetch"] == 10

    def test_histogram_counts_per_blob(self, plugin, toolaria, tmp_path):
        sp = tmp_path / "report_store"
        blobs = {}
        for i, count in enumerate([0, 0, 1, 3, 7]):
            bid = f"{i:012x}"
            entry = {"t": 1.0, "tool": "x", "size": 1, "hash": "h" * 8,
                     "fetch_count": count}
            if count > 0:
                entry["first_fetch_ts"] = 2.0
            blobs[bid] = entry
        self._write_index(sp, "s", blobs)
        hist = _rr.build_report(sp)["fetch_count_histogram"]
        assert hist == {0: 2, 1: 1, 3: 1, 7: 1}

    def test_empty_store_returns_empty_report_not_crash(
        self, plugin, toolaria, tmp_path
    ):
        sp = tmp_path / "empty_store"
        # Store dir does not even exist.
        report = _rr.build_report(sp)
        assert report["blobs_total"] == 0
        assert report["blobs_with_first_fetch"] == 0
        # render() must still produce readable text.
        text = _rr.render(report)
        assert "no data" in text

    def test_missing_fields_treated_as_missing_not_crash(
        self, plugin, toolaria, tmp_path
    ):
        """Live-format indexes from older installs have no first_fetch_ts
        or fetch_count. The report must treat them as missing data and
        emit the empty-report text, not raise."""
        sp = tmp_path / "old_store"
        self._write_index(sp, "legacy", {
            "abc123456789": {"t": 1.0, "tool": "x", "size": 1, "hash": "h" * 8},
        })
        report = _rr.build_report(sp)
        assert report["blobs_total"] == 1
        assert report["blobs_with_first_fetch"] == 0
        # NaN serialisation: the render() must not blow up.
        text = _rr.render(report)
        assert "no data" in text

    def test_tombstoned_entries_excluded(self, plugin, toolaria, tmp_path):
        sp = tmp_path / "tomb_store"
        self._write_index(sp, "x", {
            "abc123456789": {
                "t": 1.0, "tool": "x", "size": 1, "hash": "h" * 8,
                "first_fetch_ts": 2.0, "fetch_count": 1,
            },
            "def456789012": {
                "swept_at": time.time(), "tool": "x", "size": 1,
            },
        })
        report = _rr.build_report(sp)
        assert report["blobs_total"] == 1, (
            "tombstoned blobs must not pollute the report's live counts"
        )

    def test_render_labels_wall_clock_proxy_explicitly(
        self, plugin, toolaria, tmp_path
    ):
        sp = tmp_path / "wc_store"
        self._write_index(sp, "x", {
            "abc123456789": {
                "t": 1.0, "tool": "x", "size": 1, "hash": "h" * 8,
                "first_fetch_ts": 2.0, "fetch_count": 1,
            },
        })
        text = _rr.render(_rr.build_report(sp))
        assert "wall-clock proxy" in text, (
            "report must label latency as wall-clock proxy when no turn "
            "data is present, so operators are not misled"
        )


# ═══ config defaults + integration (Phase 1 infrastructure) ═════════════


class TestConfigAndIntegration:
    """Defaults + the new keys actually land in the merged cfg."""

    def test_config_exposes_new_keys(self):
        """config.yaml must include the Phase 1 instrumentation keys."""
        raw = (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
        assert "sequence_capture:" in raw
        assert "args_snapshot_max_chars:" in raw
        assert "verify_integrity:" in raw

    def test_default_values(self, toolaria, base_cfg, fake_ctx_cls):
        cfg = dict(base_cfg)
        for k in ("sequence_capture", "args_snapshot_max_chars",
                  "verify_integrity"):
            cfg.pop(k, None)
        fc = fake_ctx_cls({"toolaria": cfg})
        toolaria.register(fc)
        c = toolaria._store.cfg
        assert c.get("sequence_capture") is False
        assert c.get("args_snapshot_max_chars") == 2000
        assert c.get("verify_integrity") is True

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


# ═══ T1.2 — JSONL sequence sidecar, default OFF ═════════════════════════


class TestSequenceSidecar:
    """T1.2: store_path/sequences/events.jsonl; one line per fetch."""

    def test_default_off_writes_nothing(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        seq_dir = toolaria._store.blob_dir.parent / "sequences"
        assert not seq_dir.exists() or not any(seq_dir.glob("*.jsonl")), (
            "sequence_capture must default OFF ⇒ no sidecar file or dir"
        )

    def test_enabled_appends_one_line_per_fetch(self, plugin, toolaria):
        toolaria._store.cfg["sequence_capture"] = True
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        for _ in range(3):
            toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        path = toolaria._store.blob_dir.parent / "sequences" / "events.jsonl"
        assert path.exists()
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3, f"expected 3 events, got {len(lines)}"
        for ln in lines:
            ev = json.loads(ln)
            assert set(ev) == {"ts", "sid", "blob_id"}
            assert ev["blob_id"] == bid
            assert ev["sid"] == "s1"

    def test_ordering_preserved_across_reload(self, plugin, toolaria, tmp_path):
        """Write N events, then read them back in the same order.

        Reused by the T1.2 'N-event fixture' matrix requirement."""
        toolaria._store.cfg["sequence_capture"] = True
        bids = [toolaria._store.put(f"data{i}", "web_search",
                                     session_id="s1") for i in range(5)]
        for bid in bids:
            toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")
        path = toolaria._store.blob_dir.parent / "sequences" / "events.jsonl"
        seen = [json.loads(ln)["blob_id"] for ln in
                path.read_text().splitlines() if ln.strip()]
        assert seen == bids, (
            f"ordering not preserved: wrote {bids!r}, read {seen!r}"
        )

    def test_turn_field_included_when_provided(self, plugin, toolaria):
        toolaria._store.cfg["sequence_capture"] = True
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        # Direct path: invoke the recorder with a turn counter.
        toolaria._store._record_sequence(bid, "s1", time.time(), turn=7)
        path = toolaria._store.blob_dir.parent / "sequences" / "events.jsonl"
        ev = json.loads(path.read_text().splitlines()[-1])
        assert ev.get("turn") == 7

    def test_turn_field_absent_when_none(self, plugin, toolaria):
        toolaria._store.cfg["sequence_capture"] = True
        toolaria._store._record_sequence("abc123456789", "s1", time.time(), None)
        path = toolaria._store.blob_dir.parent / "sequences" / "events.jsonl"
        ev = json.loads(path.read_text().splitlines()[-1])
        assert "turn" not in ev

    def test_append_failure_does_not_break_fetch(
        self, plugin, toolaria, monkeypatch
    ):
        toolaria._store.cfg["sequence_capture"] = True
        # Pre-create a bid so the fetch path can succeed.
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        # Now monkey-patch open() to raise on the events.jsonl path; the
        # recorder's try/except must swallow it.
        real_open = open

        def selective_open(p, *a, **kw):
            if str(p).endswith("events.jsonl"):
                raise OSError("disk gone")
            return real_open(p, *a, **kw)

        monkeypatch.setattr("builtins.open", selective_open)
        # Direct recorder call must not raise.
        toolaria._store._record_sequence("abc123456789", "s1", time.time(), None)
        # And the fetch path must not raise either.
        toolaria._fetch(args={"id": bid, "mode": "stat"}, session_id="s1")


# ═══ T1.3 — args provenance, secret redaction, size cap ═════════════════


import hashlib as _hashlib  # noqa: E402  (local to T1.3 cluster)


class TestArgsProvenance:
    """T1.3: redacted args snapshot in the index entry."""

    def test_args_absent_produces_null_provenance(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert "args_snapshot" in entry
        assert entry["args_snapshot"] is None

    def test_args_stored_when_provided(self, plugin, toolaria):
        args = {"url": "https://example.com", "depth": 2}
        bid = toolaria._store.put("data", "web_search", session_id="s1",
                                  args=args)
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry["args_snapshot"], (
            "args_snapshot must be non-empty when args provided"
        )
        assert "https://example.com" in entry["args_snapshot"]

    def test_secret_keys_redacted(self, plugin, toolaria):
        args = {"api_key": "sk-abcdefgh12345678", "url": "https://example.com"}
        bid = toolaria._store.put("data", "web_search", session_id="s1",
                                  args=args)
        snap = toolaria._store._load_idx("s1")["blobs"][bid]["args_snapshot"]
        assert "sk-abcdefgh12345678" not in snap, (
            "secret value stored verbatim; redaction failed"
        )
        assert "[REDACTED]" in snap

    def test_secret_keys_redacted_various_names(self, plugin, toolaria):
        cases = [
            {"token": "leak-me", "x": 1},
            {"password": "leak-me", "x": 1},
            {"authorization": "leak-me", "x": 1},
            {"bearer": "leak-me", "x": 1},
            {"secret": "leak-me", "x": 1},
            {"API_KEY": "leak-me", "x": 1},
            {"X-Api-Key": "leak-me", "x": 1},
        ]
        for args in cases:
            bid = toolaria._store.put("d", "web_search", session_id="s1",
                                      args=args)
            snap = toolaria._store._load_idx("s1")["blobs"][bid]["args_snapshot"]
            assert "leak-me" not in snap, (
                f"value not redacted for key in {args!r}; snap={snap!r}"
            )

    def test_secret_value_pattern_redacted(self, plugin, toolaria):
        """A sk-... or bearer ... value is masked regardless of key."""
        args = {"url": "https://x.test?key=sk-abcdefgh12345678"}
        bid = toolaria._store.put("d", "web_search", session_id="s1", args=args)
        snap = toolaria._store._load_idx("s1")["blobs"][bid]["args_snapshot"]
        assert "sk-abcdefgh12345678" not in snap
        bid2 = toolaria._store.put("d", "web_search", session_id="s1",
                                   args={"h": "bearer eyJabcdefghij"})
        snap2 = toolaria._store._load_idx("s1")["blobs"][bid2]["args_snapshot"]
        assert "eyJabcdefghij" not in snap2

    def test_nested_args_redacted(self, plugin, toolaria):
        args = {"outer": {"api_key": "sk-abcdefgh12345678", "ok": 1},
                "list": [{"token": "sk-aaaaaaaaaaaaaa"}]}
        bid = toolaria._store.put("d", "web_search", session_id="s1", args=args)
        snap = toolaria._store._load_idx("s1")["blobs"][bid]["args_snapshot"]
        assert "sk-abcdefgh12345678" not in snap
        assert "sk-aaaaaaaaaaaaaa" not in snap
        assert "ok" in snap and "1" in snap

    def test_size_cap_truncates_snapshot(self, plugin, toolaria):
        toolaria._store.cfg["args_snapshot_max_chars"] = 200
        args = {"big": "x" * 5000, "more": "y" * 5000}
        bid = toolaria._store.put("d", "web_search", session_id="s1", args=args)
        snap = toolaria._store._load_idx("s1")["blobs"][bid]["args_snapshot"]
        assert len(snap) <= 200 + 100  # truncation marker adds a few chars
        assert "truncated" in snap

    def test_result_hash_unchanged_by_args(self, plugin, toolaria):
        """Redaction only touches the index entry; the blob_id is unchanged."""
        content = "stable-content-for-hash-1"
        args = {"api_key": "sk-abcdefgh12345678"}
        b1 = toolaria._store.put(content, "web_search", session_id="s1")
        b2 = toolaria._store.put(content, "web_search", session_id="s1", args=args)
        b3 = toolaria._store.put(content, "web_search", session_id="s1", args=None)
        assert b1 == b2 == b3
        # And the recorded hash matches the actual content.
        expected = _hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert toolaria._store._load_idx("s1")["blobs"][b1]["hash"] == expected

    def test_args_provenance_via_rescue(self, plugin, toolaria):
        """The full rescue path persists the snapshot, not just direct put()."""
        r = toolaria._on_transform(
            tool_name="web_extract",
            result="x" * 9000,
            args={"url": "https://x.test", "api_key": "sk-abcdefgh12345678"},
            session_id="s1",
        )
        assert r is not None
        # Pull the index entry for the session — should have an args_snapshot
        # with the url preserved and the secret redacted.
        idx = toolaria._store._load_idx("s1")
        snap = next(iter(idx["blobs"].values()))["args_snapshot"]
        assert "https://x.test" in snap
        assert "sk-abcdefgh12345678" not in snap

    def test_max_chars_zero_disables_capture(self, plugin, toolaria):
        toolaria._store.cfg["args_snapshot_max_chars"] = 0
        bid = toolaria._store.put("d", "web_search", session_id="s1",
                                  args={"url": "https://x.test"})
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry["args_snapshot"] is None, (
            "max_chars=0 must be a benchmark-only escape hatch (no capture)"
        )


# ═══ T1.4 — full SHA256 integrity verify on fetch ════════════════════════


import logging as _logging  # noqa: E402  (local to T1.4 cluster)


class TestIntegrityVerify:
    """T1.4: deterministic marker on hash mismatch; default on."""

    def test_valid_blob_passes_verify(self, plugin, toolaria):
        bid = toolaria._store.put("clean content", "web_search",
                                   session_id="s1")
        r = toolaria._fetch(args={"id": bid, "mode": "full"}, session_id="s1")
        assert r == "clean content"
        assert "integrity" not in r.lower()

    def test_corrupted_blob_returns_exact_marker(self, plugin, toolaria):
        bid = toolaria._store.put("clean content", "web_search",
                                   session_id="s1")
        # Corrupt the blob bytes; the index entry's hash is still the
        # original, so verify must fail.
        (toolaria._store.blob_dir / bid).write_bytes(b"corrupted!")
        r = toolaria._fetch(args={"id": bid, "mode": "full"}, session_id="s1")
        from blobstore import (INTEGRITY_FAIL_MARKER_PREFIX,
                                INTEGRITY_FAIL_MARKER_SUFFIX)
        expected = f"{INTEGRITY_FAIL_MARKER_PREFIX}{bid}{INTEGRITY_FAIL_MARKER_SUFFIX}"
        assert r == expected, f"expected exact marker, got {r!r}"

    def test_corruption_logged(self, plugin, toolaria, caplog):
        bid = toolaria._store.put("clean", "web_search", session_id="s1")
        (toolaria._store.blob_dir / bid).write_bytes(b"corrupt")
        with caplog.at_level(_logging.WARNING, logger="blobstore"):
            toolaria._fetch(args={"id": bid, "mode": "full"}, session_id="s1")
        assert any("integrity" in r.getMessage().lower() for r in caplog.records), (
            "integrity failure must be logged at WARNING"
        )

    def test_verify_disabled_skips_check(self, plugin, toolaria):
        """verify_integrity=false is a documented benchmark-only escape."""
        toolaria._store.cfg["verify_integrity"] = False
        bid = toolaria._store.put("clean", "web_search", session_id="s1")
        (toolaria._store.blob_dir / bid).write_bytes(b"corrupt")
        r = toolaria._fetch(args={"id": bid, "mode": "full"}, session_id="s1")
        assert r == "corrupt", (
            "with verify disabled, the corrupted bytes are returned; this is "
            "the documented benchmark-only behaviour"
        )

    def test_default_verify_is_on(self, toolaria, base_cfg, fake_ctx_cls):
        """A fresh cfg must default to verify_integrity=true."""
        cfg = dict(base_cfg)
        cfg.pop("verify_integrity", None)
        fc = fake_ctx_cls({"toolaria": cfg})
        toolaria.register(fc)
        assert toolaria._store.cfg.get("verify_integrity", True) is True

    def test_other_modes_also_verify(self, plugin, toolaria):
        """Integrity is enforced for every mode that reads bytes."""
        from blobstore import INTEGRITY_FAIL_MARKER_PREFIX
        bid = toolaria._store.put("hello\nworld", "web_search", session_id="s1")
        (toolaria._store.blob_dir / bid).write_bytes(b"corrupted")
        for mode in ("range", "grep", "full", "outline"):
            r = toolaria._fetch(args={"id": bid, "mode": mode}, session_id="s1")
            assert r.startswith(INTEGRITY_FAIL_MARKER_PREFIX), (
                f"mode {mode} did not return the integrity marker: {r!r}"
            )

    def test_perf_guard_multi_mb(self, plugin, toolaria):
        """Multi-MB blobs must verify within a generous bound (sleep tolerance).

        Uses range mode (full mode refuses over full_fetch_max_chars=50000 by
        default, regardless of the verify path). The point is to confirm
        the SHA256 over a multi-MB blob fits the perf guard, not to time the
        full-mode read path."""
        big = "x" * (3 * 1024 * 1024)  # 3 MB
        bid = toolaria._store.put(big, "web_extract", session_id="s1")
        t0 = time.time()
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 1}, session_id="s1")
        elapsed = time.time() - t0
        # Generous bound — slow CI runners, TSan, etc. The point is to
        # catch quadratic regressions, not microbenchmark.
        assert "lines 0..0" in r, f"unexpected range output: {r[:200]!r}"
        assert elapsed < 3.0, f"verify+range on 3MB took {elapsed:.2f}s (bound 3s)"

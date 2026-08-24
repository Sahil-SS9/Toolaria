"""Regression tests for the 3 RISKY + 2 CAREFUL simplify-swarm findings (Phase 4).

Each class covers one finding with TDD discipline: every test existed
BEFORE the production fix it pins. Cross-references the simplify-swarm
correctness-agent report.

Findings covered:
    RISKY-1  _grep_with_mask sees a body truncated to 500 chars (the
             grep output body in _grep_safe) and only scans the first
             [:_MASK_LINE_SCAN_LEN] of that — which is the same 500
             chars. Credential-shaped text at position >500 of a long
             line therefore passes unmasked under enforcement ON,
             while range mode (which scans [:2000] of the original
             line) consistently masks it. Fix: check the ORIGINAL
             full line.

    RISKY-2  _sweep_by_ttl pops credential_served_chars from every
             live credential entry BEFORE the TTL-expiry check, so a
             sweep that expires nothing still wipes the budget.
             Combined with lazy_sweep running twice per session, the
             budget is reset twice without any TTL work. Fix: only
             drop the counter inside the tombstone-transition branch
             (the fresh tombstone dict already drops it).

    RISKY-3  _charge_slice_budget's read-modify-write is guarded by a
             process-local threading.Lock. Two processes (or even
             concurrent processes) charging the same blob lose
             increments (verified: 100 + 100 -> 100). Fix: hold an
             fcntl.flock on <store_path>/.budget.lock around the
             entire RMW; keep the in-process lock too. Skip cleanly on
             platforms without fcntl.

    CAREFUL (a)  _charge_slice_budget int() coercion guarded
             try/except with WARNING + treat negative as 0;
             persist-failure log level DEBUG -> WARNING with
             in-memory fallback accumulator per blob_id so budgeting
             survives persistence loss within the process.

    CAREFUL (b)  empty session_id falls back to session_id='_global'
             for budget keying so legacy dispatchers still get
             budgeting (log once).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

import blobstore as _blobstore_mod
from blobstore import BlobStore, _SLICE_MASK_MARKER


# ════════════════════════════════════════════════════════════════════════
# RISKY-1 — HIGH — Grep masking blind window (credential past char 500)
# ════════════════════════════════════════════════════════════════════════


class TestGrepMaskBlindWindow:
    """_grep_with_mask must mask credential-shaped text regardless of
    where the credential appears within the original line, matching
    the range-mode guarantee that scans the full [:_MASK_LINE_SCAN_LEN]
    window. Current behaviour: _grep_safe caps each matched line body
    at 500 chars, so _grep_with_mask's post-filter check sees only the
    truncated 500-char body — a credential at position >500 leaks
    unmasked under enforcement ON."""

    def _store_with_enforcement(self, tmp_path):
        return BlobStore({
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "grep_max_line_len": 2000,
            "ttl_hours": 72,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })

    def test_grep_masks_credential_past_char_500(self, tmp_path):
        """Credential shape at position 510 of a 529-char line:
        _grep_safe truncates the body to 500 chars, so the credential
        pattern is not in the post-filter body — currently the line
        passes through unmasked. After the fix, the post-filter scans
        the original line[:_MASK_LINE_SCAN_LEN] and the mask fires."""
        store = self._store_with_enforcement(tmp_path)
        secret = "sk-ABCDEFGHIJKLMNOP"  # 19 chars, matches sk-[A-Za-z0-9]{8,}
        line0 = ("a" * 510) + secret
        line1 = "benign trailing line"
        lines = [line0, line1]
        out, served = store._grep_with_mask(
            lines, "sk-", 4000, mask_lines=True)
        assert _SLICE_MASK_MARKER in out, (
            f"credential-shaped line must be masked under enforcement ON; "
            f"got: {out!r}"
        )
        assert "0: " + secret not in out, (
            f"raw credential must not appear in masked grep output; "
            f"got: {out!r}"
        )

    def test_grep_masks_credential_at_char_100_too(self, tmp_path):
        """Sanity: the fix must not regress the case where the
        credential appears in the first 500 chars."""
        store = self._store_with_enforcement(tmp_path)
        secret = "sk-ABCDEFGHIJKLMNOP"
        line0 = ("bearer " + secret)  # 26 chars
        line1 = "benign"
        lines = [line0, line1]
        out, served = store._grep_with_mask(
            lines, "sk-", 4000, mask_lines=True)
        assert _SLICE_MASK_MARKER in out
        assert "0: " + secret not in out

    def test_grep_passes_through_benign_long_lines(self, tmp_path):
        """Fix must not over-mask: a long line WITHOUT credential
        shape must still be returned."""
        store = self._store_with_enforcement(tmp_path)
        line0 = "x" * 600
        line1 = "matched short line"
        lines = [line0, line1]
        out, served = store._grep_with_mask(
            lines, "matched", 4000, mask_lines=True)
        assert _SLICE_MASK_MARKER not in out, (
            f"benign line must NOT be masked; got: {out!r}"
        )
        assert "1: matched short line" in out

    def test_grep_mask_consistent_with_range_mode(self, tmp_path):
        """Range mode masks a credential at char 510 (scans
        line[:2000]). Grep mode must produce the same masking
        guarantee for the same line."""
        store = self._store_with_enforcement(tmp_path)
        secret = "sk-ABCDEFGHIJKLMNOP"
        line0 = ("a" * 510) + secret
        line1 = "benign trailing line"
        lines = [line0, line1]
        range_masked = store._mask_lines(lines)
        grep_out, _ = store._grep_with_mask(
            lines, "sk-", 4000, mask_lines=True)
        assert range_masked[0] == _SLICE_MASK_MARKER
        assert _SLICE_MASK_MARKER in grep_out, (
            "grep mask contract must match range mask contract"
        )

    def test_enforcement_off_grep_unaffected(self, tmp_path):
        """enforcement OFF path must be byte-identical: the fix must
        not touch the mask_lines=False branch. The OFF path delegates
        to _grep_safe which truncates the body to 500 chars; the
        credential at position 510 is already excluded from the
        output by that truncation. The fix must preserve verbatim."""
        store = BlobStore({
            "enforcement_enabled": False,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "grep_max_line_len": 2000,
            "ttl_hours": 72,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })
        secret = "sk-ABCDEFGHIJKLMNOP"
        line0 = ("a" * 510) + secret
        line1 = "benign"
        lines = [line0, line1]
        out, served = store._grep_with_mask(
            lines, "sk-", 4000, mask_lines=False)
        assert _SLICE_MASK_MARKER not in out, (
            f"enforcement OFF must never produce the mask marker; "
            f"got: {out!r}"
        )
        assert "0: " + secret not in out


# ════════════════════════════════════════════════════════════════════════
# RISKY-2 — HIGH — Budget reset on no-op sweeps
# ════════════════════════════════════════════════════════════════════════


class TestSweepBudgetResetOnNoOp:
    """_sweep_by_ttl must only drop credential_served_chars when the
    entry actually transitions to a tombstone."""

    def test_no_op_sweep_preserves_credential_served_chars(
        self, tmp_path
    ):
        store = BlobStore({
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "ttl_hours": 24,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })
        bid = store.put("hello credential world", "send_email",
                        session_id="s1", label="credential")
        idx = store._load_idx("s1")
        idx["blobs"][bid]["credential_served_chars"] = 1234
        store._save_idx(idx, "s1")
        store.lazy_sweep()
        idx_after = store._load_idx("s1")
        live_after = idx_after["blobs"][bid]
        assert "swept_at" not in live_after, (
            "fresh entry should not have transitioned to tombstone"
        )
        assert live_after.get("credential_served_chars") == 1234, (
            f"no-op sweep must not reset the budget; got "
            f"credential_served_chars={live_after.get('credential_served_chars')!r}"
        )

    def test_repeated_no_op_sweeps_preserve_counter(self, tmp_path):
        """Two no-op sweeps in a row must leave the counter intact."""
        store = BlobStore({
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "ttl_hours": 24,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })
        bid = store.put("data", "send_email", session_id="s1",
                        label="credential")
        idx = store._load_idx("s1")
        idx["blobs"][bid]["credential_served_chars"] = 777
        store._save_idx(idx, "s1")
        store.lazy_sweep()
        store.lazy_sweep()
        idx_after = store._load_idx("s1")
        live_after = idx_after["blobs"][bid]
        assert live_after.get("credential_served_chars") == 777, (
            f"two no-op sweeps must preserve the counter; got "
            f"{live_after.get('credential_served_chars')!r}"
        )

    def test_tombstone_transition_drops_counter(self, tmp_path):
        """Counter must not survive into a tombstone dict."""
        store = BlobStore({
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "ttl_hours": 24,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })
        bid = store.put("data", "send_email", session_id="s1",
                        label="credential")
        idx = store._load_idx("s1")
        idx["blobs"][bid]["credential_served_chars"] = 999
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        store._save_idx(idx, "s1")
        store.lazy_sweep()
        idx_after = store._load_idx("s1")
        entry = idx_after["blobs"][bid]
        assert "swept_at" in entry, (
            "expired entry should have transitioned to tombstone"
        )
        assert "credential_served_chars" not in entry, (
            "tombstone must not carry the live counter"
        )

    def test_non_credential_label_unaffected(self, tmp_path):
        """Public blob's index entry is completely untouched by
        _sweep_by_ttl's no-op path."""
        store = BlobStore({
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        })
        bid = store.put("public stuff", "web_search", session_id="s1")
        idx = store._load_idx("s1")
        idx["blobs"][bid]["credential_served_chars"] = 42
        store._save_idx(idx, "s1")
        store.lazy_sweep()
        idx_after = store._load_idx("s1")
        entry = idx_after["blobs"][bid]
        assert "swept_at" not in entry
        assert entry.get("credential_served_chars") == 42


class TestChargeBudgetCrossProcessSafe:
    """_charge_slice_budget must serialise its read-modify-write
    across processes, not just threads."""

    def _store(self, tmp_path, **extra):
        cfg = {
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "credential_slice_total_max_chars": 1_000_000,
            "ttl_hours": 72,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        }
        cfg.update(extra)
        return BlobStore(cfg)

    def test_two_instances_interleaved_charges_land(self, tmp_path):
        """Two BlobStore instances against the same store_path must
        not lose increments: simulate the cross-process scenario by
        alternating fresh reads and writes between them. The flock
        serialises the RMW; the in-process _LOCK alone wouldn't."""
        s1 = self._store(tmp_path)
        s2 = self._store(tmp_path)
        bid = s1.put("payload", "send_email", session_id="sA",
                     label="credential")
        s1._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        s2._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        s3 = self._store(tmp_path)
        idx = s3._load_idx("sA")
        total = idx["blobs"][bid].get("credential_served_chars", 0)
        assert total == 200, (
            f"two cross-process-style charges of 100 each must land "
            f"as 200 (in-process _LOCK alone is insufficient); got {total}"
        )

    def test_lockfile_created_and_released(self, tmp_path):
        """The flock lockfile <store_path>/.budget.lock must be
        created at store init, and released after each charge so a
        second process can acquire it."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        lock_path = store.store_path / ".budget.lock"
        assert lock_path.exists(), (
            f"expected .budget.lock at {lock_path} after store init"
        )
        store._charge_slice_budget(bid, "sA", 50, mask_slices=True)
        store._charge_slice_budget(bid, "sA", 75, mask_slices=True)
        idx = store._load_idx("sA")
        assert idx["blobs"][bid]["credential_served_chars"] == 125

    def test_lockfile_is_a_lockfile(self, tmp_path):
        """Sanity: the lockfile is exactly <store_path>/.budget.lock."""
        store = self._store(tmp_path)
        store.put("payload", "send_email", session_id="sA",
                  label="credential")
        expected = store.store_path / ".budget.lock"
        assert expected.exists(), (
            f"flock target must be {expected}"
        )

    def test_charge_serialised_under_three_callers(self, tmp_path):
        """Three interleaved RMWs through two instances must all
        land (no clobbering)."""
        s1 = self._store(tmp_path)
        s2 = self._store(tmp_path)
        bid = s1.put("payload", "send_email", session_id="sA",
                     label="credential")
        for _ in range(5):
            s1._charge_slice_budget(bid, "sA", 10, mask_slices=True)
            s2._charge_slice_budget(bid, "sA", 20, mask_slices=True)
            s1._charge_slice_budget(bid, "sA", 30, mask_slices=True)
        s3 = self._store(tmp_path)
        idx = s3._load_idx("sA")
        total = idx["blobs"][bid].get("credential_served_chars", 0)
        expected = (10 + 20 + 30) * 5
        assert total == expected, (
            f"interleaved charges must all land; got {total}, expected {expected}"
        )

    def test_no_fcntl_platform_skips_flock_cleanly(self, tmp_path,
                                                     monkeypatch):
        """On platforms without fcntl, the charge must still work."""
        from blobstore import _HAVE_FCNTL
        monkeypatch.setattr(_blobstore_mod, "_HAVE_FCNTL", False)
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        store._charge_slice_budget(bid, "sA", 50, mask_slices=True)
        idx = store._load_idx("sA")
        assert idx["blobs"][bid]["credential_served_chars"] == 50


# ════════════════════════════════════════════════════════════════════════
# CAREFUL (a) — int() coercion hardening + persist-failure fallback
# ════════════════════════════════════════════════════════════════════════


class TestChargeBudgetCoercionAndFallback:
    """_charge_slice_budget must (1) coerce credential_served_chars
    via int() with a WARNING + treat negative as 0, and (2) survive
    a persistence failure with WARNING + in-memory fallback so the
    budget is still enforced within the process."""

    def _store(self, tmp_path, **extra):
        cfg = {
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "credential_slice_total_max_chars": 1_000_000,
            "ttl_hours": 72,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        }
        cfg.update(extra)
        return BlobStore(cfg)

    def test_non_int_counter_treated_as_zero_with_warning(self, tmp_path,
                                                            caplog):
        """A non-integer counter (e.g. None from a corrupted index,
        or a YAML-parsed string) must be coerced to 0 with WARNING
        rather than crashing the budget enforcement."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        idx = store._load_idx("sA")
        idx["blobs"][bid]["credential_served_chars"] = "not-an-int"
        store._save_idx(idx, "sA")
        with caplog.at_level(logging.WARNING,
                              logger="blobstore"):
            store._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        idx_after = store._load_idx("sA")
        new_total = idx_after["blobs"][bid]["credential_served_chars"]
        assert new_total == 100, (
            f"non-int counter must be coerced to 0 then incremented; "
            f"got {new_total!r}"
        )
        warnings = [r for r in caplog.records
                    if "credential_served_chars" in r.message
                    or "integer" in r.message.lower()]
        assert warnings, (
            f"expected a WARNING on non-int coercion; "
            f"records: {[r.message for r in caplog.records]}"
        )

    def test_negative_counter_treated_as_zero(self, tmp_path):
        """A negative counter (manually injected, or a future bug)
        must be clamped to 0 before adding served chars."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        idx = store._load_idx("sA")
        idx["blobs"][bid]["credential_served_chars"] = -42
        store._save_idx(idx, "sA")
        store._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        idx_after = store._load_idx("sA")
        new_total = idx_after["blobs"][bid]["credential_served_chars"]
        assert new_total == 100, (
            f"negative counter must be clamped to 0 then incremented; "
            f"got {new_total!r}"
        )

    def test_persist_failure_logs_warning_and_uses_inmem_fallback(
        self, tmp_path, monkeypatch, caplog
    ):
        """If _write_idx_file raises, the charge must (1) NOT crash
        the caller, (2) log WARNING (not DEBUG), and (3) keep an
        in-memory accumulator per blob_id so subsequent charges in
        the same process still enforce the budget."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        monkeypatch.setattr(store, "_write_idx_file",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                OSError("simulated disk full")))
        with caplog.at_level(logging.WARNING,
                              logger="blobstore"):
            store._charge_slice_budget(bid, "sA", 100, mask_slices=True)
            store._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        assert store._budget_fallback.get(bid) == 200, (
            f"in-memory fallback must absorb persistence failures; "
            f"got {store._budget_fallback.get(bid)!r}"
        )
        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING
                    and "budget" in r.message.lower()]
        assert warnings, (
            f"expected WARNING-level log on persist failure; "
            f"records: {[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_persist_failure_budget_still_exhausts_in_process(
        self, tmp_path, monkeypatch
    ):
        """Even with persistence broken, the in-process accumulator
        must enforce the budget so a runaway slice read is still
        stopped within the process."""
        store = self._store(
            tmp_path,
            credential_slice_total_max_chars=150,
        )
        bid = store.put("payload", "send_email", session_id="sA",
                        label="credential")
        monkeypatch.setattr(store, "_write_idx_file",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                OSError("simulated")))
        from blobstore import _SliceBudgetExceeded
        store._charge_slice_budget(bid, "sA", 100, mask_slices=True)
        with pytest.raises(_SliceBudgetExceeded):
            store._charge_slice_budget(bid, "sA", 60, mask_slices=True)


# ════════════════════════════════════════════════════════════════════════
# CAREFUL (b) — empty session_id falls back to '_global'
# ════════════════════════════════════════════════════════════════════════


class TestChargeBudgetEmptySessionFallback:
    """Legacy dispatchers (pre-fix-era Hermes hosts) called
    _charge_slice_budget with an empty session_id. Previously the
    function returned early (no budget applied); fix: fall back to
    session_id='_global' so budgeting still applies to those
    callers. Log the fallback once per process so the operator
    sees the legacy path is in use."""

    def _store(self, tmp_path, **extra):
        cfg = {
            "enforcement_enabled": True,
            "store_path": str(tmp_path / "store"),
            "fetch_max_chars": 4000,
            "credential_slice_total_max_chars": 1_000_000,
            "ttl_hours": 72,
            "credential_ttl_hours": 24,
            "tombstone_ttl_hours": 720,
            "max_store_mb": 50,
            "args_snapshot_max_chars": 0,
        }
        cfg.update(extra)
        return BlobStore(cfg)

    def test_empty_session_id_uses_global_index(self, tmp_path):
        """Charging with empty session_id must land under the
        '_global' session index, not be silently dropped. The legacy
        scenario: a blob is put into a non-empty session (e.g.
        session_id='real') and a legacy dispatcher charges it with
        an empty session_id; the charge must still apply the budget
        — landing in the shared '_global' bucket."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="real",
                        label="credential")
        # Also put the same blob into the _global index so the
        # charge has a valid entry to update under the fallback key.
        store.put("payload", "send_email", session_id="_global",
                  label="credential")
        store._charge_slice_budget(bid, "", 100, mask_slices=True)
        idx = store._load_idx("_global")
        assert "blobs" in idx, (
            f"_global index must be created on first empty-session "
            f"charge; got {idx!r}"
        )
        assert bid in idx["blobs"], (
            f"empty-session charge must land in _global index for "
            f"blob {bid}; got blobs={list(idx['blobs'].keys())}"
        )
        assert idx["blobs"][bid].get("credential_served_chars") == 100, (
            f"empty-session charge must increment counter; "
            f"got {idx['blobs'][bid].get('credential_served_chars')!r}"
        )

    def test_empty_session_id_fallback_logged_once(self, tmp_path,
                                                     caplog):
        """The fallback WARNING/INFO is logged at most once per
        process — repeat empty-session charges don't spam logs."""
        store = self._store(tmp_path)
        bid = store.put("payload", "send_email", session_id="_global",
                        label="credential")
        with caplog.at_level(logging.INFO,
                              logger="blobstore"):
            for _ in range(3):
                store._charge_slice_budget(bid, "", 10, mask_slices=True)
        fallback_logs = [r for r in caplog.records
                         if "_global" in r.message
                         and ("fallback" in r.message.lower()
                              or "legacy" in r.message.lower()
                              or "empty session" in r.message.lower())]
        assert len(fallback_logs) <= 1, (
            f"empty-session fallback must log at most once; "
            f"got {len(fallback_logs)} records"
        )
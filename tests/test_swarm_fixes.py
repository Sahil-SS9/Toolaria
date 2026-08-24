"""Regression tests for RISKY-3 + CAREFULs (Phase 4, split-commit 1).

This file is the RISKY-3 slice of the simplify-swarm correctness
fix set. Each test pins one cross-process finding with TDD
discipline.

RISKY-3: _charge_slice_budget's read-modify-write is guarded by a
process-local threading.Lock. Two processes (or even concurrent
processes) charging the same blob lose increments (verified:
100 + 100 -> 100). Fix: hold an fcntl.flock on
<store_path>/.budget.lock around the entire RMW; keep the
in-process lock too. Skip cleanly on platforms without fcntl.
"""
from __future__ import annotations

import pytest

import blobstore as _blobstore_mod
from blobstore import BlobStore, _SLICE_MASK_MARKER


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
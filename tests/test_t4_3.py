"""Phase 4 — T4.3: Fernet key rotation.

TDD discipline: every test was written before the production code it pins.

Contract (T4.3, subagent brief):

  - ``BlobStore.rotate_key(new_key_path)`` re-encrypts every encrypted
    credential blob under a new key in one pass.
  - Exactly ONE ledger entry per rotation is written (decision
    ``key_rotated``, with a count).
  - Abort-safe: the OLD key remains the active key until the pass
    completes; if the pass fails partway, blobs already re-encrypted
    under the new key are rolled back so the store keeps working
    with the OLD key.
  - After a successful rotation the store serves blobs with the NEW
    key. Old key file (if separate) is retained; if the new path
    equals the old path the on-disk file is overwritten atomically.

Reads after a successful rotation must decrypt identically (roundtrip).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

try:
    from cryptography.fernet import Fernet
    _HAVE_FERNET = True
except ImportError:  # pragma: no cover
    _HAVE_FERNET = False


pytestmark = pytest.mark.skipif(
    not _HAVE_FERNET,
    reason="cryptography.fernet unavailable (uv run --with cryptography)",
)


def _write_key(path: Path, key: bytes | None = None) -> bytes:
    key = key or Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    os.chmod(path, 0o600)
    return key


def _enable_encryption(toolaria, key_file: Path) -> bytes:
    key = _write_key(key_file)
    toolaria._cfg["toolaria_key_file"] = str(key_file)
    # Drop cached Fernet so the store re-reads the key on next use.
    toolaria._store._fernet = None
    return key


# ── ledger helper ─────────────────────────────────────────────────────────


def _read_key_rotations(store_path: Path) -> list[dict]:
    p = store_path / "ledger" / "key_rotations.jsonl"
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ── Happy path: rotation re-encrypts every encrypted blob ─────────────────


class TestKeyRotationHappyPath:
    """Successful rotation re-encrypts each blob, switches the active
    key, writes one ledger entry, and preserves plaintext roundtrip."""

    def test_rotate_re_encrypts_every_encrypted_credential_blob(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        # Write 3 distinct credential blobs.
        b1 = toolaria._store.put("ALPHA-XYZ", "send_email",
                                   session_id="s1", label="credential")
        b2 = toolaria._store.put("BETA-XYZ", "send_email",
                                   session_id="s1", label="credential")
        b3 = toolaria._store.put("GAMMA-XYZ", "send_email",
                                   session_id="s1", label="credential")
        # Rotate.
        new_key_file = tmp_path / "new.key"
        new_key = _write_key(new_key_file)
        count = toolaria._store.rotate_key(str(new_key_file))
        assert count == 3, (
            f"T4.3: rotation must report the number of blobs re-encrypted; "
            f"got {count!r}"
        )
        # Every blob must roundtrip with the NEW key.
        for bid, expected in [(b1, "ALPHA-XYZ"), (b2, "BETA-XYZ"),
                                (b3, "GAMMA-XYZ")]:
            r = toolaria._fetch(args={"id": bid, "mode": "full"},
                                 session_id="s1")
            assert r == expected, (
                f"post-rotation fetch of {bid} must return plaintext; "
                f"got {r!r}"
            )
        # The cfg now points at the new key file.
        assert toolaria._store.cfg.get("toolaria_key_file") == str(new_key_file), (
            "T4.3: after rotate_key the store's key file must be the new path"
        )

    def test_rotate_does_not_touch_plaintext_blobs(
        self, plugin, toolaria, tmp_path,
    ):
        """Public/credential-not blobs are not encrypted ⇒ rotation
        must not touch their on-disk bytes (still plaintext)."""
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        # Public blob stays plaintext under the same key.
        public = "PUBLIC-CONTENT" * 50
        pub_bid = toolaria._store.put(public, "web_search", session_id="s1")
        # Snapshot the public blob's ciphertext... err, plaintext bytes.
        pre_bytes = (toolaria._store.blob_dir / pub_bid).read_bytes()
        # Rotate.
        new_key_file = tmp_path / "new.key"
        toolaria._store.rotate_key(str(new_key_file))
        post_bytes = (toolaria._store.blob_dir / pub_bid).read_bytes()
        assert pre_bytes == post_bytes, (
            "T4.3: rotation must not rewrite plaintext blobs (their bytes "
            "are unchanged after rotation)"
        )

    def test_rotate_writes_exactly_one_ledger_entry_per_pass(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        toolaria._store.put("TOKEN-A", "send_email", session_id="s1",
                             label="credential")
        toolaria._store.put("TOKEN-B", "send_email", session_id="s1",
                             label="credential")
        new_key_file = tmp_path / "new.key"
        toolaria._store.rotate_key(str(new_key_file))
        rows = _read_key_rotations(Path(toolaria._store.store_path))
        assert len(rows) == 1, (
            f"T4.3: rotation must emit EXACTLY ONE ledger entry per pass; "
            f"got {len(rows)} rows"
        )
        row = rows[0]
        assert row.get("decision") == "key_rotated", (
            f"T4.3: ledger row decision must be 'key_rotated'; "
            f"got {row.get('decision')!r}"
        )
        assert row.get("count") == 2, (
            f"T4.3: ledger row must carry the count of re-encrypted blobs; "
            f"got {row.get('count')!r}"
        )


# ── Atomicity: old key retained until pass completes ──────────────────────


class TestKeyRotationAtomicity:
    """The store keeps working with the old key until the rotation
    finishes. A fresh put during/after rotation uses the new key."""

    def test_rotation_with_new_path_clears_old_path_dependency(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        bid = toolaria._store.put("SECRET-OLD", "send_email",
                                   session_id="s1", label="credential")
        new_key_file = tmp_path / "new.key"
        toolaria._store.rotate_key(str(new_key_file))
        # Reading the (now new-key-encrypted) blob with the new key
        # works, but a manual decrypt with the OLD key would fail.
        old_fernet = Fernet(old_key_file.read_bytes())
        raw_cipher = (toolaria._store.blob_dir / bid).read_bytes()
        with pytest.raises(Exception):
            old_fernet.decrypt(raw_cipher), (
                "T4.3: after rotation the blob's on-disk ciphertext must "
                "NOT decrypt under the old key"
            )

    def test_new_puts_use_new_key_after_rotation(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        toolaria._store.put("FIRST", "send_email", session_id="s1",
                             label="credential")
        new_key_file = tmp_path / "new.key"
        toolaria._store.rotate_key(str(new_key_file))
        # New put must use the NEW key.
        bid = toolaria._store.put("SECOND", "send_email", session_id="s1",
                                   label="credential")
        new_fernet = Fernet(new_key_file.read_bytes())
        raw_cipher = (toolaria._store.blob_dir / bid).read_bytes()
        # New key can decrypt this new blob.
        plaintext = new_fernet.decrypt(raw_cipher).decode("utf-8")
        assert plaintext == "SECOND", (
            "T4.3: post-rotation puts must encrypt under the new key"
        )
        # And old key cannot (proves it really used the new key).
        old_fernet = Fernet(old_key_file.read_bytes())
        with pytest.raises(Exception):
            old_fernet.decrypt(raw_cipher)


# ── Abort-safety: partial pass rolls back ─────────────────────────────────


class TestKeyRotationAbortSafety:
    """A failure mid-rotation leaves the store usable under the OLD key.

    Forced by injecting a read failure on a specific blob AFTER one
    rotation step succeeds. The remaining blob stays encrypted with the
    old key, the ledger has no row, and reads still decrypt with the
    old key.
    """

    def test_partial_pass_failure_keeps_old_key_active(
        self, plugin, toolaria, tmp_path, monkeypatch,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        b1 = toolaria._store.put("FIRST-SECRET", "send_email",
                                   session_id="s1", label="credential")
        b2 = toolaria._store.put("SECOND-SECRET", "send_email",
                                   session_id="s1", label="credential")
        # Patch _atomic_write_blob on the INSTANCE so the staticmethod
        # binding isn't disturbed. Raise on the second call so the
        # rotation aborts after writing one blob — the abort path
        # must re-encrypt that blob back under the old key.
        original = toolaria._store._atomic_write_blob
        calls = {"n": 0}

        def flaky_write(bpath, data):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated mid-rotation failure")
            return original(bpath, data)

        monkeypatch.setattr(toolaria._store, "_atomic_write_blob",
                             staticmethod(flaky_write))
        new_key_file = tmp_path / "new.key"
        with pytest.raises(RuntimeError):
            toolaria._store.rotate_key(str(new_key_file))
        # The store still uses the OLD key.
        assert toolaria._store.cfg.get("toolaria_key_file") == str(old_key_file), (
            "abort-safety: cfg must remain on the old key path after a "
            "failed rotation"
        )
        # Both blobs still decrypt under the old key.
        r1 = toolaria._fetch(args={"id": b1, "mode": "full"},
                               session_id="s1")
        r2 = toolaria._fetch(args={"id": b2, "mode": "full"},
                               session_id="s1")
        assert r1 == "FIRST-SECRET"
        assert r2 == "SECOND-SECRET", (
            "abort-safety: a failed rotation must leave the original blob "
            "bytes decryptable with the old key"
        )
        # No ledger row was written for a failed pass.
        rows = _read_key_rotations(Path(toolaria._store.store_path))
        assert rows == [], (
            f"abort-safety: a failed rotation must NOT emit a ledger row; "
            f"got {rows!r}"
        )

    def test_rotate_with_no_encrypted_blobs_is_a_no_op(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        # No credential blob exists.
        new_key_file = tmp_path / "new.key"
        count = toolaria._store.rotate_key(str(new_key_file))
        assert count == 0, (
            f"rotation with no encrypted blobs must report count=0; "
            f"got {count!r}"
        )
        # One ledger row still records the no-op rotation for audit.
        rows = _read_key_rotations(Path(toolaria._store.store_path))
        assert len(rows) == 1
        assert rows[0]["count"] == 0


# ── When there are no encrypted blobs at all ─────────────────────────────


class TestRotateWithoutExistingEncryption:
    """rotate_key before any encrypted blob exists: still completes,
    rotates cfg, writes one ledger row."""

    def test_rotate_before_any_encrypted_blob_succeeds(
        self, plugin, toolaria, tmp_path,
    ):
        old_key_file = tmp_path / "old.key"
        _enable_encryption(toolaria, old_key_file)
        # No puts at all.
        new_key_file = tmp_path / "new.key"
        count = toolaria._store.rotate_key(str(new_key_file))
        assert count == 0
        assert toolaria._store.cfg.get("toolaria_key_file") == str(new_key_file)
        rows = _read_key_rotations(Path(toolaria._store.store_path))
        assert len(rows) == 1
        assert rows[0]["count"] == 0

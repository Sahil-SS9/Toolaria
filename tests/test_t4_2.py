"""Phase 4 — T4.2: Fernet at-rest encryption for credential-labelled blobs.

TDD discipline: every test was written before the production code it pins.

Contract (T4.2, approved plan D8 + subagent brief):

  - ``cfg["toolaria_key_file"]`` is the gate. Unset/empty ⇒ inert
    plaintext (byte-identical to pre-T4.2). This is the safe default
    so a fresh install carries no new behaviour and no new perms
    requirements until the operator opts in.
  - When the key file IS configured AND a put resolves to
    ``final_label == "credential"``, the on-disk payload is Fernet
    ciphertext; the index entry carries ``"enc": True``.
  - If the key file is configured but missing on disk, generate a
    Fernet key (``Fernet.generate_key()``), write it 0600, log
    WARNING loudly. This is the auto-bootstrap path so an operator
    does not need to run ``cryptography`` manually to enable the
    feature.
  - Read paths — ``blob_text``, ``rescuer_fetch`` (all modes),
    ``passref`` expansion — decrypt transparently when the entry
    carries ``enc: True``. Ciphertext is only ever at rest.
  - Missing or corrupt key ⇒ deterministic fail-safe refusal marker,
    NEVER partial plaintext.

Constraints honoured:
  - encryption-OFF + no-key = byte-identical to today (G2).
  - new marker strings only (no pre-existing markers moved).
  - perf guard G5 (≤5%) still holds — Fernet encrypt of a 17.7KB
    blob is sub-millisecond.
  - tombstone (TTL + size sweep) preserves the ``enc`` marker so
    T4.4 audit can report historical encryption coverage.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

try:
    from cryptography.fernet import Fernet
    _HAVE_FERNET = True
except ImportError:  # pragma: no cover - guarded by test runner
    _HAVE_FERNET = False


# Deterministic fail-safe refusal marker returned when an encrypted
# blob's key is missing or corrupt. Tests pin against the literal shape.
ENCRYPTION_FAIL_MARKER_PREFIX = "[Toolaria: encrypted blob "
ENCRYPTION_FAIL_MARKER_SUFFIX = (
    " cannot be decrypted (missing/corrupt key); content withheld]"
)


# Skip the entire module if cryptography is unavailable — the canonical
# runner installs it via ``uv run --with cryptography``.
pytestmark = pytest.mark.skipif(
    not _HAVE_FERNET,
    reason="cryptography.fernet unavailable (uv run --with cryptography)",
)


# ── helpers ────────────────────────────────────────────────────────────────


def _write_key(cfg_path: Path, key: bytes | None = None) -> bytes:
    """Write a Fernet key file at 0600. Returns the key bytes."""
    key = key or Fernet.generate_key()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(key)
    os.chmod(cfg_path, 0o600)
    return key


def _enable_encryption(toolaria, key_file: Path) -> bytes:
    """Configure the store with toolaria_key_file. Returns the key bytes."""
    key = _write_key(key_file)
    toolaria._cfg["toolaria_key_file"] = str(key_file)
    # Drop any cached Fernet so the store picks up the new key on first use.
    fernet = getattr(toolaria._store, "_fernet", None)
    if fernet is not None:
        toolaria._store._fernet = None
    return key


# ── Inert default ──────────────────────────────────────────────────────────


class TestInertDefault:
    """No ``toolaria_key_file`` configured ⇒ behaviour identical to pre-T4.2."""

    def test_no_key_file_means_plaintext_at_rest(self, plugin, toolaria, tmp_path):
        """Without a key file, a credential-labelled put stores plaintext.

        G2 regression guard: the existing byte-identical contract for
        enforcement-off / public / no-encryption must survive.
        """
        content = "SECRET-PLAINTEXT-XYZ " * 50
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        # Read the raw file bytes (not via fetch) — must be plaintext.
        raw = (toolaria._store.blob_dir / bid).read_bytes()
        assert raw == content.encode("utf-8"), (
            "T4.2 inert default: without toolaria_key_file the on-disk bytes "
            "must be plaintext, not ciphertext"
        )
        # Index entry must NOT carry the enc marker.
        idx = toolaria._store._load_idx("s1")
        assert idx["blobs"][bid].get("enc") is not True, (
            "no key file configured ⇒ enc marker must not be stamped"
        )

    def test_no_key_file_fetch_returns_plaintext(self, plugin, toolaria):
        """fetch on a credential blob without key file returns the
        plaintext (enforcement-off default)."""
        content = "TOKEN-PLAIN " * 50
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content

    def test_no_key_file_public_blob_untouched(self, plugin, toolaria):
        """Public blobs must remain byte-identical whether or not a key file
        is configured (inert default must be truly inert)."""
        content = "PUBLIC-DATA " * 100
        bid = toolaria._store.put(content, "web_search", session_id="s1")
        raw = (toolaria._store.blob_dir / bid).read_bytes()
        assert raw == content.encode("utf-8")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content


# ── Encryption on: roundtrip + ciphertext-at-rest ──────────────────────────


class TestEncryptionRoundtrip:
    """Key file configured + credential label ⇒ ciphertext at rest,
    transparent decrypt on read."""

    def test_credential_put_writes_ciphertext_when_key_configured(
        self, plugin, toolaria, tmp_path,
    ):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "SECRET-XYZ-abc-123 " * 30
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        # On-disk must NOT contain the plaintext (Fernet ciphertext has
        # no overlap with the input; the base64 alphabet excludes
        # most printable ASCII chars, but a robust check is "does NOT
        # contain a known plaintext substring").
        raw = (toolaria._store.blob_dir / bid).read_bytes()
        assert b"SECRET-XYZ-abc-123" not in raw, (
            "T4.2: plaintext must NEVER appear at rest when encryption is on"
        )
        assert len(raw) > len(content), (
            "Fernet ciphertext is larger than plaintext (HMAC + IV); a "
            "shorter ciphertext suggests encryption did not happen"
        )

    def test_index_entry_carries_enc_marker(self, plugin, toolaria, tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        bid = toolaria._store.put("PASSWORD abc123", "send_email",
                                   session_id="s1", label="credential")
        idx = toolaria._store._load_idx("s1")
        assert idx["blobs"][bid].get("enc") is True, (
            "encrypted credential entries must carry enc=True so reads "
            "know to decrypt"
        )

    def test_fetch_decrypts_transparently(self, plugin, toolaria, tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "DECRYPT-ME-NOW " * 40
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        # fetch(full) returns plaintext.
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content, (
            "T4.2: encrypted credential blob fetch must decrypt and "
            "return plaintext"
        )

    def test_blob_text_decrypts_transparently(self, plugin, toolaria, tmp_path):
        """blob_text() is the path passref expansion uses; must decrypt too."""
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "PASS-REF-SECRET " * 30
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        assert toolaria._store.blob_text(bid) == content, (
            "blob_text must return plaintext for an encrypted entry so "
            "passref expansion carries plaintext to the destination tool"
        )

    def test_passref_expansion_decrypts_credential_blob(
        self, plugin, toolaria, tmp_path,
    ):
        """End-to-end: put credential blob encrypted, passref expansion
        delivers plaintext to the destination tool."""
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "TOKEN-ABCDEF " * 30
        bid = toolaria._store.put(content, "send_email",
                                   session_id="test-s", label="credential")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="summarise",
                  args={"x": f"tla:{bid}"}, session_id="test-s")
        assert out is not None
        assert "TOKEN-ABCDEF" in out["args"]["x"], (
            "T4.2: passref expansion of an encrypted credential blob "
            "must deliver plaintext to the destination tool"
        )

    def test_public_blob_unaffected_when_encryption_on(
        self, plugin, toolaria, tmp_path,
    ):
        """Encryption on but label=public ⇒ plaintext at rest (encryption
        is credential-only)."""
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "PUBLIC-DATA " * 100
        bid = toolaria._store.put(content, "web_search", session_id="s1")
        raw = (toolaria._store.blob_dir / bid).read_bytes()
        assert raw == content.encode("utf-8"), (
            "non-credential blobs must NOT be encrypted even when a key "
            "file is configured"
        )
        idx = toolaria._store._load_idx("s1")
        assert idx["blobs"][bid].get("enc") is not True


# ── Key file auto-generation ──────────────────────────────────────────────


class TestKeyFileBootstrap:
    """Configured but missing key file ⇒ auto-generate, 0600, WARNING."""

    def test_missing_key_file_is_generated_with_0600_perms(
        self, plugin, toolaria, tmp_path,
    ):
        key_file = tmp_path / "subdir" / "toolaria.key"
        # Configure with a path that does NOT exist yet.
        toolaria._cfg["toolaria_key_file"] = str(key_file)
        # Trigger init that touches the key file by performing a put.
        assert not key_file.exists(), "precondition: key file must not exist"
        toolaria._store.put("FIRST-CRED", "send_email", session_id="s1",
                             label="credential")
        assert key_file.exists(), (
            "missing key file must be auto-generated on first use"
        )
        mode = key_file.stat().st_mode & 0o777
        assert mode == 0o600, (
            f"auto-generated key file must be 0600, got {oct(mode)}"
        )
        # And the generated key must be a valid Fernet key.
        assert _HAVE_FERNET
        Fernet(key_file.read_bytes())  # raises if malformed

    def test_generated_key_round_trips_a_credential_blob(
        self, plugin, toolaria, tmp_path,
    ):
        key_file = tmp_path / "toolaria.key"
        toolaria._cfg["toolaria_key_file"] = str(key_file)
        content = "ROUNDTRIP " * 30
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        # Now rotate the in-memory Fernet to force a re-read of the
        # freshly written key file.
        toolaria._store._fernet = None
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content


# ── Fail-safe: missing/corrupt key returns refusal marker ─────────────────


class TestFailSafeRefusal:
    """Missing/corrupt key returns the deterministic refusal marker,
    never partial plaintext."""

    def test_missing_key_file_returns_refusal_marker(
        self, plugin, toolaria, tmp_path,
    ):
        key_file = tmp_path / "toolaria.key"
        # Configure the store to use this (not-yet-existing) key file
        # path so the first credential put auto-generates it. Without
        # this line the store would stay inert (no key configured) and
        # the test would degenerate to a plain plaintext round-trip.
        toolaria._cfg["toolaria_key_file"] = str(key_file)
        # Put a credential blob, which auto-generates the key + encrypts.
        content = "SECRET-XYZ " * 30
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        assert key_file.exists(), (
            "T4.2: the first credential put must auto-generate the key "
            "file at the configured path (missing-key bootstrap path)"
        )
        # Now delete the key file to simulate the missing-key failure mode.
        key_file.unlink()
        # Drop cached Fernet so the store attempts to reload.
        toolaria._store._fernet = None
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r.startswith(ENCRYPTION_FAIL_MARKER_PREFIX), (
            f"missing-key fetch must return the deterministic refusal "
            f"marker; got {r!r}"
        )
        assert bid in r, (
            f"refusal marker must reference the blob id so the operator "
            f"can locate the failed blob; got {r!r}"
        )
        assert ENCRYPTION_FAIL_MARKER_SUFFIX in r
        # Never partial plaintext.
        assert "SECRET-XYZ" not in r, (
            "fail-safe: missing key MUST NOT leak plaintext"
        )

    def test_corrupt_key_file_returns_refusal_marker(
        self, plugin, toolaria, tmp_path,
    ):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "TOPSECRET-12345 " * 20
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        # Corrupt the key file with garbage.
        key_file.write_bytes(b"this-is-not-a-valid-fernet-key-XXXXXXXX")
        toolaria._store._fernet = None
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r.startswith(ENCRYPTION_FAIL_MARKER_PREFIX), (
            f"corrupt-key fetch must return the deterministic refusal "
            f"marker; got {r!r}"
        )
        assert "TOPSECRET-12345" not in r, (
            "fail-safe: corrupt key MUST NOT leak plaintext"
        )

    def test_missing_key_blob_text_returns_refusal_marker(
        self, plugin, toolaria, tmp_path,
    ):
        """blob_text() is passref's read path; on key failure it must
        surface the refusal marker (not None) so passref expansion
        carries an honest refusal rather than the generic 'unavailable'
        line."""
        key_file = tmp_path / "toolaria.key"
        toolaria._cfg["toolaria_key_file"] = str(key_file)
        content = "SECRET-PASSREF " * 30
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        key_file.unlink()
        toolaria._store._fernet = None
        r = toolaria._store.blob_text(bid)
        assert isinstance(r, str) and r.startswith(ENCRYPTION_FAIL_MARKER_PREFIX), (
            f"blob_text on key-failure must return the refusal marker "
            f"string so passref expands an honest refusal; got {r!r}"
        )


# ── tombstone preservation: enc marker survives sweep ─────────────────────


class TestTombstoneEncryptionMarker:
    """An encrypted credential blob that gets swept to a tombstone must
    carry the ``enc`` marker forward so the T4.4 audit can attribute
    encryption coverage historically."""

    def test_enc_marker_preserved_across_ttl_sweep(self, plugin, toolaria,
                                                     tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        bid = toolaria._store.put("EXPIRED-SECRET", "send_email",
                                   session_id="s1", label="credential")
        # Age past TTL.
        idx = toolaria._store._load_idx("s1")
        ttl = toolaria._store.cfg.get("ttl_hours", 1) * 3600
        idx["blobs"][bid]["t"] = time.time() - (ttl + 3600)
        toolaria._store._save_idx(idx, "s1")
        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        assert "swept_at" in entry, "precondition: blob must be swept"
        assert entry.get("enc") is True, (
            "tombstone must preserve enc=True so historical audit "
            "counts encrypted credential blobs correctly"
        )

    def test_enc_marker_preserved_across_size_sweep(self, plugin, toolaria,
                                                      tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        # Pad the credential blob so its on-disk ciphertext size
        # comfortably exceeds the 1KB size cap. Fernet ciphertext is
        # ~100 bytes of overhead + ciphertext; a 2KB plaintext
        # encrypts to ~2.1KB on disk, which is over the cap.
        content = "SIZE-SWEPT-SECRET-PADDING " * 200
        bid = toolaria._store.put(content, "send_email",
                                   session_id="s1", label="credential")
        # 1KB size cap to force a size sweep on this single blob.
        toolaria._store.cfg["max_store_mb"] = 0  # 0 bytes cap → sweep all
        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        assert "swept_at" in entry, (
            "size sweep must convert the encrypted entry to a tombstone"
        )
        assert entry.get("enc") is True, (
            "T4.2: size-sweep tombstone must preserve enc=True"
        )


# ── Cross-tool coverage: other read paths decrypt ─────────────────────────


class TestSearchAndRangeDecrypt:
    """Non-``full`` fetch modes (search, range, grep) must also decrypt
    before slicing."""

    def test_range_decrypts_credential_blob(self, plugin, toolaria,
                                              tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "LINE\n" * 50
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "range",
                                   "start": 0, "count": 5},
                             session_id="s1")
        assert "LINE" in r, (
            "range mode on an encrypted credential blob must decrypt "
            "before slicing"
        )

    def test_grep_decrypts_credential_blob(self, plugin, toolaria,
                                             tmp_path):
        key_file = tmp_path / "toolaria.key"
        _enable_encryption(toolaria, key_file)
        content = "alpha\nbeta\ngamma\n" + ("SECRET-HIT\n" * 5)
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "grep",
                                   "pattern": "SECRET-HIT"},
                             session_id="s1")
        assert "SECRET-HIT" in r, (
            "grep mode on an encrypted credential blob must decrypt "
            "before matching"
        )

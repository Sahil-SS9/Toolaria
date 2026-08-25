"""Regression tests for the 2026-08-25 independent-review correction gate.

One test group per reproduced reviewer probe:

  RF1 (CRITICAL)  search/chunks/vectors must NOT write plaintext
                  sidecars for encrypted blobs.
  RF2 (CRITICAL)  a public-first blob must be atomically re-encrypted
                  when its content-owned label upgrades to credential.
  RF3 (HIGH)      configured key + missing cryptography = credential
                  writes REFUSED, never plaintext (fail closed).
  RF4 (HIGH)      /toolaria-rotate-key updates config.yaml durably so
                  decryption survives a restart.
  RF5 (MEDIUM)    negative enc-state is never cached; cross-process
                  upgrades are discovered.

v2: every BlobStore gets an explicit isolated store_path (the default
~/.hermes/toolaria must never be touched by tests). RF3 runs in-process
by monkeypatching blobstore._HAVE_FERNET (no separate interpreter).
"""
from __future__ import annotations

import pytest

import blobstore as bs_mod
from blobstore import BlobStore, _EncryptionUnavailable

SECRET = "sk-live-reviewer-fix-9911"
REAL_KEY = "sk-live9f8e7d6c5b4a3210"


@pytest.fixture
def mk_store(tmp_path):
    """Factory: isolated BlobStore with optional key file.

    NOTE: BlobStore takes a FLAT config (store_path at the top level),
    not the plugin's {"toolaria": {...}} wrapper — the plugin merges
    defaults + user cfg into a flat dict before constructing it.
    """
    def _make(with_key=False, store_name="store", **extra):
        cfg = {"store_path": str(tmp_path / store_name), **extra}
        if with_key:
            cfg["toolaria_key_file"] = str(tmp_path / f"{store_name}.key")
        return BlobStore(cfg)
    return _make


needs_crypto = pytest.mark.skipif(
    not bs_mod._HAVE_FERNET,
    reason="requires the cryptography package (encryption tier)")


class TestRF1NoSidecarsForEncrypted:
    """Every sidecar-writing path honours the encrypted-blob ban."""

    @needs_crypto
    def test_outline_and_chunks_write_nothing_for_encrypted(
            self, mk_store, tmp_path):
        store = mk_store(with_key=True)
        bid = store.put(REAL_KEY, "web_search", session_id="s",
                        label="credential")
        # drive every sidecar-producing path directly
        store._chunks(bid, REAL_KEY)
        store._outline(bid, REAL_KEY)
        files = list(store.sidecar_dir.glob(f"{bid}*"))
        assert not files, (
            f"RF1: outline/chunks wrote sidecars for encrypted blob: "
            f"{[p.name for p in files]}")

    @needs_crypto
    def test_build_outline_returns_structure_without_writing(
            self, mk_store):
        store = mk_store(with_key=True)
        bid = store.put(REAL_KEY, "web_search", session_id="s",
                        label="credential")
        out = store.build_outline(bid, REAL_KEY)
        assert isinstance(out, dict), (
            "RF1: build_outline must still return the outline dict "
            "(caller contract unchanged); only the disk write is skipped")
        assert not list(store.sidecar_dir.glob(f"{bid}*"))

    def test_plaintext_blob_still_gets_sidecars(self, mk_store):
        # guard against over-suppression: normal path unchanged
        store = mk_store()
        bid = store.put("public data line", "web_search",
                        session_id="s", label="public")
        store.build_outline(bid, "public data line")
        assert list(store.sidecar_dir.glob(f"{bid}*")), (
            "RF1 regression guard: plaintext blobs must keep their "
            "outline sidecar (behaviour unchanged)")

    @needs_crypto
    def test_stale_sidecars_removed_after_upgrade(self, mk_store,
                                                   tmp_path):
        # public-first: sidecar exists
        plain = mk_store(store_name="up")
        content = "data\n" + REAL_KEY
        bid = plain.put(content, "web_search", session_id="s",
                        label="public")
        plain.build_outline(bid, content)
        assert list(plain.sidecar_dir.glob(f"{bid}*")), "precondition"

        # credential upgrade under encryption -> ciphertext at rest;
        # the rescue-path gate now deletes stale plaintext sidecars.
        enc = mk_store(with_key=True, store_name="up")
        enc.put(content, "web_search", session_id="s2",
                label="credential")
        assert not list(enc.sidecar_dir.glob(f"{bid}*"))

    def test_uncertain_disk_state_suppresses_sidecar_write(
            self, mk_store, monkeypatch):
        """An unreadable blob must fail closed: no plaintext cache write."""
        store = mk_store()
        bid = store.put("public data", "web_search", session_id="s",
                        label="public")
        monkeypatch.setattr(store, "_blob_encrypted", lambda _bid: False)
        target = store.blob_dir / bid
        path_type = type(target)
        original_read_bytes = path_type.read_bytes

        def unreadable(self):
            if self == target:
                raise PermissionError("simulated unreadable blob")
            return original_read_bytes(self)

        monkeypatch.setattr(path_type, "read_bytes", unreadable)
        assert store._sidecars_forbidden(bid) is True


class TestRF2UpgradeEncryptsInPlace:
    """Public-first content encrypts when its label upgrades."""

    @needs_crypto
    def test_public_first_then_credential_reencrypts(self, mk_store,
                                                      tmp_path):
        plain = mk_store(store_name="u1")
        bid = plain.put(REAL_KEY, "web_search", session_id="p1",
                        label="public")
        bpath = plain.blob_dir / bid
        assert bpath.read_text() == REAL_KEY, "precondition: plaintext"

        enc = mk_store(with_key=True, store_name="u1")
        out_bid = enc.put(REAL_KEY, "web_search", session_id="p2",
                          label="credential")
        assert out_bid == bid, "same content maps to same bid"
        raw = bpath.read_bytes()
        assert REAL_KEY.encode() not in raw, (
            "RF2: public-first blob stayed plaintext after its "
            "content-owned label upgraded to credential under an "
            "active key")
        assert enc._load_idx("p2")["blobs"][bid].get("enc") is True
        assert enc.blob_text(bid) == REAL_KEY

    def test_upgrade_refuses_when_key_dir_unavailable(self, mk_store,
                                                       tmp_path):
        plain = mk_store(store_name="u2")
        bid = plain.put(REAL_KEY, "web_search", session_id="q",
                        label="public")

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        broken = BlobStore({
            "store_path": str(tmp_path / "u2"),
            "toolaria_key_file": str(blocker / "sub" / "k.key")})
        with pytest.raises(_EncryptionUnavailable):
            broken.put(REAL_KEY, "web_search", session_id="q2",
                       label="credential")
        # original bytes untouched (still plaintext, still readable)
        assert (plain.blob_dir / bid).read_text() == REAL_KEY


class TestRF3FailClosedWithoutCrypto:
    """Configured key + missing cryptography => refuse, never plaintext.

    Runs in-process by flipping blobstore._HAVE_FERNET — exactly the
    condition an operator without the package hits.
    """

    def test_credential_put_refused_without_fernet(self, mk_store,
                                                    tmp_path,
                                                    monkeypatch):
        monkeypatch.setattr(bs_mod, "_HAVE_FERNET", False)
        store = mk_store(with_key=True, store_name="nf")
        with pytest.raises(_EncryptionUnavailable):
            store.put(REAL_KEY, "web_search", session_id="nf",
                      label="credential")
        leaked = any(REAL_KEY.encode() in f.read_bytes()
                     for f in store.blob_dir.glob("*") if f.is_file())
        assert not leaked, (
            "RF3: credential content reached disk as plaintext while "
            "encryption was configured but unavailable")

    def test_no_key_configured_stays_inert(self, mk_store, monkeypatch):
        # inert contract preserved: no key configured + no crypto =>
        # public puts behave exactly as before.
        monkeypatch.setattr(bs_mod, "_HAVE_FERNET", False)
        store = mk_store(with_key=False, store_name="inert")
        bid = store.put("plain-public-bytes", "web_search",
                        session_id="i", label="public")
        assert (store.blob_dir / bid).read_text() == "plain-public-bytes"


class TestRF4DurableRotation:
    """The rotate-key command coordinates blobs AND durable config."""

    @needs_crypto
    def test_rotate_command_updates_config_yaml(
            self, base_cfg, toolaria, monkeypatch, tmp_path):
        mod = toolaria

        fake_home = tmp_path / "home"
        hermes = fake_home / ".hermes"
        hermes.mkdir(parents=True)
        cfg_yaml = hermes / "config.yaml"
        old_key = tmp_path / "old.key"
        old_key.write_bytes(__import__(
            "cryptography.fernet", fromlist=["Fernet"]).Fernet
            .generate_key())
        # The '#' proves the persisted value is YAML-quoted rather than
        # interpreted as an inline comment.
        new_key = tmp_path / "new # key.key"
        cfg_yaml.write_text(f"toolaria:\n"
                            f"  toolaria_key_file: {old_key}\n")
        __import__("os").chmod(cfg_yaml, 0o600)

        real_expanduser = __import__("pathlib").Path.expanduser

        def fake_expanduser(self):
            s = str(self)
            if s.startswith("~"):
                return real_expanduser(
                    type(self)(s.replace("~", str(fake_home))))
            return self

        import pathlib
        monkeypatch.setattr(pathlib.Path, "expanduser", fake_expanduser)
        monkeypatch.setenv("HERMES_HOME", str(hermes))

        cfg = dict(base_cfg)
        cfg["toolaria_key_file"] = str(old_key)
        store = BlobStore(cfg)
        bid = store.put(REAL_KEY, "web_search", session_id="rot",
                        label="credential")
        monkeypatch.setattr(mod, "_store", store, raising=False)
        monkeypatch.setattr(mod, "_cfg",
                            {"toolaria_key_file": str(old_key)},
                            raising=False)

        out = mod._rotate_key_cmd(str(new_key))
        assert "Key rotation complete" in out, out
        yaml_text = cfg_yaml.read_text()
        assert str(new_key) in yaml_text, (
            "RF4: successful rotation did not durably update "
            "config.yaml — restart would break decryption")
        import json
        scalar = next(
            line.split(":", 1)[1].strip()
            for line in yaml_text.splitlines()
            if line.strip().startswith("toolaria_key_file:")
        )
        assert json.loads(scalar) == str(new_key), (
            "RF4: key path was not encoded as a safely quoted YAML scalar")
        assert cfg_yaml.stat().st_mode & 0o777 == 0o600, (
            "RF4: replacing config.yaml weakened its 0600 permissions")
        assert "sudo systemctl restart" in out, (
            "RF4: operator must be told to restart")
        fresh = BlobStore({**base_cfg, "toolaria_key_file": str(new_key)})
        assert fresh.blob_text(bid) == REAL_KEY

    def test_rotation_failure_is_not_reported_as_partial_success(
            self, toolaria, monkeypatch, tmp_path):
        hermes = tmp_path / "hermes"
        hermes.mkdir()
        old_key = tmp_path / "old.key"
        cfg_yaml = hermes / "config.yaml"
        cfg_yaml.write_text(
            f"toolaria:\n  toolaria_key_file: {old_key}\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes))

        class BrokenStore:
            store_path = str(tmp_path / "store")

            @staticmethod
            def rotate_key(_new_path):
                raise RuntimeError("simulated pre-rotation failure")

        monkeypatch.setattr(toolaria, "_store", BrokenStore())
        monkeypatch.setattr(
            toolaria, "_cfg", {"toolaria_key_file": str(old_key)})
        out = toolaria._rotate_key_cmd(str(tmp_path / "new.key"))
        assert out.startswith("Error: rotation failed"), out
        assert "PARTIAL SUCCESS" not in out


class TestRF5NegativeCacheNeverStale:
    """A cached negative must never mask a later encryption."""

    @needs_crypto
    def test_false_not_cached_across_processes(self, mk_store, tmp_path):
        store_a = mk_store(store_name="x")
        bid = store_a.put("upgrade-me-" + REAL_KEY, "web_search",
                          session_id="a")
        assert store_a._blob_encrypted(bid) is False
        assert bid not in store_a._enc_cache, (
            "RF5: negative result was cached — it would go stale when "
            "another process upgrades the blob")

        store_b = mk_store(with_key=True, store_name="x")
        store_b.put("upgrade-me-" + REAL_KEY, "web_search",
                    session_id="b", label="credential")

        # process A re-checks: scan must run again and find enc=True
        assert store_a._blob_encrypted(bid) is True, (
            "RF5: cross-process upgrade not discoverable from process A")

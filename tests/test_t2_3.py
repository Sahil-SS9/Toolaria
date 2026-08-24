"""Phase 2 data-governance tests — T2.3.

TDD discipline: every test was written before the implementation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from blobstore import BlobStore



# Marker returned (and logged) when a credential-labelled blob's full
# content is refused via rescuer_fetch full mode or passref expansion.
CREDENTIAL_REFUSE_MARKER_PREFIX = (
    "[Toolaria: credential-labelled blob "
)
CREDENTIAL_REFUSE_MARKER_SUFFIX = (
    " withheld; use range/grep slices or add destination to "
    "credential_destinations]"
)


class TestCredentialEnforcementOffByDefault:
    """enforcement_enabled defaults to FALSE — behavior identical to today."""

    def test_enforcement_enabled_defaults_to_false(self, toolaria, base_cfg,
                                                     fake_ctx_cls):
        cfg = dict(base_cfg)
        cfg.pop("enforcement_enabled", None)
        fc = fake_ctx_cls({"toolaria": cfg})
        toolaria.register(fc)
        assert toolaria._store.cfg.get("enforcement_enabled", False) is False, (
            "T2.3: enforcement MUST default off — Sahil reviews the audit "
            "data before any enforcement flip"
        )

    def test_public_blob_fetch_full_unaffected_when_off(self, plugin, toolaria):
        # Default config has enforcement off; public blob fetch full works.
        content = "PUBLIC DATA " * 100  # under full_fetch_max_chars
        bid = toolaria._store.put(content, "web_search", session_id="s1")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content, (
            "T2.3: with enforcement off, public blobs must be byte-identical "
            "to pre-T2 behaviour (G2 regression guard)"
        )

    def test_credential_blob_unaffected_when_off(self, plugin, toolaria):
        """Even with a credential label, off-mode means full content
        flows — enforcement flip is a separate gate."""
        content = "TOKEN-XYZ " * 100
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content


class TestCredentialEnforcementOn:
    """enforcement_enabled=true activates the credential gates."""

    def test_credential_full_fetch_returns_deterministic_marker(
        self, plugin, toolaria
    ):
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET " * 50, "send_email",
                                   session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        expected = (f"{CREDENTIAL_REFUSE_MARKER_PREFIX}{bid}"
                    f"{CREDENTIAL_REFUSE_MARKER_SUFFIX}")
        assert r == expected, (
            f"credential full-fetch must return the exact refusal marker; "
            f"got {r!r}"
        )

    def test_credential_range_fetch_still_works(self, plugin, toolaria):
        """range is a slice — the only acceptable fetch mode for credential."""
        toolaria._cfg["enforcement_enabled"] = True
        content = "SECRET-LINE\n" * 50
        bid = toolaria._store.put(content, "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 5}, session_id="s1")
        assert "SECRET-LINE" in r
        assert "withheld" not in r.lower()

    def test_credential_stat_fetch_still_works(self, plugin, toolaria):
        """stat never returns content; should always work."""
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET", "send_email", session_id="s1",
                                   label="credential")
        r = toolaria._fetch(args={"id": bid, "mode": "stat"},
                             session_id="s1")
        assert "blob:" in r
        assert "withheld" not in r.lower()

    def test_credential_passref_into_deny_dst_returns_marker(
        self, plugin, toolaria
    ):
        """With credential_destinations allowlist empty (default),
        expansion into ANY destination returns the marker — content
        never reaches the downstream tool."""
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("TOKEN " * 50, "send_email",
                                   session_id="test-s",
                                   label="credential")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        assert out is not None
        v = out["args"]["x"]
        assert "TOKEN" not in v, (
            "credential blob content leaked into summarise; default-empty "
            "credential_destinations must deny all"
        )

    def test_credential_passref_allowlist_enables_expansion(
        self, plugin, toolaria
    ):
        """An explicit allowlist entry permits expansion into that one
        destination (controlled-release pattern)."""
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._cfg["credential_destinations"] = ["vault_browser"]
        bid = toolaria._store.put("TOKEN " * 50, "send_email",
                                   session_id="test-s",
                                   label="credential")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="vault_browser", args={"x": f"tla:{bid}"})
        assert out is not None
        # Allowlist entry wins → expansion happens.
        assert "TOKEN" in out["args"]["x"]

    def test_credential_allowlist_is_strict(self, plugin, toolaria):
        """Allowlisting ONE tool must not bleed into siblings."""
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._cfg["credential_destinations"] = ["vault_browser"]
        bid = toolaria._store.put("TOKEN " * 50, "send_email",
                                   session_id="test-s",
                                   label="credential")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="publish", args={"x": f"tla:{bid}"})
        assert out is not None
        assert "TOKEN" not in out["args"]["x"], (
            "allowlist must be exact-match; a non-listed tool cannot inherit"
        )

    def test_public_blob_untouched_by_enforcement(self, plugin, toolaria):
        """G2 regression guard: public blobs are byte-identical
        regardless of enforcement_enabled."""
        toolaria._cfg["enforcement_enabled"] = True
        content = "PUBLIC CONTENT " * 100
        bid = toolaria._store.put(content, "web_search", session_id="s1")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="s1")
        assert r == content
        # And passref into summarise still works (not in destinations list).
        bid2 = toolaria._store.put("PUBLIC-PASS " * 50, "web_search",
                                    session_id="test-s")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="summarise", args={"x": f"tla:{bid2}"})
        assert "PUBLIC-PASS" in out["args"]["x"]


class TestCredentialHardTTL:
    """24h hard TTL overrides hot-blob pinning for credential blobs.

    The 7-day hot_ttl_hours exemption (frequent fetches → longer life)
    must NOT apply to credential blobs. A token-bearing blob must
    expire on the same 24h schedule no matter how often it's fetched.
    """

    def test_credential_blob_24h_ttl_overrides_hot_pinning(
        self, plugin, toolaria
    ):
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._cfg["hot_ttl_hours"] = 168  # 7-day hot exemption
        bid = toolaria._store.put("TOKEN " * 50, "send_email",
                                   session_id="s1",
                                   label="credential")
        # Age to 25h — past the 24h hard TTL but well under 168h hot TTL.
        idx = toolaria._store._load_idx("s1")
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "s1")
        # Bump the in-memory fetch log to a value that would normally
        # promote to hot (>= hot_fetch_threshold = 3.0). Without
        # enforcement, this would extend effective_ttl to hot_ttl_hours.
        from blobstore import BlobStore
        key = (BlobStore._safe_sid("s1"), bid)
        now = time.time()
        toolaria._store._fetch_log[key] = [now] * 10

        toolaria._store.lazy_sweep()

        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        assert "swept_at" in entry, (
            "credential blob must be swept past 24h regardless of "
            "fetch_count — the hot-blob exemption must not apply"
        )
        assert entry.get("label") == "credential"

    def test_non_credential_blob_still_gets_hot_pinning(self, plugin, toolaria):
        """Sanity: hot exemption still works for non-credential blobs."""
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._cfg["hot_ttl_hours"] = 168
        bid = toolaria._store.put("public " * 50, "web_search",
                                   session_id="s1",
                                   label="public")
        # Age to 25h — past ttl_hours=1 (base_cfg), but well under hot TTL.
        idx = toolaria._store._load_idx("s1")
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "s1")
        # Populate the in-memory fetch log with enough recent entries
        # that the recency-weighted count crosses hot_fetch_threshold (3.0).
        from blobstore import BlobStore
        key = (BlobStore._safe_sid("s1"), bid)
        now = time.time()
        toolaria._store._fetch_log[key] = [now] * 10

        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        assert "swept_at" not in entry, (
            "non-credential blob with high fetch_weight must survive the "
            "TTL sweep via hot-pinning"
        )

    def test_credential_24h_ttl_holds_for_enforcement_off(self, plugin, toolaria):
        """Sanity: when enforcement is OFF, the 24h hard TTL must NOT
        apply — the public blobs must keep their regular TTL behaviour."""
        toolaria._cfg["enforcement_enabled"] = False
        bid = toolaria._store.put("TOKEN " * 50, "send_email",
                                   session_id="s1",
                                   label="credential")
        idx = toolaria._store._load_idx("s1")
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "s1")
        # Populate the fetch log so hot-pinning kicks in (this is the
        # path that the off-mode must NOT override).
        from blobstore import BlobStore
        key = (BlobStore._safe_sid("s1"), bid)
        now = time.time()
        toolaria._store._fetch_log[key] = [now] * 10

        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        # With enforcement off, hot-pinning wins → survives.
        assert "swept_at" not in entry, (
            "without enforcement, hot-pinning is not overridden — "
            "credential hard TTL is an enforcement-tier feature"
        )



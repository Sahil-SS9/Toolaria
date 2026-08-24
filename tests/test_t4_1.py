"""Phase 4 — T4.1: Blob versioning.

TDD: every test in this file was written BEFORE the production code it pins.

Contract (T4.1, approved plan + subagent brief):
  - ``put()`` on the same ``(tool, session)`` builds a version chain.
  - The first put for a (tool, session) is version 1.
  - Second put of CHANGED content bumps version; new entry records
    ``supersedes`` -> old head bid; old head records ``superseded_by`` -> new bid.
  - Second put of UNCHANGED content dedups (same bid, same version).
  - Different tool or different session → independent chain (each at v1).
  - ``fetch`` accepts ``tla:<id>@N`` to address a specific version.
  - Unknown @N returns a deterministic refusal marker.
  - Old (superseded) versions remain sweepable by TTL like normal blobs.

Constraints honoured:
  - All Phase 4 gates OFF/inert by default: versioning metadata is additive,
    no behaviour change for any pre-T4.1 contract.
  - Refusal marker strings for this file are NEW (no pre-existing markers
    moved) so legacy G2 byte-identity guards stay green.
"""
from __future__ import annotations

import time

import pytest


# Deterministic refusal marker for unknown @N (exported at module level so
# tests can pin the literal shape). Implementation must use the same.
VERSION_NOT_FOUND_MARKER_PREFIX = "[Toolaria: version "
VERSION_NOT_FOUND_MARKER_SUFFIX = " not found for blob "

# Fetch-handler error prefix used for malformed @N suffixes.
_FETCH_INVALID_VERSION_PREFIX = "Error: invalid blob id "


class TestPutVersionChain:
    """put() on identical (tool, session) builds a version chain."""

    def test_first_put_for_tool_session_is_version_one(self, plugin, toolaria):
        bid = toolaria._store.put("v1-content", "web_search", session_id="s1")
        idx = toolaria._store._load_idx("s1")
        e = idx["blobs"][bid]
        assert e.get("version") == 1, (
            f"first put for a (tool, session) must be version 1, "
            f"got {e.get('version')!r}"
        )
        # No chain pointers on a fresh head.
        assert "supersedes" not in e
        assert "superseded_by" not in e

    def test_second_put_same_content_dedups_same_id_same_version(
        self, plugin, toolaria,
    ):
        b1 = toolaria._store.put("dup", "web_search", session_id="s1")
        b2 = toolaria._store.put("dup", "web_search", session_id="s1")
        assert b1 == b2, "same content + tool + session must dedup to same id"
        idx = toolaria._store._load_idx("s1")
        e = idx["blobs"][b1]
        assert e.get("version") == 1, (
            f"unchanged-content dedup must NOT bump version, "
            f"got {e.get('version')!r}"
        )
        # No chain pointers — a single-entry chain is just a head.
        assert "supersedes" not in e
        assert "superseded_by" not in e

    def test_changed_content_bumps_version_and_chains_to_predecessor(
        self, plugin, toolaria,
    ):
        b1 = toolaria._store.put("vA", "web_search", session_id="s1")
        b2 = toolaria._store.put("vB", "web_search", session_id="s1")
        assert b1 != b2, "different content must give different blob_id"
        idx = toolaria._store._load_idx("s1")
        e1 = idx["blobs"][b1]
        e2 = idx["blobs"][b2]
        assert e1.get("version") == 1
        assert e2.get("version") == 2
        assert e2.get("supersedes") == b1, (
            f"v2.supersedes must point to v1 bid {b1!r}, "
            f"got {e2.get('supersedes')!r}"
        )
        assert e1.get("superseded_by") == b2, (
            f"v1.superseded_by must point to v2 bid {b2!r}, "
            f"got {e1.get('superseded_by')!r}"
        )

    def test_three_versions_chain_in_order(self, plugin, toolaria):
        b1 = toolaria._store.put("V1", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2", "web_search", session_id="s1")
        b3 = toolaria._store.put("V3", "web_search", session_id="s1")
        idx = toolaria._store._load_idx("s1")
        e1, e2, e3 = idx["blobs"][b1], idx["blobs"][b2], idx["blobs"][b3]
        assert (e1.get("version"), e2.get("version"), e3.get("version")) == (1, 2, 3)
        assert e3.get("supersedes") == b2
        assert e2.get("supersedes") == b1
        assert e1.get("superseded_by") == b2
        assert e2.get("superseded_by") == b3

    def test_interleaved_dedup_does_not_bump(self, plugin, toolaria):
        """A dup-put between two new-content puts must not create a v2
        version that conflicts with the real v2 (which carries the new
        content). The chain must reflect the actual content timeline."""
        b1 = toolaria._store.put("V1", "web_search", session_id="s1")
        # dup of v1 (same content) — should NOT bump.
        toolaria._store.put("V1", "web_search", session_id="s1")
        # new content — should bump to v2.
        b2 = toolaria._store.put("V2", "web_search", session_id="s1")
        idx = toolaria._store._load_idx("s1")
        e1 = idx["blobs"][b1]
        e2 = idx["blobs"][b2]
        assert e1.get("version") == 1
        assert e2.get("version") == 2
        assert e2.get("supersedes") == b1

    def test_different_tool_no_chain(self, plugin, toolaria):
        b1 = toolaria._store.put("anything-A", "web_search", session_id="s1")
        b2 = toolaria._store.put("anything-B", "browser_navigate",
                                  session_id="s1")
        idx = toolaria._store._load_idx("s1")
        # Distinct logical objects; no chain between them.
        assert idx["blobs"][b1].get("version") == 1
        assert idx["blobs"][b2].get("version") == 1
        assert "supersedes" not in idx["blobs"][b1]
        assert "superseded_by" not in idx["blobs"][b1]
        assert "supersedes" not in idx["blobs"][b2]
        assert "superseded_by" not in idx["blobs"][b2]

    def test_different_session_no_chain_within_session(self, plugin, toolaria):
        b1 = toolaria._store.put("alpha", "web_search", session_id="s1")
        b2 = toolaria._store.put("alpha", "web_search", session_id="s2")
        # Same content → same blob_id (content-addressing), but each
        # session's index is a fresh chain head at version 1.
        assert b1 == b2
        idx1 = toolaria._store._load_idx("s1")
        idx2 = toolaria._store._load_idx("s2")
        assert idx1["blobs"][b1].get("version") == 1
        assert idx2["blobs"][b2].get("version") == 1


class TestFetchVersionAddressed:
    """``fetch`` accepts ``<bid>@N`` to address a specific version."""

    def test_fetch_at_n_resolves_to_that_versions_content(self, plugin, toolaria):
        b1 = toolaria._store.put("V1-DATA", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2-DATA", "web_search", session_id="s1")
        b3 = toolaria._store.put("V3-DATA", "web_search", session_id="s1")
        r1 = toolaria._fetch(args={"id": f"{b1}@1", "mode": "full"},
                             session_id="s1")
        assert r1 == "V1-DATA", (
            f"@1 must return V1 content, got {r1!r}"
        )
        r2 = toolaria._fetch(args={"id": f"{b2}@2", "mode": "full"},
                             session_id="s1")
        assert r2 == "V2-DATA", (
            f"@2 must return V2 content, got {r2!r}"
        )
        r3 = toolaria._fetch(args={"id": f"{b3}@3", "mode": "full"},
                             session_id="s1")
        assert r3 == "V3-DATA", (
            f"@3 must return V3 content, got {r3!r}"
        )

    def test_fetch_at_n_walks_chain_back_from_any_bid(self, plugin, toolaria):
        """Using a later-version bid with an earlier @N must walk the
        chain backward to the matching version."""
        b1 = toolaria._store.put("V1-X", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2-X", "web_search", session_id="s1")
        # Use b2's bid with @1 → should walk back to b1.
        r = toolaria._fetch(args={"id": f"{b2}@1", "mode": "full"},
                            session_id="s1")
        assert r == "V1-X", (
            f"@N resolution from a later bid must walk back; got {r!r}"
        )

    def test_fetch_unknown_version_returns_deterministic_marker(
        self, plugin, toolaria,
    ):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        r = toolaria._fetch(args={"id": f"{bid}@99", "mode": "full"},
                            session_id="s1")
        assert r.startswith(VERSION_NOT_FOUND_MARKER_PREFIX), (
            f"unknown @N must return the deterministic marker; got {r!r}"
        )
        assert str(99) in r, (
            f"marker must echo the requested version; got {r!r}"
        )
        assert bid in r, (
            f"marker must reference the source blob id; got {r!r}"
        )

    def test_fetch_no_at_returns_head_content(self, plugin, toolaria):
        """``rescuer_fetch(id=bid)`` without @N → the chain head
        (latest version) content. Preserves pre-T4.1 byte-identity for
        non-versioned callers (G2)."""
        toolaria._store.put("V1-only", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2-only", "web_search", session_id="s1")
        r = toolaria._fetch(args={"id": b2, "mode": "full"}, session_id="s1")
        assert r == "V2-only", (
            f"plain fetch of head must return its content byte-identically; "
            f"got {r!r}"
        )

    def test_fetch_invalid_at_n_format_returns_deterministic_error(
        self, plugin, toolaria,
    ):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        r = toolaria._fetch(args={"id": f"{bid}@notanumber", "mode": "full"},
                            session_id="s1")
        assert r.startswith(_FETCH_INVALID_VERSION_PREFIX), (
            f"non-numeric @N must return a deterministic Error; got {r!r}"
        )

    def test_fetch_at_n_outside_session_owner_returns_marker(
        self, plugin, toolaria,
    ):
        """Existing session-scoping still applies at the resolver —
        a versioned handle from another session is refused with the
        tombstone / session-not-found message."""
        toolaria._store.put("V1", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2", "web_search", session_id="s1")
        # Try to fetch s1's chain using a different session.
        r = toolaria._fetch(args={"id": f"{b2}@1", "mode": "full"},
                            session_id="other-session")
        assert "not available" in r.lower() or "not found" in r.lower() or \
            r.startswith("[Toolaria:"), (
            "versioned fetch from another session must be refused "
            "(session-scoping preserved at resolver)"
        )


class TestOldVersionSweep:
    """Old (superseded) versions remain sweepable by TTL like any normal blob."""

    def test_superseded_old_version_swept_by_ttl(self, plugin, toolaria):
        b1 = toolaria._store.put("OLD-VERSION", "web_search",
                                  session_id="s1")
        b2 = toolaria._store.put("NEW-VERSION", "web_search",
                                  session_id="s1")
        # Past TTL for the OLD version, NEW stays fresh.
        idx = toolaria._store._load_idx("s1")
        ttl = toolaria._store.cfg.get("ttl_hours", 1) * 3600
        idx["blobs"][b1]["t"] = time.time() - (ttl + 3600)
        # Keep b2 fresh.
        toolaria._store._save_idx(idx, "s1")
        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        assert "swept_at" in idx["blobs"][b1], (
            "old superseded version should be swept past its TTL"
        )
        assert "swept_at" not in idx["blobs"][b2], (
            "current head must NOT be swept"
        )
        # The tombstone may carry label + tool + version metadata.
        entry = idx["blobs"][b1]
        assert entry.get("version") == 1
        assert entry.get("tool") == "web_search"

    def test_size_sweep_handles_old_versions_without_crash(
        self, plugin, toolaria,
    ):
        # Stuff many successive versions for the same tool.
        toolaria._store.put("FIRST", "web_search", session_id="s1")
        for i in range(30):
            toolaria._store.put(f"P-{i:03d}-{'x'*100}", "web_search",
                                  session_id="s1")
        # Force size sweep with a tiny cap.
        toolaria._store.cfg["max_store_mb"] = 1
        # Sweep should not crash regardless of version complexity.
        toolaria._store.lazy_sweep()
        # Index must still be readable.
        idx = toolaria._store._load_idx("s1")
        assert isinstance(idx["blobs"], dict)


class TestPassrefVersionAddressed:
    """passref expansion supports ``tla:<id>@N`` for versioned expansion."""

    def test_passref_at_n_expands_specific_version(self, plugin, toolaria):
        b1 = toolaria._store.put("V1-DATA", "web_search", session_id="s1")
        b2 = toolaria._store.put("V2-DATA", "web_search", session_id="s1")
        mw = plugin[0].middleware["tool_request"][0]
        out_v1 = mw(tool_name="summarise",
                    args={"x": f"tla:{b1}@1"}, session_id="s1")
        out_v2 = mw(tool_name="summarise",
                    args={"x": f"tla:{b2}@2"}, session_id="s1")
        assert out_v1 is not None, "passref must return a patch"
        assert out_v2 is not None, "passref must return a patch"
        # V1 token (with @1) → only V1 content.
        assert "V1-DATA" in out_v1["args"]["x"] or \
            "V1 ONLY" in out_v1["args"]["x"] or \
            "V1-FIXTURE" in out_v1["args"]["x"], (
            f"@1 expansion must include V1 content; got "
            f"{out_v1['args']['x']!r}"
        )
        assert "V2" in out_v2["args"]["x"], (
            f"@2 expansion must include V2 content; got "
            f"{out_v2['args']['x']!r}"
        )

    def test_passref_at_n_unknown_version_returns_marker(
        self, plugin, toolaria,
    ):
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        mw = plugin[0].middleware["tool_request"][0]
        out = mw(tool_name="summarise",
                 args={"x": f"tla:{bid}@99"}, session_id="s1")
        assert out is not None, (
            "passref must return a patch even when @N is unknown"
        )
        v = out["args"]["x"]
        assert v.startswith(VERSION_NOT_FOUND_MARKER_PREFIX), (
            f"passref @N unknown must echo the version marker; got {v!r}"
        )
        assert "data" not in v or v.startswith(VERSION_NOT_FOUND_MARKER_PREFIX)

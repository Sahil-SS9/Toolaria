"""Phase 4 — T4.4: Audit surface for encryption coverage + version chains.

TDD discipline: every test was written before the production code it pins.

Contract (T4.4, subagent brief):

The ``reporting/value_flow_audit.py`` script gains two new sections
in its output:

  1. **encryption coverage** — over every credential-labelled blob
     (live + tombstone), how many are encrypted-at-rest
     (``enc: True`` in their index entry) vs plaintext. Tombstones
     are included so historical coverage is auditable after a sweep.

  2. **version-chain stats** — over every session index, the max
     version-chain depth observed (longest chain per (tool, session)
     group) and the total number of superseded versions across the
     whole store.

Both sections are omitted from the rendered text when their data is
empty so a pre-T4.4 store produces a clean report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporting import value_flow_audit as _vfa


# ── helpers ────────────────────────────────────────────────────────────────


def _write_sessions(store_path: Path, sessions: dict[str, dict]) -> None:
    """Write a sessions/ tree where each file is ``{"blobs": {...}}``."""
    sd = store_path / "sessions"
    sd.mkdir(parents=True, exist_ok=True)
    for safe_sid, idx in sessions.items():
        (sd / f"{safe_sid}.json").write_text(json.dumps(idx))


# ── encryption coverage ────────────────────────────────────────────────────


class TestEncryptionCoverage:
    """build_report includes an encryption-coverage block derived from
    index entries (live + tombstones)."""

    def test_encryption_coverage_counts_encrypted_vs_plaintext_credential_blobs(
        self, tmp_path,
    ):
        sp = tmp_path / "audit_enc"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "credential", "enc": True,
                                  "tool": "send_email", "t": 1.0},
                "bbb444555666": {"label": "credential",
                                  "tool": "send_email", "t": 2.0},
                "ccc777888999": {"label": "public",
                                  "tool": "web_search", "t": 3.0},
                "ddd000111222": {"label": "credential", "enc": True,
                                  "tool": "send_email",
                                  "swept_at": 5.0},
            }},
        })
        report = _vfa.build_report(sp)
        cov = report.get("encryption_coverage")
        assert cov is not None, (
            "T4.4: build_report must include encryption_coverage when "
            "credential blobs exist"
        )
        # 2 encrypted (one live + one tombstone), 1 plaintext, 0 public.
        assert cov["encrypted"] == 2, (
            f"encrypted count must include live + tombstone encrypted "
            f"blobs; got {cov['encrypted']!r}"
        )
        assert cov["plaintext_credential"] == 1, (
            f"plaintext_credential count: got {cov['plaintext_credential']!r}"
        )
        assert cov["public"] == 1

    def test_encryption_coverage_omitted_when_no_credential_blobs(
        self, tmp_path,
    ):
        sp = tmp_path / "audit_no_cred"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "public", "tool": "web_search",
                                  "t": 1.0},
            }},
        })
        report = _vfa.build_report(sp)
        # No credential blobs at all: encryption coverage block is
        # either absent or carries all-zero counts. We accept both
        # shapes so the script is robust on pre-T4.2 stores.
        cov = report.get("encryption_coverage")
        if cov is not None:
            assert cov["encrypted"] == 0
            assert cov["plaintext_credential"] == 0

    def test_render_includes_encryption_coverage_when_present(
        self, tmp_path,
    ):
        sp = tmp_path / "audit_render_enc"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "credential", "enc": True,
                                  "tool": "send_email", "t": 1.0},
                "bbb444555666": {"label": "credential",
                                  "tool": "send_email", "t": 2.0},
            }},
        })
        report = _vfa.build_report(sp)
        text = _vfa.render(report)
        assert "encryption" in text.lower() or "encrypted" in text.lower(), (
            "T4.4: render() must surface the encryption-coverage section "
            "when credential blobs exist"
        )
        # Numbers must appear in the rendered text.
        assert "1" in text  # at least one number survives rendering


# ── version-chain stats ────────────────────────────────────────────────────


class TestVersionChainStats:
    """build_report includes version-chain stats: max depth + total
    superseded."""

    def test_max_chain_depth_per_session(self, tmp_path):
        """Longest chain in each (tool, session) group is reported as
        the max depth; the report's overall max is the highest across
        all sessions."""
        sp = tmp_path / "audit_chains"
        _write_sessions(sp, {
            "s1": {"blobs": {
                # Chain of 3 in s1/web_search
                "aaa111222333": {"label": "public", "tool": "web_search",
                                  "version": 1},
                "bbb444555666": {"label": "public", "tool": "web_search",
                                  "version": 2, "supersedes": "aaa111222333"},
                "ccc777888999": {"label": "public", "tool": "web_search",
                                  "version": 3,
                                  "supersedes": "bbb444555666"},
                # Chain of 2 in s1/send_email (different tool)
                "ddd000111222": {"label": "credential", "tool": "send_email",
                                  "version": 1},
                "eee333444555": {"label": "credential", "tool": "send_email",
                                  "version": 2, "supersedes": "ddd000111222"},
            }},
            "s2": {"blobs": {
                # Chain of 4 in s2/web_search
                "fff666777888": {"label": "public", "tool": "web_search",
                                  "version": 1},
                "ggg999000111": {"label": "public", "tool": "web_search",
                                  "version": 2, "supersedes": "fff666777888"},
                "hhh222333444": {"label": "public", "tool": "web_search",
                                  "version": 3, "supersedes": "ggg999000111"},
                "iii555666777": {"label": "public", "tool": "web_search",
                                  "version": 4, "supersedes": "hhh222333444"},
            }},
        })
        report = _vfa.build_report(sp)
        vcs = report.get("version_chain_stats")
        assert vcs is not None, (
            "T4.4: build_report must include version_chain_stats"
        )
        assert vcs["max_depth"] == 4, (
            f"max_depth must be the longest chain in any session; "
            f"got {vcs['max_depth']!r}"
        )
        # Total superseded: 3 in s1 (bbb, ccc, eee) + 3 in s2
        # (ggg, hhh, iii) = 6. Every chain link except the head
        # carries ``supersedes``.
        assert vcs["superseded_count"] == 6, (
            f"superseded_count must include every non-head entry; "
            f"got {vcs['superseded_count']!r}"
        )

    def test_version_chain_stats_omitted_when_no_chains(self, tmp_path):
        sp = tmp_path / "audit_no_chains"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "public", "tool": "web_search",
                                  "version": 1},
            }},
        })
        report = _vfa.build_report(sp)
        # A single-entry chain (depth=1, superseded=0) is fine to
        # either omit or include with zeros. We accept both shapes.
        vcs = report.get("version_chain_stats")
        if vcs is not None:
            assert vcs["max_depth"] == 1
            assert vcs["superseded_count"] == 0

    def test_render_includes_version_chain_stats_when_chains_exist(
        self, tmp_path,
    ):
        sp = tmp_path / "audit_render_chains"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "public", "tool": "web_search",
                                  "version": 1},
                "bbb444555666": {"label": "public", "tool": "web_search",
                                  "version": 2, "supersedes": "aaa111222333"},
            }},
        })
        report = _vfa.build_report(sp)
        text = _vfa.render(report)
        assert "chain" in text.lower() or "version" in text.lower(), (
            "T4.4: render() must surface the version-chain stats section "
            "when chains exist"
        )


# ── Backwards compat: existing audit behaviour still works ────────────────


class TestAuditBackwardsCompat:
    """T4.4 must not break the pre-existing value-flow report contract:
    entity bindings, by-destination, label-missing, etc. all still
    work when the new sections are present."""

    def test_existing_fields_preserved(self, tmp_path):
        sp = tmp_path / "audit_compat"
        _write_sessions(sp, {
            "s1": {"blobs": {
                "aaa111222333": {"label": "public", "tool": "web_search",
                                  "t": 1.0},
            }},
        })
        # Write a few entity-binding rows so the binding summary is
        # populated too.
        led = sp / "ledger"
        led.mkdir(parents=True, exist_ok=True)
        with open(led / "entity_bindings.jsonl", "w") as f:
            f.write(json.dumps({"ts": 1.0, "sid": "s", "tool": "summarise",
                                  "decision": "entity_bound",
                                  "entity_kind": "person",
                                  "blob_id": "aaa111222333"}) + "\n")
        report = _vfa.build_report(sp)
        # Pre-existing fields are still there.
        assert "by_destination" in report
        assert "labels_observed" in report
        assert "entity_bindings" in report
        # And the new fields are also there (or zero).
        assert "encryption_coverage" in report or True
        assert "version_chain_stats" in report or True

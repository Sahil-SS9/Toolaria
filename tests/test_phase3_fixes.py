"""Hermaguard Phase 3 fix tests (CRITICAL + 3 HIGH + MEDIUMs).

TDD discipline: every test in this file was written BEFORE the production
fix it pins. Each class groups the assertions for one finding from
``~/.hermes/cache/delegation/hermaguard-phase2-report.md``.

Findings covered:
    FIX-1  CRITICAL — Cross-session label divergence (label is a property
           of content, not session).
    FIX-2  HIGH     — Slice-paging bypass (range/grep/search leak raw
           credential bytes; needs masking + byte budget).
    FIX-3  HIGH     — Fail-open label lookup (enforcement ON must treat
           unresolved/None as credential).
    FIX-4  HIGH     — Bad sensitivity_tool_labels config breaks rescue
           (validate at register, harden put()).
    MEDIUM  — Bool-coercion footgun for enforcement_enabled; malformed
              credential_ttl_hours breaks sweep; _audit_summary count=0
              slice bug; docstring drift; gate ordering (full-fetch
              refusal bumps fetch_count).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import blobstore as _blobstore_mod
from blobstore import BlobStore


# ── Marker constants mirrored from passref.py for cross-file assertions ─

CREDENTIAL_REFUSE_MARKER_PREFIX = (
    "[Toolaria: credential-labelled blob "
)
CREDENTIAL_REFUSE_MARKER_SUFFIX = (
    " withheld; use range/grep slices or add destination to "
    "credential_destinations]"
)
# New markers introduced by FIX-2 (per the report's "new text" allowance).
SLICE_MASK_MARKER = "[masked:credential-shape]"
SLICE_BUDGET_MARKER_PREFIX = (
    "[Toolaria: credential slice budget exhausted for "
)
SLICE_BUDGET_MARKER_SUFFIX = (
    "; use allowlisted destinations]"
)


# ════════════════════════════════════════════════════════════════════════
# FIX-1 — CRITICAL — Cross-session label divergence
# ════════════════════════════════════════════════════════════════════════


class TestMaxLabelForBlob:
    """BlobStore._max_label_for_blob(blob_id) — content-aware label lookup.

    The label must be a property of the *content* (the blob_id is the
    SHA256 prefix, content-addressed). A blob rescued through `send_email`
    and then again through `web_search` must resolve to the highest
    sensitivity label seen across every session that holds it.
    """

    def test_max_label_for_blob_returns_highest_across_sessions(self, plugin,
                                                                  toolaria):
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Same bytes, rescued through a public tool in a different session.
        toolaria._store.put("SECRET", "web_search", session_id="B",
                             label="public")
        # Sanity: same blob_id (content-addressed).
        assert bid
        max_label = toolaria._store._max_label_for_blob(bid)
        assert max_label == "credential", (
            "cross-session divergence: blob rescued as 'credential' in "
            "session A and 'public' in session B; gate must see 'credential'"
        )

    def test_max_label_for_blob_resolves_personal_over_internal(self, plugin,
                                                                 toolaria):
        bid = toolaria._store.put("X", "t", session_id="A", label="internal")
        toolaria._store.put("X", "t", session_id="B", label="personal")
        assert toolaria._store._max_label_for_blob(bid) == "personal"

    def test_max_label_for_blob_resolves_personal_over_public(self, plugin,
                                                               toolaria):
        bid = toolaria._store.put("X", "t", session_id="A", label="public")
        toolaria._store.put("X", "t", session_id="B", label="personal")
        assert toolaria._store._max_label_for_blob(bid) == "personal"

    def test_max_label_for_blob_returns_none_when_missing_everywhere(
        self, plugin, toolaria
    ):
        # No entries anywhere.
        assert toolaria._store._max_label_for_blob("000000000000") is None

    def test_max_label_for_blob_ignores_tombstones_label_only(self, plugin,
                                                                toolaria):
        """A tombstone carries a label for audit; live entries win
        regardless of order. Tombstone labels alone should still be seen."""
        bid = toolaria._store.put("X", "t", session_id="A", label="public")
        # Tombstone it (sweep)
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["swept_at"] = time.time()
        idx["blobs"][bid]["label"] = "credential"
        toolaria._store._save_idx(idx, "A")
        assert toolaria._store._max_label_for_blob(bid) == "credential"


class TestPutNeverDowngradesLabel:
    """put() must NEVER overwrite a higher label with a lower one for the
    same blob_id. The label is content-owned, not session-owned."""

    def test_put_with_public_label_does_not_overwrite_credential_entry(
        self, plugin, toolaria
    ):
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Same bytes, different session, public label. The session-A
        # entry must remain credential (cross-session upgrade-only).
        toolaria._store.put("SECRET", "web_search", session_id="B",
                             label="public")
        idx_a = toolaria._store._load_idx("A")
        idx_b = toolaria._store._load_idx("B")
        assert idx_a["blobs"][bid]["label"] == "credential", (
            "put() must not downgrade session A's credential label when "
            "session B re-rescues identical bytes with a public label"
        )
        assert idx_b["blobs"][bid]["label"] == "credential", (
            "put() must upgrade session B's entry to the max across "
            "sessions (credential wins over public)"
        )

    def test_in_session_public_put_does_not_overwrite_credential_label(
        self, plugin, toolaria
    ):
        """Even within the SAME session, a re-rescue through a public
        tool must not downgrade the credential label — the bytes are
        credential, the session owns nothing about the label."""
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Same session, public tool, same bytes.
        toolaria._store.put("SECRET", "web_search", session_id="A",
                             label="public")
        idx = toolaria._store._load_idx("A")
        assert idx["blobs"][bid]["label"] == "credential", (
            "in-session re-rescue through a public tool must NOT "
            "downgrade a credential label"
        )

    def test_put_stamps_max_label_when_other_session_holds_higher(
        self, plugin, toolaria
    ):
        # Session B already holds it as credential (via args upgrade).
        bid = toolaria._store.put("sk-REALKEY12345678", "web_search",
                                   session_id="B",
                                   args={"data": "sk-REALKEY12345678"})
        # Now session A tries to put same bytes with explicit public.
        toolaria._store.put("sk-REALKEY12345678", "web_search",
                             session_id="A", label="public")
        idx_b = toolaria._store._load_idx("B")
        assert idx_b["blobs"][bid]["label"] == "credential"


class TestFetchUsesContentAwareLabel:
    """fetch()'s full-fetch gate must consult the content-aware label."""

    def test_full_fetch_refuses_when_other_session_has_credential_label(
        self, plugin, toolaria
    ):
        toolaria._cfg["enforcement_enabled"] = True
        # Session A marks it credential.
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Session B re-rescues same bytes via web_search (public).
        toolaria._store.put("SECRET", "web_search", session_id="B",
                             label="public")
        # Fetch from session B (the public-labeled one). Without the fix,
        # B's entry says "public" and full returns the content. With the
        # fix, the gate reads the max (credential) and refuses.
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="B")
        assert r == (f"{CREDENTIAL_REFUSE_MARKER_PREFIX}{bid}"
                     f"{CREDENTIAL_REFUSE_MARKER_SUFFIX}"), (
            f"cross-session credential gate failed; got {r!r}"
        )

    def test_full_fetch_returns_content_when_all_sessions_are_public(
        self, plugin, toolaria
    ):
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._store.put("PUBLIC", "web_search", session_id="A")
        toolaria._store.put("PUBLIC", "web_search", session_id="B")
        idx = toolaria._store._load_idx("A")
        bid = next(iter(idx["blobs"].keys()))
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="A")
        assert r == "PUBLIC"


class TestPassrefLabelResolution:
    """passref._find_label() must use _max_label_for_blob; fail-closed
    under enforcement_enabled=True."""

    def test_find_label_uses_content_aware_label(self, plugin, toolaria):
        from passref import _find_label
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        toolaria._store.put("SECRET", "web_search", session_id="B",
                             label="public")
        # Resolving for session B (the public entry) must see credential.
        assert _find_label(toolaria._store, bid, "B") == "credential"

    def test_find_label_fails_closed_when_enforcement_on_and_label_missing(
        self, plugin, toolaria
    ):
        """FIX-3 explicit: deleted/unreadable label under enforcement ON
        must be treated as credential (fail-closed)."""
        from passref import _find_label
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("X", "send_email", session_id="A",
                                   label="credential")
        idx = toolaria._store._load_idx("A")
        del idx["blobs"][bid]["label"]
        toolaria._store._save_idx(idx, "A")
        assert _find_label(toolaria._store, bid, "A") == "credential", (
            "fail-closed: missing label under enforcement ON must resolve "
            "to 'credential', not 'public'"
        )

    def test_find_label_fails_open_when_enforcement_off_for_audit(
        self, plugin, toolaria
    ):
        """FIX-3: under enforcement OFF, missing label keeps the audit-
        friendly default ('public') so the audit script can run cleanly
        against mixed-version stores."""
        from passref import _find_label
        # enforcement off (default)
        bid = toolaria._store.put("X", "send_email", session_id="A",
                                   label="credential")
        idx = toolaria._store._load_idx("A")
        del idx["blobs"][bid]["label"]
        toolaria._store._save_idx(idx, "A")
        assert _find_label(toolaria._store, bid, "A") == "public", (
            "fail-open under enforcement OFF: missing label must keep the "
            "audit-friendly 'public' default"
        )


class TestBackfillLabelsOnRegister:
    """register() must backfill labels on every existing index entry."""

    def test_backfill_stamps_label_on_existing_entry(self, plugin, toolaria):
        # Build an entry WITHOUT a label (simulating pre-T2.1 store).
        # We need to bypass put() to plant a label-less entry; do it via
        # direct index write so the test exercises backfill_labels itself.
        bid = toolaria._store.put("data", "web_search", session_id="A")
        idx = toolaria._store._load_idx("A")
        del idx["blobs"][bid]["label"]
        toolaria._store._save_idx(idx, "A")
        assert "label" not in toolaria._store._load_idx("A")["blobs"][bid]
        # Run backfill directly.
        toolaria._store.backfill_labels()
        assert toolaria._store._load_idx("A")["blobs"][bid]["label"] == "public"

    def test_backfill_is_idempotent(self, plugin, toolaria):
        bid = toolaria._store.put("data", "web_search", session_id="A")
        toolaria._store.backfill_labels()
        first = toolaria._store._load_idx("A")["blobs"][bid]["label"]
        toolaria._store.backfill_labels()
        second = toolaria._store._load_idx("A")["blobs"][bid]["label"]
        assert first == second == "public"

    def test_backfill_walks_every_session_index(self, plugin, toolaria):
        bid_a = toolaria._store.put("a", "send_email", session_id="A")
        bid_b = toolaria._store.put("b", "web_search", session_id="B")
        # Strip labels from both.
        for sid in ("A", "B"):
            idx = toolaria._store._load_idx(sid)
            for bid in idx["blobs"]:
                idx["blobs"][bid].pop("label", None)
            toolaria._store._save_idx(idx, sid)
        toolaria._store.backfill_labels()
        assert toolaria._store._load_idx("A")["blobs"][bid_a]["label"] == "personal"
        assert toolaria._store._load_idx("B")["blobs"][bid_b]["label"] == "public"

    def test_register_backfill_swallows_exceptions(self, plugin, toolaria):
        """backfill_labels() must NEVER break register() — bad entries
        are logged at WARNING but the plugin comes up."""
        bid = toolaria._store.put("data", "web_search", session_id="A")
        idx = toolaria._store._load_idx("A")
        # Plant a corrupt entry that backfill can't classify (tool=None,
        # args=None, no label). The function should default safely.
        idx["blobs"]["badbidbadbid"] = {"t": time.time(), "tool": "",
                                         "size": 0, "hash": "h" * 64}
        toolaria._store._save_idx(idx, "A")
        # No exception, regardless.
        toolaria._store.backfill_labels()


# ════════════════════════════════════════════════════════════════════════
# FIX-2 — HIGH — Slice-paging bypass (masking + byte budget)
# ════════════════════════════════════════════════════════════════════════


class TestCredentialSliceMasking:
    """range/grep/search under enforcement ON + credential label must
    MASK any line matching labels._LABEL_UPGRADE_PATTERNS, preserving
    line count / structure. Non-matching lines pass through."""

    def _enable_enforcement_and_put_cred(self, plugin, toolaria,
                                          content="secret\npublic line\n"
                                                  "sk-ABCDEFGHIJKLMNOP"):
        toolaria._cfg["enforcement_enabled"] = True
        return toolaria._store.put(content, "send_email",
                                    session_id="A", label="credential")

    def test_credential_range_masks_credential_shaped_lines(self, plugin,
                                                              toolaria):
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria)
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 100}, session_id="A")
        assert SLICE_MASK_MARKER in r, (
            f"credential-shaped line must be masked; got: {r!r}"
        )
        assert "sk-ABCDEFGHIJKLMNOP" not in r, "raw secret leaked via range mask"

    def test_credential_range_preserves_line_count(self, plugin, toolaria):
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria)
        # 3 lines: "secret", "public line", "sk-ABCDEFGHIJKLMNOP"
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 100}, session_id="A")
        # Total lines (header + body) should be 4: [lines 0..2 of 3] + 3 body.
        lines = r.splitlines()
        assert len(lines) == 4, (
            f"line count must be preserved after masking; got {len(lines)}: {lines!r}"
        )
        # The 3rd body line (index 3) is the masked marker.
        assert lines[-1] == SLICE_MASK_MARKER, (
            f"last line must be the mask marker; got {lines[-1]!r}"
        )

    def test_credential_range_passes_through_non_matching_lines(self, plugin,
                                                                  toolaria):
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria)
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 100}, session_id="A")
        assert "public line" in r, (
            "non-matching lines must pass through under slice masking"
        )

    def test_credential_grep_masks_match_lines(self, plugin, toolaria):
        # Content with two matches, the second is credential-shaped.
        content = "sk-ABCDEFGHIJKLMNOP\nbenign\nsk-ABCDEFGHIJKLMNOP"
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria,
                                                       content=content)
        r = toolaria._fetch(args={"id": bid, "mode": "grep",
                                   "pattern": "sk-"},
                             session_id="A")
        assert "sk-ABCDEFGHIJKLMNOP" not in r, (
            "raw credential shape leaked via grep under enforcement ON"
        )
        assert SLICE_MASK_MARKER in r

    def test_credential_search_masks_match_lines(self, plugin, toolaria):
        bid = self._enable_enforcement_and_put_cred(
            plugin, toolaria,
            content="sk-ABCDEFGHIJKLMNOP\nbenign text\nsk-ABCDEFGHIJKLMNOP",
        )
        # Force lexical path (no embeddings available in tests).
        from blobstore import _sem
        orig_avail = _sem.embeddings_available
        _sem.embeddings_available = lambda: False
        try:
            r = toolaria._fetch(args={"id": bid, "mode": "search",
                                       "query": "sk-ABCDEFGHIJKLMNOP"},
                                 session_id="A")
        finally:
            _sem.embeddings_available = orig_avail
        # search may return no hits (lexical exact match); what we care
        # about is: the raw key never appears in the output.
        assert "sk-ABCDEFGHIJKLMNOP" not in r, (
            f"raw secret leaked via search; got: {r!r}"
        )

    def test_credential_outline_still_structural(self, plugin, toolaria):
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria)
        r = toolaria._fetch(args={"id": bid, "mode": "outline"},
                             session_id="A")
        # Outline is structural-only; the body of lines never appears.
        assert "sk-ABCDEFGHIJKLMNOP" not in r

    def test_credential_stat_still_structural(self, plugin, toolaria):
        bid = self._enable_enforcement_and_put_cred(plugin, toolaria)
        r = toolaria._fetch(args={"id": bid, "mode": "stat"},
                             session_id="A")
        assert "blob:" in r
        assert "sk-ABCDEFGHIJKLMNOP" not in r


class TestCredentialSliceBudget:
    """Per-blob cumulative char budget for credential slices."""

    def test_credential_slice_budget_default_2000(self, plugin, toolaria):
        # Default config should expose the budget key with value 2000.
        assert toolaria._store.cfg.get("credential_slice_total_max_chars",
                                        2000) == 2000

    def test_credential_range_exhausts_budget(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        # 30 lines of 80 chars each = 2400 chars body (over 2000 budget).
        content = "\n".join("x" * 80 for _ in range(30))
        bid = toolaria._store.put(content, "send_email",
                                    session_id="A", label="credential")
        # First call: serves lines, stays under budget (2400 - cap=4000).
        # Actually fetch_max_chars caps the *output*, but the budget tracks
        # the chars the slice would have served without the cap. We test
        # the marker when the budget is exceeded regardless of cap.
        r1 = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                    "count": 30}, session_id="A")
        # Subsequent reads add to the cumulative counter; force a tiny
        # budget so we can exhaust it in a few calls.
        toolaria._store.cfg["credential_slice_total_max_chars"] = 100
        # Reset counter explicitly so we can drive it deterministically.
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["credential_served_chars"] = 0
        toolaria._store._save_idx(idx, "A")
        r2 = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                    "count": 30}, session_id="A")
        # The exact budget-exhausted marker (FIX-2 specified text).
        assert SLICE_BUDGET_MARKER_PREFIX in r2, (
            f"budget exceeded marker missing; got: {r2!r}"
        )
        assert SLICE_BUDGET_MARKER_SUFFIX in r2

    def test_credential_slice_budget_does_not_apply_when_enforcement_off(
        self, plugin, toolaria
    ):
        # enforcement off (default) → masking/budget inactive.
        toolaria._store.cfg["credential_slice_total_max_chars"] = 100
        bid = toolaria._store.put("secret\npublic line",
                                    "send_email", session_id="A",
                                    label="credential")
        # Force the counter to "exhausted" to prove it's NOT consulted
        # when enforcement is OFF.
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["credential_served_chars"] = 99999
        toolaria._store._save_idx(idx, "A")
        r = toolaria._fetch(args={"id": bid, "mode": "range", "start": 0,
                                   "count": 10}, session_id="A")
        assert SLICE_BUDGET_MARKER_PREFIX not in r, (
            "budget marker must NOT fire under enforcement OFF"
        )

    def test_credential_slice_budget_resets_on_sweep(self, plugin, toolaria):
        """The counter is cumulative 'since last sweep reset' — sweeps
        clear it so a long-running blob's budget is not exhausted by
        yesterday's reads."""
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._store.cfg["credential_ttl_hours"] = 24
        toolaria._store.cfg["credential_slice_total_max_chars"] = 50
        bid = toolaria._store.put("X " * 20, "send_email",
                                    session_id="A", label="credential")
        # Drive the counter up via direct index mutation (avoids going
        # through the slice path which would refuse past the budget).
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["credential_served_chars"] = 999
        toolaria._store._save_idx(idx, "A")
        # Force the entry to be touched by the sweep (age past 24h).
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "A")
        toolaria._store.lazy_sweep()
        # After the sweep, the entry is a tombstone — the counter is
        # gone with the live entry. Verify the counter does not survive.
        idx = toolaria._store._load_idx("A")
        live = idx["blobs"][bid].get("credential_served_chars", 0)
        assert live == 0, (
            f"sweep must reset the budget counter; got {live}"
        )


# ════════════════════════════════════════════════════════════════════════
# FIX-3 — HIGH — Fail-open label lookup
# ════════════════════════════════════════════════════════════════════════
# Covered by TestPassrefLabelResolution and the gate-ordering tests below.
# Adding an explicit fetch-side witness for clarity.


class TestFailClosedLookupFetch:
    """fetch()'s gate must treat None/missing label as credential under
    enforcement ON, and as public under enforcement OFF."""

    def test_deleted_label_refuses_full_under_enforcement_on(self, plugin,
                                                               toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        idx = toolaria._store._load_idx("A")
        del idx["blobs"][bid]["label"]
        toolaria._store._save_idx(idx, "A")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="A")
        assert CREDENTIAL_REFUSE_MARKER_PREFIX in r, (
            "fail-closed: deleted label under enforcement ON must "
            "trigger credential refusal"
        )

    def test_deleted_label_returns_content_under_enforcement_off(self, plugin,
                                                                   toolaria):
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        idx = toolaria._store._load_idx("A")
        del idx["blobs"][bid]["label"]
        toolaria._store._save_idx(idx, "A")
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="A")
        assert r == "SECRET", (
            "fail-open under enforcement OFF: deleted label must default "
            "to public so audit runs cleanly"
        )


# ════════════════════════════════════════════════════════════════════════
# FIX-4 — HIGH — Bad sensitivity_tool_labels config breaks rescue
# ════════════════════════════════════════════════════════════════════════


class TestRegisterTimeValidation:
    """A malformed sensitivity_tool_labels must fail LOUD at register,
    not silently disable rescue later."""

    def test_register_raises_on_invalid_label_value(self, plugin, toolaria,
                                                      base_cfg,
                                                      fake_ctx_cls):
        cfg = dict(base_cfg)
        cfg["sensitivity_tool_labels"] = {"send_email": "TOP-SECRET"}
        fc = fake_ctx_cls({"toolaria": cfg})
        with pytest.raises(ValueError, match="sensitivity_tool_labels"):
            toolaria.register(fc)

    def test_register_raises_on_non_dict_value(self, plugin, toolaria,
                                                 base_cfg, fake_ctx_cls):
        cfg = dict(base_cfg)
        cfg["sensitivity_tool_labels"] = ["send_email: credential"]
        fc = fake_ctx_cls({"toolaria": cfg})
        with pytest.raises(ValueError, match="sensitivity_tool_labels"):
            toolaria.register(fc)

    def test_register_raises_lists_offending_keys(self, plugin, toolaria,
                                                    base_cfg, fake_ctx_cls):
        cfg = dict(base_cfg)
        cfg["sensitivity_tool_labels"] = {
            "send_email": "credential",       # OK
            "web_search": "TOP-SECRET",      # bad
            "browser_navigate": "ULTRA",     # bad
        }
        fc = fake_ctx_cls({"toolaria": cfg})
        with pytest.raises(ValueError) as excinfo:
            toolaria.register(fc)
        msg = str(excinfo.value)
        assert "web_search" in msg
        assert "browser_navigate" in msg

    def test_register_accepts_valid_config(self, plugin, toolaria, base_cfg,
                                             fake_ctx_cls):
        cfg = dict(base_cfg)
        cfg["sensitivity_tool_labels"] = {"send_email": "credential"}
        fc = fake_ctx_cls({"toolaria": cfg})
        # Must not raise.
        toolaria.register(fc)


class TestPutFallsBackOnLabelFailure:
    """put() must not crash the rescue path if label resolution fails
    after register validation; it falls back to built-in defaults."""

    def test_put_warns_and_uses_default_when_label_parse_raises(
        self, plugin, toolaria, monkeypatch
    ):
        """Phase 3 FIX-4 hardening: if label resolution raises inside
        put() (e.g. a future bug in label_for_tool or a corrupted
        args_snapshot), the rescue must NEVER crash — fall back to
        built-in defaults with a WARNING log."""
        # Force label_for_tool to raise so the put() fallback path
        # triggers. This simulates the kind of edge case that
        # register-time validation might miss.
        from labels import label_for_tool as real_lft
        def boom(tool_name, cfg):
            raise RuntimeError("simulated label_for_tool failure")
        monkeypatch.setattr("blobstore.label_for_tool", boom)
        bid = toolaria._store.put("DATA", "send_email", session_id="A")
        entry = toolaria._store._load_idx("A")["blobs"][bid]
        # Falls back to public (the last-ditch default after the
        # built-in map is also bypassed by the simulated failure).
        assert entry["label"] in ("public", "personal", "credential"), (
            f"put() must fall back to a valid default; got {entry['label']!r}"
        )

    def test_put_survives_malformed_args(self, plugin, toolaria):
        # An args object that label_for_args cannot recurse into must
        # not crash the rescue.
        class BadArgs:
            def __repr__(self):
                raise RuntimeError("unrepr-able")

        # No crash.
        bid = toolaria._store.put("DATA", "send_email", session_id="A",
                                    args=BadArgs())
        assert bid
        assert toolaria._store._load_idx("A")["blobs"][bid]["label"] in (
            "credential", "personal", "internal", "public")


# ════════════════════════════════════════════════════════════════════════
# MEDIUMs — bool-coercion footgun, ttl eager coerce, audit clamp,
# docstring drift, gate ordering
# ════════════════════════════════════════════════════════════════════════


class TestEnforcementEnabledBoolCoercion:
    """YAML-quoted 'false' is a string; bool('false') is True. The
    enforcement check must use explicit truthy semantics."""

    def test_string_true_means_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "true"
        assert toolaria._store._enforcement_enabled() is True

    def test_string_false_means_not_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "false"
        assert toolaria._store._enforcement_enabled() is False, (
            "bool('false') is True in Python; explicit truthy set must "
            "be used so 'false' actually disables enforcement"
        )

    def test_string_zero_means_not_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "0"
        assert toolaria._store._enforcement_enabled() is False

    def test_string_off_means_not_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "off"
        assert toolaria._store._enforcement_enabled() is False

    def test_string_no_means_not_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "no"
        assert toolaria._store._enforcement_enabled() is False

    def test_string_yes_means_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "yes"
        assert toolaria._store._enforcement_enabled() is True

    def test_string_on_means_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "on"
        assert toolaria._store._enforcement_enabled() is True

    def test_string_one_means_enforced(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = "1"
        assert toolaria._store._enforcement_enabled() is True

    def test_passref_enforcement_uses_same_coercion(self, plugin, toolaria):
        from passref import _credential_enforcement_active
        toolaria._cfg["enforcement_enabled"] = "false"
        assert _credential_enforcement_active(toolaria._cfg) is False

    def test_empty_string_treated_as_false(self, plugin, toolaria):
        toolaria._cfg["enforcement_enabled"] = ""
        assert toolaria._store._enforcement_enabled() is False


class TestCredentialTtlHoursEagerCoerce:
    """A malformed credential_ttl_hours must NOT crash lazy_sweep — log a
    WARNING and fall back to the default (24h)."""

    def test_string_credential_ttl_hours_falls_back_to_default(self, plugin,
                                                                  toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._store.cfg["credential_ttl_hours"] = "24"  # quoted str
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Age past 24h, sweep must not crash.
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "A")
        # Must not raise TypeError on min(int, str).
        toolaria._store.lazy_sweep()
        entry = toolaria._store._load_idx("A")["blobs"][bid]
        assert "swept_at" in entry

    def test_none_credential_ttl_hours_falls_back_to_default(self, plugin,
                                                               toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        toolaria._store.cfg["credential_ttl_hours"] = None
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["t"] = time.time() - (25 * 3600)
        toolaria._store._save_idx(idx, "A")
        toolaria._store.lazy_sweep()
        entry = toolaria._store._load_idx("A")["blobs"][bid]
        assert "swept_at" in entry


class TestAuditSummaryCountClamp:
    """_audit_summary count=0 currently returns the full ledger mislabeled
    as 'last 0 events'. Clamp to [0..1000]; treat ≤0 as empty."""

    def test_count_zero_returns_empty_message(self, plugin, toolaria):
        # Plant one expansion so the ledger has data.
        bid = toolaria._store.put("DATA", "web_search", session_id="A")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        r = toolaria._store._audit_summary(0)
        # Should NOT show "last 0 events" — that's the bug we're fixing.
        assert "last 0 events" not in r
        # Should be friendly-empty instead.
        low = r.lower()
        assert "no data" in low or "empty" in low or "last 0" not in low

    def test_count_negative_returns_empty_message(self, plugin, toolaria):
        bid = toolaria._store.put("DATA", "web_search", session_id="A")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        r = toolaria._store._audit_summary(-5)
        assert "last -5" not in r and "last 0" not in r

    def test_count_over_1000_clamped_to_1000(self, plugin, toolaria):
        bid = toolaria._store.put("DATA", "web_search", session_id="A")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        r = toolaria._store._audit_summary(999999)
        # Must not crash; header should not advertise >1000.
        assert "last 1000" in r or "last 1,000" in r or "last 999999" not in r

    def test_count_normal_still_works(self, plugin, toolaria):
        bid = toolaria._store.put("DATA", "web_search", session_id="A")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise", args={"x": f"tla:{bid}"})
        r = toolaria._store._audit_summary(5)
        assert "summarise" in r


class TestCredentialEnforcementActiveDocstring:
    """The docstring must not claim to check the allowlist (it doesn't).
    Fix the drift; the test pins the new text."""

    def test_docstring_does_not_claim_allowlist_check(self):
        import inspect
        from passref import _credential_enforcement_active
        src = inspect.getdoc(_credential_enforcement_active) or ""
        # Old claim was "True iff enforcement_enabled is on AND the
        # allowlist is configured" — that was wrong.
        assert "AND the allowlist" not in src, (
            f"docstring still claims allowlist check; got: {src!r}"
        )


class TestGateOrderingFullFetchBeforeRefresh:
    """A refused full-fetch must NOT bump fetch_count or stamp
    first_fetch_ts — gate ordering: credential check before _refresh_blob."""

    def test_credential_full_refusal_does_not_bump_fetch_count(self, plugin,
                                                                  toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        before = toolaria._store._load_idx("A")["blobs"][bid].copy()
        r = toolaria._fetch(args={"id": bid, "mode": "full"},
                             session_id="A")
        # Marker returned.
        assert CREDENTIAL_REFUSE_MARKER_PREFIX in r
        # Counter untouched.
        after = toolaria._store._load_idx("A")["blobs"][bid]
        assert after.get("fetch_count", 0) == before.get("fetch_count", 0), (
            f"refused full-fetch must not bump fetch_count; "
            f"before={before.get('fetch_count')} after={after.get('fetch_count')}"
        )
        assert "first_fetch_ts" not in after, (
            "refused full-fetch must not stamp first_fetch_ts"
        )

    def test_credential_full_refusal_does_not_refresh_t(self, plugin,
                                                          toolaria):
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("SECRET", "send_email", session_id="A",
                                   label="credential")
        # Age to t = 0.
        idx = toolaria._store._load_idx("A")
        idx["blobs"][bid]["t"] = 0.0
        toolaria._store._save_idx(idx, "A")
        toolaria._fetch(args={"id": bid, "mode": "full"}, session_id="A")
        idx_after = toolaria._store._load_idx("A")["blobs"][bid]
        assert idx_after["t"] == 0.0, (
            f"refused full-fetch must not refresh t; got t={idx_after['t']}"
        )


# ════════════════════════════════════════════════════════════════════════
# G5 perf-guard (existing test runs against new code; add explicit
# witness that the new label logic stays under the bound).
# ════════════════════════════════════════════════════════════════════════


class TestPhase3PerfGuard:
    """The new content-aware label scan must not regress the rescue path
    beyond the existing G5 5% bound (tested loosely at <50ms/iter)."""

    def test_rescue_path_stays_within_bound(self, plugin, toolaria):
        body = "x" * (17700 - 200)
        payload = '{"url": "https://example.test/x", "body": "' + body + '"}'

        # Warm-up.
        for _ in range(3):
            b = toolaria._store.put(payload, "send_email", session_id="perf")
            toolaria._fetch(args={"id": b, "mode": "stat"}, session_id="perf")
            toolaria._store.lazy_sweep()

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            b = toolaria._store.put(payload, "send_email",
                                     session_id="perf",
                                     args={"url": "https://example.test/x",
                                           "api_key": "«redacted:sk-…»"})
            toolaria._fetch(args={"id": b, "mode": "stat"}, session_id="perf")
        elapsed = time.perf_counter() - t0
        per_iter_ms = (elapsed / iters) * 1000
        # Generous CI-tolerant bound; the actual added cost of the
        # cross-session label scan is microseconds.
        assert per_iter_ms < 50.0, (
            f"Phase 3 fix regressed rescue-path latency to "
            f"{per_iter_ms:.2f}ms/iter (G5 budget <50ms)"
        )

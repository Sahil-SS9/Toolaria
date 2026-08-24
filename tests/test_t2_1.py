"""Phase 2 data-governance tests — T2.1.

TDD discipline: every test was written before the implementation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from labels import label_for_args, label_for_tool, VALID_LABELS



# Built-in defaults that ship out of the box so a fresh install classifies
# common tool classes without any operator config.
DEFAULT_LABEL_BY_TOOL = {
    # mail/MCP read → personal
    "send_email": "personal",
    "send_mail": "personal",
    "post_email": "personal",
    "compose_email": "personal",
    "mail_send": "personal",
    "peer_send_message": "personal",
    "peer_broadcast": "personal",
    # web/browser → public
    "web_extract": "public",
    "web_search": "public",
    "browser_navigate": "public",
    "browser_snapshot": "public",
    "browser_console": "public",
    "browser_get_images": "public",
}


class TestLabelForTool:
    """label_for_tool() — map-hit, map-miss-default, and config override."""

    def test_map_hit_returns_configured_label(self):
        # Built-in default: mail-class is personal.
        assert label_for_tool("send_email", {}) == "personal"

    def test_map_hit_browser_is_public_by_default(self):
        # Built-in default: web/browser is public. Critical — the
        # false-positive budget depends on this staying public so a normal
        # browser navigate does not get the credential-grade treatment.
        for t in ("web_search", "web_extract", "browser_navigate"):
            assert label_for_tool(t, {}) == "public", (
                f"{t} must default to public; misclassification would "
                f"gate every browser result behind credential controls"
            )

    def test_map_miss_returns_default_public(self):
        # Tool not in the built-in map nor operator override: public.
        assert label_for_tool("exotic_tool_no_one_has_heard_of", {}) == "public"

    def test_operator_override_wins_over_default(self):
        cfg = {"sensitivity_tool_labels": {"send_email": "credential"}}
        # Operator can promote a normally-personal tool to credential.
        assert label_for_tool("send_email", cfg) == "credential"

    def test_operator_override_to_lower_label(self):
        cfg = {"sensitivity_tool_labels": {"web_search": "internal"}}
        # Operators can also demote; the label map is a flat override.
        assert label_for_tool("web_search", cfg) == "internal"

    def test_invalid_label_name_rejected(self):
        cfg = {"sensitivity_tool_labels": {"send_email": "TOP-SECRET"}}
        with pytest.raises(ValueError):
            label_for_tool("send_email", cfg)

    def test_non_string_tool_name_handled(self):
        # Defensive: never raise on weird input; treat as public.
        assert label_for_tool("", {}) == "public"

    def test_valid_labels_constant(self):
        # The four labels are an enum; tests may pin against it.
        assert set(VALID_LABELS) == {"credential", "personal", "internal",
                                      "public"}


class TestLabelPersistedInIndex:
    """put() stores the label in the index entry."""

    def test_index_entry_carries_label(self, plugin, toolaria):
        bid = toolaria._store.put("data", "send_email", session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert "label" in entry, (
            "T2.1: put() must persist the label on the index entry so the "
            "audit script and passref enforcement can read it"
        )
        assert entry["label"] == "personal"

    def test_unlabelled_tool_defaults_to_public(self, plugin, toolaria):
        bid = toolaria._store.put("data", "some_random_tool",
                                   session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry["label"] == "public"

    def test_label_does_not_affect_blob_id(self, plugin, toolaria):
        # Re-tagging must NOT change content addressing (D2: redaction
        # point is read-side; labels are metadata).
        c = "same-content"
        b1 = toolaria._store.put(c, "send_email", session_id="s1",
                                  label="personal")
        b2 = toolaria._store.put(c, "web_search", session_id="s1",
                                  label="public")
        assert b1 == b2


class TestLabelSurvivesSweeps:
    """Tombstone conversion must keep {label} alongside {swept_at,tool,size}.

    Regression for D3 (review finding): both _sweep_by_ttl and
    _sweep_by_size previously dropped custom fields / ignored labels.
    """

    def test_ttl_sweep_preserves_label_in_tombstone(self, plugin, toolaria):
        bid = toolaria._store.put("data", "send_email", session_id="s1")
        idx = toolaria._store._load_idx("s1")
        idx["blobs"][bid]["t"] = time.time() - 7200
        toolaria._store._save_idx(idx, "s1")
        toolaria._store.lazy_sweep()

        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"][bid]
        assert "swept_at" in entry, "sweep should have produced a tombstone"
        assert entry.get("label") == "personal", (
            "T2.1 / D3: tombstone must carry the label alongside "
            "swept_at/tool/size"
        )
        assert entry.get("tool") == "send_email"
        assert "size" in entry

    def test_size_sweep_preserves_label_in_tombstone(self, plugin, toolaria):
        # Force size-cap eviction by setting max_store_mb = 0.
        toolaria._store.cfg["max_store_mb"] = 0
        bid = toolaria._store.put("x" * 5000, "web_search",
                                   session_id="s1",
                                   label="internal")
        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        entry = idx["blobs"].get(bid)
        assert entry is not None, "size sweep should leave a tombstone"
        assert "swept_at" in entry
        assert entry.get("label") == "internal", (
            "T2.1: size-cap sweep must also preserve the label — without "
            "this, the audit script loses label info on size-evicted blobs"
        )

    def test_label_survives_size_sweep_via_tombstone_everywhere(
        self, plugin, toolaria
    ):
        """Size sweep uses _tombstone_everywhere; verify both paths.

        Regression guard: a future refactor that bypasses _tombstone_everywhere
        would silently break the audit script's label coverage.
        """
        toolaria._store.cfg["max_store_mb"] = 0
        bid = toolaria._store.put("data", "send_email", session_id="s1")
        toolaria._store.lazy_sweep()
        idx = toolaria._store._load_idx("s1")
        assert idx["blobs"][bid].get("label") == "personal"



"""Phase 3 data-governance tests — T3.3: ambiguity detection + confirmation gate.

TDD discipline: every test was written before the production code it pins.

When ``confirmation_required`` is TRUE and a single tool request's args
mention entities of multiple distinct kinds AND there is at least one
``tla:<id>`` token to expand, the middleware returns the deterministic
confirmation marker in place of the expansion — no content leaks into
the downstream tool. When ``confirmation_required`` is FALSE (the
default) the path is byte-identical to the pre-T3.3 passref.
"""
from __future__ import annotations

import json

import pytest

from passref import (
    ENTITY_CONFIRMATION_MARKER,
    ENTITY_CONFIRMATION_MARKER_PREFIX,
    ENTITY_CONFIRMATION_MARKER_KINDS,
    ENTITY_CONFIRMATION_MARKER_SEP,
    ENTITY_CONFIRMATION_MARKER_MIDDLE,
    _confirmation_marker,
)
from ledger import ledger_path


def _mw(plugin):
    return plugin[0].middleware["tool_request"][0]


def _enable_registry(toolaria, entries):
    toolaria._cfg["entity_registry"] = entries
    toolaria._cfg.pop("_entity_registry_frozen", None)


# ── Confirmation marker shape ─────────────────────────────────────────────


class TestConfirmationMarkerShape:
    """The marker is deterministic, exact, and exposed as a constant."""

    def test_marker_exact_shape(self):
        # The spec: "[Toolaria: action spans N entity kinds (<kinds>);
        # confirm target or widen entity_registry]"
        kinds = ["document", "person"]
        m = _confirmation_marker(kinds)
        assert m == (
            f"{ENTITY_CONFIRMATION_MARKER_PREFIX}2"
            f"{ENTITY_CONFIRMATION_MARKER_KINDS}document, person"
            f"{ENTITY_CONFIRMATION_MARKER_MIDDLE}"
        )

    def test_marker_kinds_sorted(self):
        # Order independent — kinds must be sorted before formatting.
        m1 = _confirmation_marker(["person", "document"])
        m2 = _confirmation_marker(["document", "person"])
        assert m1 == m2

    def test_marker_n_reflects_count(self):
        m = _confirmation_marker(["a", "b", "c"])
        assert "3 entity kinds" in m
        assert "a, b, c" in m

    def test_marker_template_is_complete(self):
        # The template constant has every placeholder used by the builder.
        assert "{n}" in ENTITY_CONFIRMATION_MARKER
        assert "{kinds}" in ENTITY_CONFIRMATION_MARKER


# ── Gate default behaviour ─────────────────────────────────────────────────


class TestGateDefaultOff:
    """``confirmation_required`` defaults to FALSE — byte-identical path."""

    def test_default_off_means_byte_identical_expansion(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        # Ambiguous args: 2 kinds + 1 token. confirmation_required off.
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        assert out is not None
        assert out["args"]["x"] == "DATA", (
            "confirmation_required=FALSE must expand normally — the gate "
            "is dormant under the default config (enforcement-OFF contract)"
        )

    def test_yaml_quoted_false_disables_gate(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        # Explicit quoted-false string — truthy semantics required.
        toolaria._cfg["confirmation_required"] = "false"
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        assert out["args"]["x"] == "DATA", (
            "a YAML-quoted \"false\" string must NOT enable the gate"
        )


# ── Gate on: ambiguity → marker ───────────────────────────────────────────


class TestAmbiguityGateFires:
    """When ON, ambiguous requests return the marker instead of expanding."""

    def test_ambiguous_returns_marker(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        assert out is not None
        assert out["args"]["x"] == (
            f"{ENTITY_CONFIRMATION_MARKER_PREFIX}2"
            f"{ENTITY_CONFIRMATION_MARKER_KINDS}person, record"
            f"{ENTITY_CONFIRMATION_MARKER_MIDDLE}"
        )

    def test_marker_replaces_token_in_nested_args(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "name", "pattern": "spec.md",
             "kind": "document", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"a": {"b": [f"tla:{bid}"]},
                                "to": "alice", "ref": "spec.md"},
                          session_id="test-s")
        expected_marker = (
            f"{ENTITY_CONFIRMATION_MARKER_PREFIX}2"
            f"{ENTITY_CONFIRMATION_MARKER_KINDS}document, person"
            f"{ENTITY_CONFIRMATION_MARKER_MIDDLE}"
        )
        assert out["args"]["a"]["b"][0] == expected_marker

    def test_ambiguous_does_not_leak_content(self, plugin, toolaria):
        # The downstream tool must never see the expanded content
        # when the gate fires.
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("TOPSECRET" * 500, "web_extract",
                                   session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        rendered = json.dumps(out)
        assert "TOPSECRET" not in rendered, (
            "gated request must NEVER include the expanded content"
        )

    def test_ambiguous_records_ambiguous_gated_in_ledger(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "ref": "PR-42",
                          "x": f"tla:{bid}"},
                    session_id="test-s")
        rows = [json.loads(l) for l in
                ledger_path(toolaria._cfg, "entity_bindings")
                .read_text().splitlines()]
        decisions = [r["decision"] for r in rows]
        assert "ambiguous_gated" in decisions, (
            "T3.3: gate-firing must log an ambiguous_gated row for the "
            "T3.4 audit surface (top ambiguous tools metric)"
        )
        gated = [r for r in rows if r["decision"] == "ambiguous_gated"][0]
        assert sorted(gated["entity_kinds"]) == ["person", "record"]
        assert gated["tool"] == "summarise"


# ── Gate non-firing cases ──────────────────────────────────────────────────


class TestGateNonFiring:
    """Single-kind args or no-token args must NOT fire the gate."""

    def test_single_kind_does_not_fire(self, plugin, toolaria):
        # Two person names — same kind → not ambiguous.
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "name", "pattern": "bob",
             "kind": "person", "sensitivity": "personal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "cc": "bob",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        assert out["args"]["x"] == "DATA", (
            "single-kind mentions (person + person) are NOT ambiguous — "
            "the gate requires distinct kinds"
        )

    def test_no_token_does_not_fire(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "note": "plain text only"},
                          session_id="test-s")
        # No token → no expansion was ever going to happen; gate must
        # not interfere (return None so the host passes args unchanged).
        assert out is None, (
            "no-token request must not be intercepted by the gate"
        )

    def test_no_entity_does_not_fire(self, plugin, toolaria):
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"x": f"tla:{bid}"},
                          session_id="test-s")
        assert out["args"]["x"] == "DATA"

    def test_disabled_passref_still_respects_gate(self, plugin, toolaria):
        # If passref_enabled=false, the middleware short-circuits to None
        # before any governor code runs — gate must not mutate that.
        _enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        toolaria._cfg["confirmation_required"] = True
        toolaria._cfg["passref_enabled"] = False
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="summarise",
                          args={"to": "alice", "ref": "PR-42",
                                "x": f"tla:{bid}"},
                          session_id="test-s")
        assert out is None


# ── Refusal marker contract unchanged ─────────────────────────────────────


class TestRefusalMarkersUnchanged:
    """The Phase 0/2 refusal markers must still appear verbatim — T3.3
    does not touch the destination-deny / credential-refusal paths.
    """

    def test_destination_deny_marker_unchanged(self, plugin, toolaria):
        from passref import _DEST_DENY_MARKER_PREFIX
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        out = _mw(plugin)(tool_name="send_email",
                          args={"x": f"tla:{bid}"},
                          session_id="test-s")
        assert out["args"]["x"].startswith(_DEST_DENY_MARKER_PREFIX)

    def test_credential_refusal_marker_unchanged(self, plugin, toolaria):
        from passref import (CREDENTIAL_REFUSE_MARKER_PREFIX,
                             CREDENTIAL_REFUSE_MARKER_SUFFIX)
        toolaria._cfg["enforcement_enabled"] = True
        bid = toolaria._store.put("X " * 50, "send_email",
                                   session_id="test-s",
                                   label="credential")
        out = _mw(plugin)(tool_name="summarise",
                          args={"x": f"tla:{bid}"},
                          session_id="test-s")
        v = out["args"]["x"]
        assert v.startswith(CREDENTIAL_REFUSE_MARKER_PREFIX)
        assert CREDENTIAL_REFUSE_MARKER_SUFFIX in v
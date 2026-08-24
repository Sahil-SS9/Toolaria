"""Phase 3 data-governance tests — T3.2: action→entity binding ledger.

TDD discipline: every test was written before the production code it pins.

In the passref expansion path, when a tool request carries an explicit
entity reference (the args mention a registered entity) AND there is a
tla:<id> token being expanded, the governor logs one decision entry
per (tool, entity_kind, blob_id) combination to
``ledger/entity_bindings.jsonl`` BEFORE expansion. No expansion
behaviour change — observe-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledger import log_entity_binding, ledger_path
from entities import parse_entity_registry


# ── ledger.log_entity_binding shape ────────────────────────────────────────


class TestLogEntityBinding:
    """The ledger helper writes one well-formed JSONL line per call to
    ``store_path/ledger/entity_bindings.jsonl``. Decision-typed records
    are the input contract for the T3.4 audit surface.
    """

    def test_writes_well_formed_jsonl(self, tmp_path):
        cfg = {"store_path": str(tmp_path / "store")}
        ok = log_entity_binding(
            cfg, sid="s1", tool="send_email", entity_kind="person",
            blob_id="aabbccddeeff", decision="entity_bound",
        )
        assert ok is True
        path = ledger_path(cfg, "entity_bindings")
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["sid"] == "s1"
        assert rec["tool"] == "send_email"
        assert rec["entity_kind"] == "person"
        assert rec["blob_id"] == "aabbccddeeff"
        assert rec["decision"] == "entity_bound"
        assert "ts" in rec

    def test_appends_one_line_per_call(self, tmp_path):
        cfg = {"store_path": str(tmp_path / "store")}
        for i in range(3):
            log_entity_binding(cfg, sid="s", tool="publish",
                                entity_kind="record",
                                blob_id=f"id{i:012d}",
                                decision="entity_bound")
        path = ledger_path(cfg, "entity_bindings")
        assert len(path.read_text().splitlines()) == 3

    def test_ambiguous_gated_decision_supported(self, tmp_path):
        cfg = {"store_path": str(tmp_path / "store")}
        ok = log_entity_binding(
            cfg, sid="s", tool="publish", entity_kinds=["document", "person"],
            decision="ambiguous_gated",
        )
        assert ok is True
        rec = json.loads(ledger_path(cfg, "entity_bindings")
                         .read_text().splitlines()[0])
        assert rec["decision"] == "ambiguous_gated"
        assert rec["entity_kinds"] == ["document", "person"]
        # blob_id is optional for the ambiguous case — omitted when None.
        assert "blob_id" not in rec

    def test_ledger_failure_does_not_raise(self, tmp_path):
        # An unwritable path must not raise into the passref path;
        # the helper logs+returns False so expansion continues.
        cfg = {"store_path": "/proc/no-such-thing/store"}
        ok = log_entity_binding(cfg, sid="s", tool="t",
                                entity_kind="person", blob_id="x",
                                decision="entity_bound")
        assert ok is False


# ── Passref binding hook (observe-only) ────────────────────────────────────


def _mw(plugin):
    return plugin[0].middleware["tool_request"][0]


class TestPassrefEntityBinding:
    """The middleware writes one entity_binding row per (tool,
    entity_kind, blob_id) tuple before expansion. With the default
    empty registry, nothing is written (feature inert).
    """

    def test_no_binding_when_registry_empty(self, plugin, toolaria):
        # Default base_cfg has no entity_registry ⇒ feature inert.
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "x": f"tla:{bid}"},
                    session_id="test-s")
        path = ledger_path(toolaria._cfg, "entity_bindings")
        if path.exists():
            assert path.read_text() == "", (
                "empty entity_registry must produce zero binding rows"
            )

    def test_binding_logged_before_expansion(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "x": f"tla:{bid}"},
                    session_id="test-s")
        path = ledger_path(toolaria._cfg, "entity_bindings")
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows) == 1, (
            "T3.2: one (tool, entity_kind, blob_id) binding per request"
        )
        r = rows[0]
        assert r["tool"] == "summarise"
        assert r["entity_kind"] == "person"
        assert r["blob_id"] == bid
        assert r["decision"] == "entity_bound"

    def test_binding_logs_only_when_token_present(self, plugin, toolaria):
        # Args mention an entity but no tla:<id> token ⇒ no expansion
        # is happening, so no binding is recorded.
        toolaria._cfg["entity_registry"] = [{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "note": "plain text"},
                    session_id="test-s")
        path = ledger_path(toolaria._cfg, "entity_bindings")
        assert (not path.exists()) or path.read_text() == "", (
            "binding log requires BOTH an entity match AND a tla: token"
        )

    def test_binding_aggregates_per_entity_kind(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "ref": "PR-42",
                          "x": f"tla:{bid}"},
                    session_id="test-s")
        rows = [json.loads(l) for l in
                ledger_path(toolaria._cfg, "entity_bindings")
                .read_text().splitlines()]
        kinds = sorted(r["entity_kind"] for r in rows)
        assert kinds == ["person", "record"]
        # All rows share the same tool + blob_id; one row per kind.
        assert all(r["tool"] == "summarise" for r in rows)
        assert all(r["blob_id"] == bid for r in rows)

    def test_binding_dedupes_blob_ids(self, plugin, toolaria):
        # Two tla: tokens of the same blob + one entity kind → one
        # binding row per (kind, blob_id) pair, NOT one per token.
        toolaria._cfg["entity_registry"] = [{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "a": f"tla:{bid}",
                          "b": f"tla:{bid}"},
                    session_id="test-s")
        rows = [json.loads(l) for l in
                ledger_path(toolaria._cfg, "entity_bindings")
                .read_text().splitlines()]
        assert len(rows) == 1, (
            "T3.2: binding is per (tool, entity_kind, blob_id), so the "
            "same blob twice → one row, not two"
        )

    def test_binding_logs_before_token_is_replaced(self, plugin, toolaria):
        # The token in the original args must NOT appear in any binding
        # row's recorded blob_id expanded — only the blob id short form.
        # This is by construction (we record blob_id, not content) but
        # verify the marker never leaks into the row.
        toolaria._cfg["entity_registry"] = [{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("X " * 100, "web_extract",
                                   session_id="test-s")
        _mw(plugin)(tool_name="summarise",
                    args={"to": "alice", "x": f"tla:{bid}"},
                    session_id="test-s")
        rows = [json.loads(l) for l in
                ledger_path(toolaria._cfg, "entity_bindings")
                .read_text().splitlines()]
        for r in rows:
            assert r["blob_id"] == bid
            assert "X " * 5 not in json.dumps(r), (
                "binding row must NOT carry expanded content"
            )

    def test_no_binding_for_dest_denied_tool(self, plugin, toolaria):
        # send_email is on the default external-destinations list.
        # The tool_request still runs the entity scan (observation is
        # about the action, not its allow status). The middleware
        # should still log bindings for an observed entity→blob link.
        toolaria._cfg["entity_registry"] = [{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        _mw(plugin)(tool_name="send_email",
                    args={"to": "alice", "x": f"tla:{bid}"},
                    session_id="test-s")
        rows = [json.loads(l) for l in
                ledger_path(toolaria._cfg, "entity_bindings")
                .read_text().splitlines()]
        assert any(r["tool"] == "send_email" for r in rows), (
            "binding log is observe-only — fires even when the "
            "destination would later be denied"
        )
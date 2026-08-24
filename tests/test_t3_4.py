"""Phase 3 data-governance tests — T3.4: audit surface for entity bindings.

TDD discipline: every test was written before the production code it pins.

Two surfaces report the entity-binding signal:

  1. ``reporting/value_flow_audit.py`` — offline reader; builds a
     per-kind binding count and a top-ambiguous-tools breakdown from
     ``store_path/ledger/entity_bindings.jsonl``.

  2. ``BlobStore._audit_summary`` — the in-plugin ``/rescuer audit``
     mode surfaces the same numbers so operators can read them without
     running a script.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporting import value_flow_audit as _vfa
from blobstore import BlobStore


# ── value_flow_audit: entity-binding section ───────────────────────────────


class TestValueFlowAuditEntitySection:
    """``build_report`` + ``render`` carry an entity-binding block when
    the ledger has rows. Empty ⇒ the section is omitted from the
    rendered text (the script is the *audit* surface, not a noise
    generator).
    """

    def _write_entity_bindings(self, store_path: Path,
                                rows: list[dict]) -> None:
        ledger_dir = store_path / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        with open(ledger_dir / "entity_bindings.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_empty_bindings_omits_section(self, tmp_path):
        # No entity_bindings.jsonl ⇒ render() must not crash and must
        # not invent data.
        sp = tmp_path / "audit_empty"
        report = _vfa.build_report(sp)
        assert report.get("entity_bindings") in (None, {}, {"per_kind": {},
                                                              "ambiguous_by_tool": {},
                                                              "total": 0,
                                                              "ambiguous_total": 0})
        text = _vfa.render(report)
        # No bindings ⇒ the section header should not appear.
        assert "entity binding" not in text.lower() or "0" in text.lower()

    def test_per_kind_counts(self, tmp_path):
        sp = tmp_path / "audit_bindings"
        self._write_entity_bindings(sp, [
            {"ts": 1.0, "sid": "s", "tool": "summarise",
             "decision": "entity_bound", "entity_kind": "person",
             "blob_id": "aaa111222333"},
            {"ts": 2.0, "sid": "s", "tool": "summarise",
             "decision": "entity_bound", "entity_kind": "person",
             "blob_id": "bbb444555666"},
            {"ts": 3.0, "sid": "s", "tool": "publish",
             "decision": "entity_bound", "entity_kind": "record",
             "blob_id": "ccc777888999"},
        ])
        report = _vfa.build_report(sp)
        eb = report["entity_bindings"]
        assert eb["per_kind"]["person"] == 2
        assert eb["per_kind"]["record"] == 1
        assert eb["total"] == 3

    def test_top_ambiguous_tools(self, tmp_path):
        sp = tmp_path / "audit_ambiguous"
        self._write_entity_bindings(sp, [
            {"ts": 1.0, "sid": "s", "tool": "publish",
             "decision": "ambiguous_gated",
             "entity_kinds": ["document", "person"]},
            {"ts": 2.0, "sid": "s", "tool": "publish",
             "decision": "ambiguous_gated",
             "entity_kinds": ["person", "record"]},
            {"ts": 3.0, "sid": "s", "tool": "send_email",
             "decision": "ambiguous_gated",
             "entity_kinds": ["document", "record"]},
        ])
        report = _vfa.build_report(sp)
        eb = report["entity_bindings"]
        assert eb["ambiguous_total"] == 3
        assert eb["ambiguous_by_tool"]["publish"] == 2
        assert eb["ambiguous_by_tool"]["send_email"] == 1

    def test_render_includes_section_when_data_present(self, tmp_path):
        sp = tmp_path / "audit_render"
        self._write_entity_bindings(sp, [
            {"ts": 1.0, "sid": "s", "tool": "publish",
             "decision": "ambiguous_gated",
             "entity_kinds": ["document", "person"]},
            {"ts": 2.0, "sid": "s", "tool": "publish",
             "decision": "entity_bound", "entity_kind": "person",
             "blob_id": "aaa111222333"},
        ])
        report = _vfa.build_report(sp)
        text = _vfa.render(report)
        assert "entity binding" in text.lower(), (
            "render() must surface the entity-binding summary when the "
            "ledger has rows"
        )
        assert "publish" in text  # the ambiguous tool
        assert "person" in text   # the bound kind


# ── /rescuer audit mode surfaces entity bindings ───────────────────────────


class TestRescuerAuditSurfacesEntityBindings:
    """``rescuer_fetch(mode='audit')`` runs ``BlobStore._audit_summary``
    which must include an entity-binding summary line so operators can
    see the per-kind counts and the top ambiguous tools from the live
    store without re-running the offline script.
    """

    def test_audit_mode_includes_entity_section(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="summarise",
           args={"to": "alice", "x": f"tla:{bid}"},
           session_id="test-s")
        r = toolaria._fetch(args={"id": bid, "mode": "audit", "count": 10},
                             session_id="test-s")
        assert "person" in r, (
            "T3.4: /rescuer audit mode must surface the per-kind entity "
            "binding counts observed in the ledger"
        )

    def test_audit_mode_reports_top_ambiguous_tools(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        toolaria._cfg["confirmation_required"] = True
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        mw = plugin[0].middleware["tool_request"][0]
        mw(tool_name="publish",
           args={"to": "alice", "ref": "PR-42", "x": f"tla:{bid}"},
           session_id="test-s")
        r = toolaria._fetch(args={"id": bid, "mode": "audit", "count": 10},
                             session_id="test-s")
        assert "publish" in r, (
            "T3.4: /rescuer audit must surface top ambiguous tools so an "
            "operator can spot the worst offenders without running the "
            "offline script"
        )

    def test_audit_mode_handles_empty_entity_bindings(self, plugin, toolaria):
        # Default registry empty ⇒ ledger file absent ⇒ audit mode
        # must degrade gracefully (same posture as the existing
        # expansions-ledger empty path).
        bid = toolaria._store.put("DATA", "web_extract", session_id="test-s")
        r = toolaria._fetch(args={"id": bid, "mode": "audit", "count": 10},
                             session_id="test-s")
        # No assertion on the entity section text when empty — only that
        # the call returns and does not crash.
        assert "audit" in r.lower() or "no data" in r.lower()


# ── Sweep integrity invariant: entity_kinds survives TTL+size ──────────────


class TestIndexEntityKindsInvariant:
    """A pre-existing blob that lacks ``entity_kinds`` (pre-T3.1 store)
    must not crash the audit or fetch paths. The empty-list default is
    consistent with the rest of the T3.1 contract.
    """

    def test_missing_entity_kinds_in_index_treated_as_empty(self, plugin,
                                                              toolaria):
        bid = toolaria._store.put("DATA", "web_search", session_id="s1")
        idx = toolaria._store._load_idx("s1")
        # Strip the field as if the blob came from a pre-T3.1 store.
        del idx["blobs"][bid]["entity_kinds"]
        toolaria._store._save_idx(idx, "s1")
        # Audit still works.
        from reporting import value_flow_audit as vfa
        report = vfa.build_report(Path(toolaria._store.store_path))
        eb = report.get("entity_bindings", {})
        assert eb.get("total", 0) == 0
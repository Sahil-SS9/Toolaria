"""Phase 3 data-governance tests — T3.1: entity registry + index storage + sweep survival.

TDD discipline: every test in this file was written before the production code it pins.
"""
from __future__ import annotations

import time
import pytest

from entities import (
    parse_entity_registry,
    extract_entities,
    distinct_entity_kinds,
    get_registry,
)


# ── Config-driven registry validation ────────────────────────────────────


class TestEntityRegistryConfig:
    """parse_entity_registry() — strict, fail-loud validation. The default
    is an empty registry so the whole governor feature is inert until an
    operator opts in (the Phase 3 'enforcement-OFF byte-identical' contract).
    """

    def test_none_or_empty_returns_empty_registry(self):
        # Default config: no entity_registry key. Feature must be inert.
        assert parse_entity_registry(None) == []
        assert parse_entity_registry([]) == []

    def test_valid_name_entry_compiles(self):
        reg = parse_entity_registry([{
            "pattern_type": "name", "pattern": "alice",
            "kind": "person", "sensitivity": "personal",
        }])
        assert len(reg) == 1
        e = reg[0]
        assert e["pattern_type"] == "name"
        assert e["pattern"] == "alice"
        assert e["kind"] == "person"
        assert e["sensitivity"] == "personal"

    def test_valid_regex_entry_compiles(self):
        reg = parse_entity_registry([{
            "pattern_type": "regex", "pattern": r"PR-\d+",
            "kind": "record", "sensitivity": "internal",
        }])
        assert reg[0]["regex"] is not None

    def test_sensitivity_defaults_to_public(self):
        reg = parse_entity_registry([{
            "pattern_type": "name", "pattern": "x",
            "kind": "record",
        }])
        assert reg[0]["sensitivity"] == "public"

    def test_must_be_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse_entity_registry({"pattern_type": "name", "pattern": "x"})

    def test_entry_must_be_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            parse_entity_registry(["not a dict"])

    def test_invalid_pattern_type_rejected(self):
        with pytest.raises(ValueError, match="pattern_type"):
            parse_entity_registry([{
                "pattern_type": "glob", "pattern": "x",
                "kind": "person", "sensitivity": "public",
            }])

    def test_empty_pattern_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_entity_registry([{
                "pattern_type": "name", "pattern": "",
                "kind": "person", "sensitivity": "public",
            }])

    def test_empty_kind_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_entity_registry([{
                "pattern_type": "name", "pattern": "x",
                "kind": "", "sensitivity": "public",
            }])

    def test_invalid_sensitivity_rejected(self):
        with pytest.raises(ValueError, match="sensitivity"):
            parse_entity_registry([{
                "pattern_type": "name", "pattern": "x",
                "kind": "person", "sensitivity": "TOP-SECRET",
            }])

    def test_invalid_regex_rejected(self):
        with pytest.raises(ValueError, match="regex"):
            parse_entity_registry([{
                "pattern_type": "regex", "pattern": "(unclosed",
                "kind": "record", "sensitivity": "public",
            }])

    def test_all_errors_reported_in_one_message(self):
        with pytest.raises(ValueError) as exc:
            parse_entity_registry([
                {"pattern_type": "name", "pattern": "",
                 "kind": "person", "sensitivity": "public"},
                {"pattern_type": "regex", "pattern": "(bad",
                 "kind": "record", "sensitivity": "public"},
            ])
        msg = str(exc.value)
        assert "non-empty" in msg
        assert "regex" in msg


# ── extract_entities matching ─────────────────────────────────────────────


class TestExtractEntities:
    """extract_entities(text, registry) — name substring + regex search.

    Recurses over dict/list so a nested args structure is covered.
    """

    def _reg(self):
        return parse_entity_registry([
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])

    def test_name_match_case_insensitive(self):
        reg = self._reg()
        kinds = distinct_entity_kinds(
            extract_entities("say hi to Alice please", reg))
        assert "person" in kinds

    def test_name_no_match_returns_empty(self):
        reg = self._reg()
        assert extract_entities("nothing here", reg) == []

    def test_regex_match_returns_value(self):
        reg = self._reg()
        ms = extract_entities("see PR-42 and PR-100", reg)
        assert any(m["value"] == "PR-42" for m in ms)
        assert any(m["value"] == "PR-100" for m in ms)

    def test_distinct_kinds_sorted(self):
        reg = self._reg()
        kinds = distinct_entity_kinds(
            extract_entities("Alice filed PR-42", reg))
        assert kinds == ["person", "record"]

    def test_recursive_scan_over_nested_args(self):
        reg = self._reg()
        args = {"to": "Alice", "meta": {"ref": "PR-7"}, "list": ["x"]}
        kinds = distinct_entity_kinds(extract_entities(args, reg))
        assert kinds == ["person", "record"]

    def test_empty_registry_returns_empty(self):
        assert extract_entities("Alice PR-42", []) == []

    def test_non_string_non_container_returns_empty(self):
        reg = self._reg()
        assert extract_entities(42, reg) == []
        assert extract_entities(None, reg) == []

    def test_match_dict_shape(self):
        reg = parse_entity_registry([{
            "pattern_type": "name", "pattern": "bob",
            "kind": "person", "sensitivity": "public",
        }])
        ms = extract_entities("Bob was here", reg)
        assert len(ms) == 1
        m = ms[0]
        assert m["kind"] == "person"
        assert m["sensitivity"] == "public"
        assert m["pattern_type"] == "name"
        assert m["value"] == "bob"


# ── Registry caching on cfg ───────────────────────────────────────────────


class TestRegistryCaching:
    """get_registry(cfg) freezes the compiled registry on cfg so the
    per-call hot path is O(1). The freeze key is the normalized entry
    list so two cfgs with identical content share the same compiled
    object.
    """

    def test_cached_on_cfg(self):
        cfg = {"entity_registry": [
            {"pattern_type": "name", "pattern": "x",
             "kind": "person", "sensitivity": "public"},
        ]}
        r1 = get_registry(cfg)
        r2 = get_registry(cfg)
        assert r1 is r2
        assert cfg["_entity_registry_frozen"] is r1

    def test_empty_cfg_yields_empty_frozen(self):
        cfg: dict = {}
        r = get_registry(cfg)
        assert r == []

    def test_invalid_config_raises_on_freeze(self):
        cfg = {"entity_registry": [{"pattern_type": "bad"}]}
        with pytest.raises(ValueError):
            get_registry(cfg)


# ── put() stores entity_kinds ─────────────────────────────────────────────


class TestEntityKindsInIndex:
    """put() must stamp entity_kinds on the index entry alongside label.

    entity_kinds is the sorted distinct list of kinds referenced by the
    blob's args (the action dimension that T3.2 binding logs share).
    Empty registry ⇒ entity_kinds is the empty list (feature inert).
    """

    def _enable_registry(self, toolaria, entries):
        toolaria._cfg["entity_registry"] = entries
        toolaria._cfg.pop("_entity_registry_frozen", None)

    def test_put_stores_entity_kinds_from_args(self, plugin, toolaria):
        self._enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ])
        bid = toolaria._store.put(
            "data", "send_email", session_id="s1",
            args={"to": "alice", "ref": "PR-42"},
        )
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert "entity_kinds" in entry, (
            "T3.1: put() must persist entity_kinds alongside label"
        )
        assert sorted(entry["entity_kinds"]) == ["person", "record"]

    def test_put_empty_registry_stores_empty_entity_kinds(self, plugin, toolaria):
        # Default empty registry.
        bid = toolaria._store.put("data", "web_search", session_id="s1",
                                   args={"to": "alice"})
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry.get("entity_kinds") == []

    def test_entity_kinds_dedup(self, plugin, toolaria):
        self._enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
        ])
        # Two person mentions → still one kind in the set.
        bid = toolaria._store.put("data", "send_email", session_id="s1",
                                   args={"to": "alice", "cc": "alice"})
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry["entity_kinds"] == ["person"]

    def test_put_with_no_args_stores_empty_entity_kinds(self, plugin, toolaria):
        self._enable_registry(toolaria, [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
        ])
        bid = toolaria._store.put("data", "web_search", session_id="s1")
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert entry.get("entity_kinds") == []

    def test_register_validates_entity_registry(self, plugin, toolaria,
                                                 fake_ctx_cls):
        # Bad operator config must fail loud at register time (FIX-4
        # posture: a broken config is the operator's problem, surfaced
        # at load, not at first rescue).
        cfg = dict(plugin[1])
        cfg["entity_registry"] = [{"pattern_type": "name", "pattern": "",
                                   "kind": "person", "sensitivity": "public"}]
        fc = fake_ctx_cls({"toolaria": cfg})
        with pytest.raises(ValueError):
            toolaria.register(fc)


# ── Sweep survival ────────────────────────────────────────────────────────


class TestEntityKindsSurviveSweeps:
    """Both sweep paths (TTL tombstone + size-cap tombstone) must
    preserve entity_kinds alongside label so the audit script can
    attribute historical flow after sweeps (D3 regression guard).
    """

    def test_ttl_sweep_preserves_entity_kinds(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [
            {"pattern_type": "name", "pattern": "alice",
             "kind": "person", "sensitivity": "personal"},
        ]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        bid = toolaria._store.put("data", "send_email", session_id="s1",
                                   args={"to": "alice"})
        idx = toolaria._store._load_idx("s1")
        idx["blobs"][bid]["t"] = time.time() - 7200
        toolaria._store._save_idx(idx, "s1")
        toolaria._store.lazy_sweep()
        entry = toolaria._store._load_idx("s1")["blobs"][bid]
        assert "swept_at" in entry
        assert entry.get("entity_kinds") == ["person"], (
            "T3.1: TTL tombstone must preserve entity_kinds alongside label"
        )

    def test_size_sweep_preserves_entity_kinds(self, plugin, toolaria):
        toolaria._cfg["entity_registry"] = [
            {"pattern_type": "regex", "pattern": r"PR-\d+",
             "kind": "record", "sensitivity": "internal"},
        ]
        toolaria._cfg.pop("_entity_registry_frozen", None)
        toolaria._store.cfg["max_store_mb"] = 0
        bid = toolaria._store.put("x" * 5000, "web_search", session_id="s1",
                                   args={"ref": "PR-42"})
        toolaria._store.lazy_sweep()
        entry = toolaria._store._load_idx("s1")["blobs"].get(bid)
        assert entry is not None
        assert "swept_at" in entry
        assert entry.get("entity_kinds") == ["record"], (
            "T3.1: size-cap tombstone must preserve entity_kinds (D3 guard)"
        )
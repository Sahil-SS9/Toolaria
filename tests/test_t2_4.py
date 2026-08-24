"""Phase 2 data-governance tests — T2.4.

TDD discipline: every test was written before the implementation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from labels import label_for_args, label_for_tool, VALID_LABELS



class TestLabelForArgsHook:
    """label_for_args() upgrades a label when arg content looks secret-shaped.

    Combined with label_for_tool, this lets _rescue decide the final label
    at put() time: a public tool that receives credential-shaped args gets
    promoted to credential for downstream enforcement.
    """

    def test_sk_prefix_in_args_upgrades_to_credential(self):
        # Public tool, but arg value contains an API-key-looking string.
        args = {"url": "https://api.example.com", "data": "sk-abcdef12345"}
        # Public base label + credential-shaped arg → credential.
        assert label_for_args(args, "public", {}) == "credential"

    def test_bearer_token_in_args_upgrades(self):
        args = {"Authorization": "Bearer xyzzy.foobar.baz12345"}
        assert label_for_args(args, "public", {}) == "credential"

    def test_private_key_marker_upgrades(self):
        args = {"pem": "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END"}
        assert label_for_args(args, "public", {}) == "credential"

    def test_no_match_keeps_base_label(self):
        args = {"url": "https://example.com", "q": "weather today"}
        assert label_for_args(args, "public", {}) == "public"

    def test_never_downgrades(self):
        """A higher base label must not be brought down by clean args."""
        args = {"url": "https://example.com"}
        assert label_for_args(args, "personal", {}) == "personal"
        assert label_for_args(args, "credential", {}) == "credential"

    def test_nested_args_are_scanned(self):
        args = {"outer": {"api_key": "sk-zzzzzzzz1234"}, "list": [{"x": 1}]}
        assert label_for_args(args, "public", {}) == "credential"

    def test_none_args_keeps_base(self):
        assert label_for_args(None, "public", {}) == "public"

    def test_empty_args_keeps_base(self):
        assert label_for_args({}, "public", {}) == "public"

    def test_read_only_does_not_mutate_args(self):
        """label_for_args must be a pure function (T2.4 matrix: read-only)."""
        args = {"data": "sk-abcdef12345", "nested": {"k": "v"}}
        snapshot = json.dumps(args, sort_keys=True)
        label_for_args(args, "public", {})
        assert json.dumps(args, sort_keys=True) == snapshot, (
            "label_for_args must not mutate the args dict in place"
        )

    def test_credential_label_consumed_by_rescue(self, plugin, toolaria):
        """End-to-end: a public tool called with sk-shaped args lands as
        credential in the index — proves the hook is wired into _rescue."""
        # Bump base_cfg so the rescue fires (need > max_result_chars).
        # Override with a fake long result + sk-shaped arg.
        r = toolaria._on_transform(
            tool_name="web_search",
            result="x" * 9000,
            args={"url": "https://x.test", "api_key": "sk-abcdef1234567"},
            session_id="s1",
        )
        assert r is not None
        idx = toolaria._store._load_idx("s1")
        entry = next(iter(idx["blobs"].values()))
        assert entry["label"] == "credential", (
            "_rescue must call label_for_args so the secret-shaped arg "
            "promotes the label even when the tool itself is public"
        )


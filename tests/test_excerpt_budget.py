"""Exact-budget contract tests at the rescue-handle level (2026-08-29
takeover of PR #4): the handle's preview description must match what the
excerpt actually carries, and excerpt_max_chars is validated at register
time."""
import pytest

from test_excerpt import _oversized_html


def test_truncated_handle_describes_budget(plugin, toolaria):
    """When the exact budget truncates the excerpt, the handle says so
    instead of advertising line counts that no longer apply."""
    fc, cfg = plugin
    cfg["excerpt_max_chars"] = 2_000
    toolaria._cfg.clear()
    toolaria._cfg.update(cfg)
    if toolaria._store is None:
        toolaria._store = toolaria.BlobStore({"store_path": cfg["store_path"]})
    handle = toolaria._on_transform(tool_name="web_extract",
                                    result=_oversized_html())
    assert handle is not None, "store failed to persist the blob"
    assert "truncated" in handle, handle[:200]
    assert "Preview (budget 2000 chars, truncated)" in handle


def test_untruncated_handle_keeps_classic_description(plugin, toolaria):
    fc, cfg = plugin
    cfg["excerpt_max_chars"] = 8_000
    toolaria._cfg.clear()
    toolaria._cfg.update(cfg)
    if toolaria._store is None:
        toolaria._store = toolaria.BlobStore({"store_path": cfg["store_path"]})
    handle = toolaria._on_transform(tool_name="web_extract",
                                    result="small result\n" + "x" * 9_000)
    assert handle is not None
    assert "Preview (first 10 / last 5 lines)" in handle


def test_rescue_handle_excerpt_itself_within_budget(plugin, toolaria):
    """End-to-end: the excerpt block inside the handle never exceeds the
    configured cap, for any payload shape."""
    fc, cfg = plugin
    cfg["excerpt_max_chars"] = 2_000
    toolaria._cfg.clear()
    toolaria._cfg.update(cfg)
    if toolaria._store is None:
        toolaria._store = toolaria.BlobStore({"store_path": cfg["store_path"]})
    big_json = '{"k": "' + "z" * 200_000 + '"}'
    for result in (_oversized_html(), big_json, "w" * 100_000):
        handle = toolaria._on_transform(tool_name="web_extract", result=result)
        assert handle is not None
        # INVARIANT: handle length <= header/appendix + excerpt budget.
        # Extract the excerpt block between the preview line and the
        # "Retrieve more" appendix and check it directly against the cap.
        preview_idx = handle.find(":\n")
        app_idx = handle.find("Retrieve more with rescuer_fetch")
        assert preview_idx != -1 and app_idx != -1, handle[:200]
        excerpt_block = handle[preview_idx + 2:app_idx].rstrip("\n")
        assert len(excerpt_block) <= 2_000, (
            f"excerpt block len {len(excerpt_block)} exceeds 2000")
        # Fixed overhead (header, preview line, fetch appendix, tla tail)
        # measured at ~730 chars; 800 is the guard against runaway growth.
        assert len(handle) <= 2_000 + 800, f"handle len {len(handle)}"


def test_excerpt_max_chars_validated_at_register(fake_ctx_cls, toolaria,
                                                 base_cfg):
    fc = fake_ctx_cls({"toolaria": dict(base_cfg, excerpt_max_chars=5)})
    with pytest.raises(ValueError, match="excerpt_max_chars"):
        toolaria.register(fc)
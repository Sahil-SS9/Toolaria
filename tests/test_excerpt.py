"""Tests for Toolaria: type detection and excerpt building."""
import json

from excerpt import build_excerpt, detect_type


CFG = {
    "head_lines": 10, "tail_lines": 5,
    "json_head_items": 3, "json_tail_items": 1,
    "error_line_patterns": [], "excerpt_max_chars": 8000,
}


def test_json_tail_shown_by_item_count():
    """Tail appears when the container has more items than head+tail."""
    raw = json.dumps([{"i": i} for i in range(10)])  # 10 > 3 + 1
    ex = build_excerpt(raw, "json", CFG)
    assert "--- tail ---" in ex


def test_json_tail_hidden_when_small():
    raw = json.dumps([{"i": i} for i in range(3)])  # 3 <= 3 + 1
    ex = build_excerpt(raw, "json", CFG)
    assert "--- tail ---" not in ex


def test_error_patterns_respect_config_change():
    """Changing error_line_patterns takes effect (no stale global cache)."""
    raw = "\n".join(["alpha"] * 20 + ["BOOM happened"] + ["omega"] * 20)
    cfg1 = dict(CFG, error_line_patterns=["nomatch"])
    ex1 = build_excerpt(raw, "text", cfg1)
    assert "BOOM" not in ex1.split("--- error lines ---")[-1] or \
        "--- error lines ---" not in ex1

    cfg2 = dict(CFG, error_line_patterns=["BOOM"])
    ex2 = build_excerpt(raw, "text", cfg2)
    assert "--- error lines ---" in ex2
    assert "BOOM happened" in ex2


def test_text_excerpt_cap_configurable():
    raw = "z" * 20000  # single line, hits the short-content path
    ex = build_excerpt(raw, "text", dict(CFG, excerpt_max_chars=1000))
    assert len(ex) < 1200


def test_detect_json_array():
    kind, meta = detect_type(json.dumps([1, 2, 3]))
    assert kind == "json"
    assert "array[3]" in meta


def test_detect_html():
    raw = "<!DOCTYPE html>\n<html><body>hi</body></html>"
    kind, _ = detect_type(raw)
    assert kind == "html"


def test_detect_binary():
    kind, _ = detect_type(b"\x00\x01\x02" * 100)
    assert kind == "binary"


def test_html_excerpt_keeps_structure():
    raw = "<!DOCTYPE html>\n<html>\n<body>\n" + "<!--pad-->\n" * 5000 + "</body>\n</html>"
    ex = build_excerpt(raw, "html", CFG)
    assert "<body>" in ex or "<html>" in ex


def test_html_long_lines_capped_per_line():
    # Regression: minified/long-line HTML produced a 60k+ char handle because
    # head/tail lines were joined uncapped. Each line must respect the same
    # 500-char cap used for anchor/error lines. Needs >hl+tl lines to take
    # the head/tail path (the short-content path is budget-capped instead).
    raw = "<html><body>\n" + "\n".join("<div>" + "x" * 20000 + "</div>" for _ in range(80)) + "\n</body></html>"
    ex = build_excerpt(raw, "html", CFG)
    for line in ex.splitlines():
        assert len(line) <= 500, f"line exceeds cap: {len(line)}"


def test_excerpt_hard_budget_enforced():
    # Aggregate budget: many individually-capped lines can still sum past
    # excerpt_max_chars; the assembled excerpt must be truncated.
    body_line = "y" * 480
    raw = "<html><body>\n" + "\n".join(f"<p>{body_line}</p>" for _ in range(80)) + "\n</body></html>"
    ex = build_excerpt(raw, "html", dict(CFG, excerpt_max_chars=2000))
    assert len(ex) <= 2400, f"excerpt blew budget: {len(ex)}"
    assert "truncated to excerpt_max_chars" in ex

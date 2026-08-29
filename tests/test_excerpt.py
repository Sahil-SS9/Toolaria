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


# ---------------------------------------------------------------------------
# Exact-budget contract (2026-08-29 takeover): excerpt_max_chars is an EXACT
# cap on the assembled excerpt including the truncation marker; one shared
# seam for every payload kind; marker counts inside the cap; tail and
# promoted anchors survive truncation; preview description stays honest.


def _oversized_html(n_lines=200, line_len=480):
    body = "y" * line_len
    return "<html><body>\n" + "\n".join(f"<p>{body}</p>" for _ in range(n_lines)) + "\n</body></html>"


def test_excerpt_exact_budget_html():
    raw = _oversized_html()
    for cap in (200, 2_000, 8_000):
        ex = build_excerpt(raw, "html", dict(CFG, excerpt_max_chars=cap))
        assert len(ex) <= cap, f"cap {cap}: excerpt blew budget: {len(ex)}"
        if ex.endswith("[... excerpt truncated to excerpt_max_chars]"):
            assert len(ex) > cap - 100, (
                f"cap {cap}: truncated excerpt wastes budget: {len(ex)}")


def test_excerpt_exact_budget_json_with_large_values():
    # Regression: the JSON path returned early and bypassed any cap —
    # a 200k-char JSON produced a 140k-char excerpt (2026-08-26 review).
    raw = '{"k": "' + "z" * 500_000 + '"}'
    for cap in (2_000, 8_000):
        ex = build_excerpt(raw, "json", dict(CFG, excerpt_max_chars=cap))
        assert len(ex) <= cap, f"cap {cap}: JSON excerpt blew budget: {len(ex)}"


def test_excerpt_exact_budget_json_array():
    raw = json.dumps(["z" * 30_000] * 40)
    ex = build_excerpt(raw, "json", dict(CFG, excerpt_max_chars=3_000))
    assert len(ex) <= 3_000, f"JSON array excerpt blew budget: {len(ex)}"


def test_excerpt_exact_budget_text_short_content_path():
    # <= head+tail lines rides the raw-passthrough branch; it must be
    # budget-exact too.
    raw = "w" * 100_000
    ex = build_excerpt(raw, "text", dict(CFG, excerpt_max_chars=1_500))
    assert len(ex) <= 1_500, f"text short-path blew budget: {len(ex)}"


def test_truncation_marker_counts_inside_cap():
    for cap in (200, 2_000):
        ex = build_excerpt(_oversized_html(), "html", dict(CFG, excerpt_max_chars=cap))
        assert ex.endswith("[... excerpt truncated to excerpt_max_chars]"), ex[-80:]
        assert len(ex) <= cap


def test_tiny_and_invalid_caps_degrade_safely():
    raw = _oversized_html()
    # Below-minimum caps clamp to the minimum (200) — never zero-length.
    for bad in (0, -5, 10, 199):
        ex = build_excerpt(raw, "html", dict(CFG, excerpt_max_chars=bad))
        assert len(ex) <= 200, f"cap {bad}: len {len(ex)}"
        assert ex, f"cap {bad}: empty excerpt"
    # Malformed values fall back to the default cap path, not a crash.
    ex = build_excerpt(raw, "html", dict(CFG, excerpt_max_chars="oops"))
    assert len(ex) <= 8_000


def test_tail_survives_truncation():
    # A fat head must not push the tail out of the output (2026-08-26
    # finding: prefix slicing deleted the tail while claiming it existed).
    fat_line = "h" * 480
    raw = ("<html><body>\n" + "\n".join(fat_line for _ in range(120))
           + "\nUNIQUE-TAIL-SENTINEL-9f3a\n</body></html>")
    ex = build_excerpt(raw, "html", dict(CFG, excerpt_max_chars=2_000))
    assert "UNIQUE-TAIL-SENTINEL-9f3a" in ex, "tail lost under truncation"


def test_promoted_anchors_survive_truncation():
    fat_line = "q" * 480
    raw = ("\n".join(fat_line for _ in range(120))
           + "\nFATAL: disk full on /var\n" + "\n".join(fat_line for _ in range(120)))
    cfg = dict(CFG, excerpt_max_chars=1_500,
               anchor_patterns={"error": ["FATAL"]})
    ex = build_excerpt(raw, "html", cfg)
    assert "FATAL: disk full on /var" in ex, "promoted anchor lost under truncation"

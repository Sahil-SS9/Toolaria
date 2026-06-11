"""Tests for Toolaria v2: chunking and the structural outline."""
import json

from chunking import chunk_lines
from index import build_outline, render_outline


# ── chunking ───────────────────────────────────────────────────────────────


def test_chunk_empty():
    assert chunk_lines("") == []


def test_chunk_covers_all_lines():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_lines(text, target_chars=200, overlap_lines=2)
    assert chunks[0].start_line == 0
    assert chunks[-1].end_line == 199
    # contiguous-with-overlap: each chunk starts at or before the previous end
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line <= a.end_line + 1


def test_chunk_overlap_present():
    text = "\n".join(f"line {i}" for i in range(60))
    chunks = chunk_lines(text, target_chars=120, overlap_lines=2)
    assert len(chunks) >= 2
    # consecutive chunks share lines
    assert chunks[1].start_line < chunks[0].end_line


def test_chunk_long_single_line_is_own_chunk():
    text = "x" * 5000 + "\nshort"
    chunks = chunk_lines(text, target_chars=1000)
    assert chunks[0].start_line == 0 and chunks[0].end_line == 0


def test_chunk_terminates_tiny_target():
    text = "\n".join(str(i) for i in range(50))
    chunks = chunk_lines(text, target_chars=1, overlap_lines=2)
    assert chunks[-1].end_line == 49


# ── JSON outline ────────────────────────────────────────────────────────────


def test_json_array_of_objects_numeric_stats():
    rows = [{"id": i, "price": i * 2.0, "name": f"x{i}"} for i in range(100)]
    outline = build_outline(json.dumps(rows), "json", {})
    assert outline["root"] == "array"
    assert outline["length"] == 100
    cols = {c["column"]: c for c in outline["element"]["columns"]}
    assert cols["price"]["stats"]["min"] == 0.0
    assert cols["price"]["stats"]["max"] == 198.0
    assert cols["price"]["stats"]["count"] == 100
    rendered = render_outline(outline)
    assert "price" in rendered and "min=" in rendered


def test_json_object_keys():
    obj = {"name": "test", "items": [1, 2, 3], "meta": {"a": 1}}
    outline = build_outline(json.dumps(obj), "json", {})
    assert outline["root"] == "object"
    keys = {k["key"] for k in outline["keys"]}
    assert {"name", "items", "meta"} <= keys
    rendered = render_outline(outline)
    assert "items" in rendered


def test_json_key_cap():
    obj = {f"k{i}": i for i in range(100)}
    outline = build_outline(json.dumps(obj), "json", {"outline_json_max_keys": 10})
    assert any("more keys" in k["key"] for k in outline["keys"])


# ── HTML outline ────────────────────────────────────────────────────────────


def test_html_headings_with_lines():
    html = (
        "<html><body>\n"
        "<h1>Title</h1>\n"
        "padding\npadding\n"
        "<h2>Section A</h2>\n"
        "<h2>Section B</h2>\n"
        "</body></html>"
    )
    outline = build_outline(html, "html", {})
    headings = outline["headings"]
    assert headings[0]["text"] == "Title" and headings[0]["level"] == 1
    assert any(h["text"] == "Section A" for h in headings)
    # line numbers should be increasing and plausible
    assert headings[1]["line"] > headings[0]["line"]
    rendered = render_outline(outline)
    assert "Section A" in rendered and "L" in rendered


# ── text/log outline ────────────────────────────────────────────────────────


def test_text_error_clusters():
    lines = (["info: starting"] * 5
             + ["ERROR: boom"] * 3
             + ["warning: careful"] * 2
             + ["all good"] * 10)
    outline = build_outline("\n".join(lines), "text", {})
    signals = {c["signal"]: c for c in outline["clusters"]}
    assert signals["error"]["count"] == 3
    assert signals["warning"]["count"] == 2
    assert signals["error"]["first_line"] == 5
    rendered = render_outline(outline)
    assert "error x3" in rendered


def test_text_no_signals():
    outline = build_outline("just\nplain\ntext", "text", {})
    assert outline["clusters"] == []
    assert "no error" in render_outline(outline)


def test_outline_never_raises_on_garbage():
    # malformed JSON falls through to text handling, not an exception
    outline = build_outline("{not valid json", "json", {})
    assert "error" in outline or outline["kind"] == "json"
